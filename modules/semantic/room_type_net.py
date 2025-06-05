import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as fn
import torchvision.transforms as T

from transformers import pipeline
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

##############################################################################
# 1) Cross Attention Module (single-head)
##############################################################################
class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        """
        embed_dim: feature dimension (e.g. 64)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(p=0.0)

    def forward(self, queries, keys, values, attn_mask=None):
        """
        queries: (N, num_queries, embed_dim)
        keys:    (N, spatial, embed_dim)
        values:  (N, spatial, embed_dim)
        attn_mask: (N, num_queries, spatial) or None
        Returns:
          output: (N, num_queries, embed_dim)
          attn:   (N, num_queries, spatial) attention weights
        """
        Q = self.query_proj(queries)  # (N, num_queries, d)
        K = self.key_proj(keys)       # (N, spatial, d)
        V = self.value_proj(values)   # (N, spatial, d)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.embed_dim ** 0.5)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out = self.dropout(torch.matmul(attn, V))
        return out, attn

##############################################################################
# 2) Feature extractor using Mask2Former with cross attention (for room type)
##############################################################################
class semantic_feature_mask2former(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        # Load Mask2Former (using the small variant)
        self.pipe = pipeline(
            "image-segmentation",
            model="facebook/mask2former-swin-small-ade-semantic",
            device=0
        )
        self.mask2former_model: Mask2FormerForUniversalSegmentation = self.pipe.model
        self.mask2former_processor: Mask2FormerImageProcessor = self.pipe.image_processor

        # Freeze all Mask2Former parameters
        for name, param in self.mask2former_model.named_parameters():
            param.requires_grad = False

        # The encoder produces typically 768-dim features.
        self.backbone_out_channels = 768

        # Reduce 768 -> embed_dim via a simple 1x1 conv block
        self.conv = nn.Conv2d(
            self.backbone_out_channels,
            embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)

        # Positional embedding: create an embed_dim-dim embedding from normalized (x,y)
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.Tanh()
        )

        # Use a single learnable query token for room type prediction.
        self.num_tokens = 1
        self.queries = nn.Parameter(torch.randn(self.num_tokens, embed_dim))

        # Cross attention module (single-head)
        self.cross_attn = CrossAttention(embed_dim=embed_dim)

    def forward(self, x, mask=None):
        """
        x: (N,3,H,W)
        mask: (N,H,W) or None
        Returns:
          token: (N,1,embed_dim) -- token for room prediction
          attn:  attention weights from cross attention (for debugging)
        """
        N, C, H, W = x.shape

        # Convert tensor images to PIL for the HF processor.
        pil_images = [T.ToPILImage()(x[i].cpu()) for i in range(N)]
        inputs = self.mask2former_processor(images=pil_images, return_tensors="pt")
        for k, v in inputs.items():
            inputs[k] = v.to(x.device)

        # Run the frozen Mask2Former to get encoder features: (N,768,h',w')
        with torch.no_grad():
            outputs = self.mask2former_model(**inputs, output_hidden_states=True)
        encoder_feature = outputs.encoder_last_hidden_state  # (N,768,h',w')

        # 1x1 Conv -> BN -> ReLU: (N,embed_dim,h',w')
        x_features = self.conv(encoder_feature)
        x_features = self.bn(x_features)
        x_features = self.relu(x_features)
        fH, fW = x_features.shape[2], x_features.shape[3]

        # Flatten spatial dimensions: (N,embed_dim, fH*fW) -> (N, fH*fW, embed_dim)
        x_flat = x_features.view(N, -1, x_features.shape[1]).contiguous()

        # Create a normalized (x,y) grid for the feature map
        grid_y = torch.linspace(-0.5, 0.5, fH, device=x.device)
        grid_x = torch.linspace(-0.5, 0.5, fW, device=x.device)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
        pos_coords = torch.stack((grid_x, grid_y), dim=-1).view(-1, 2)  # (fH*fW,2)

        # Compute positional embeddings and add to the flattened features
        pos_emb = self.pos_mlp(pos_coords)  # (fH*fW,embed_dim)
        pos_emb = pos_emb.unsqueeze(0).expand(N, -1, -1)  # (N, fH*fW, embed_dim)
        x_flat = x_flat + pos_emb

        # Prepare learnable query: (N,1,embed_dim)
        queries = self.queries.unsqueeze(0).expand(N, -1, -1)

        # If a mask is provided, downsample and create an attention mask.
        attn_mask = None
        if mask is not None:
            mask_down = fn.resize(mask, (fH, fW), interpolation=fn.InterpolationMode.NEAREST).bool()
            mask_down = ~mask_down  # True = ignore
            mask_down = mask_down.view(N, -1)
            attn_mask = mask_down.unsqueeze(1).expand(-1, self.num_tokens, -1)

        # Cross attention: query attends over the flattened keys/values.
        token, attn = self.cross_attn(queries, x_flat, x_flat, attn_mask=attn_mask)
        return token, attn

##############################################################################
# 3) Minimal Self-Attention Block (single-head) for the room branch
##############################################################################
class _SingleHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        # x: (N,L,embed_dim)
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.embed_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        return out

class _SingleHeadSelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim=64, ffn_multiplier=6):
        """
        A minimal self-attention block with residual connections and an FFN.
        """
        super().__init__()
        self.attn_layer = _SingleHeadSelfAttention(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        hidden_dim = embed_dim * ffn_multiplier
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x_attn = self.attn_layer(x)
        x = x + x_attn
        x = self.ln1(x)
        x_ffn = self.ffn(x)
        x = x + x_ffn
        x = self.ln2(x)
        return x

##############################################################################
# 4) Room Branch for Room Type Prediction
##############################################################################
class RoomBranch(nn.Module):
    def __init__(self, num_room_types, embed_dim=64):
        """
        Processes the single room token.
        """
        super().__init__()
        # Self-attention block for the room token.
        self.self_attn = _SingleHeadSelfAttentionBlock(embed_dim=embed_dim, ffn_multiplier=6)
        # MLP for room classification with an extra hidden layer to increase parameter count.
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(256, num_room_types)
        )

    def forward(self, x):
        # x: (N,embed_dim) -> unsqueeze to (N,1,embed_dim)
        x = x.unsqueeze(1)
        x = self.self_attn(x)
        x = x.squeeze(1)
        out = self.mlp(x)
        return out

##############################################################################
# 5) Final Network (Room Type Prediction Only)
##############################################################################
class room_type_net(nn.Module):
    def __init__(self, num_room_types, embed_dim=64):
        """
        num_room_types: number of room classes.
        """
        super().__init__()
        # Feature extractor: returns token (N,1,embed_dim)
        self.semantic_feature = semantic_feature_mask2former(embed_dim=embed_dim)
        # Room branch
        self.room_branch = RoomBranch(num_room_types, embed_dim=embed_dim)
        # self.print_trainable_parameters()

    def forward(self, x, mask=None):
        """
        x: (N,3,H,W)
        mask: (N,H,W) or None
        Returns:
          room_logits: (N,num_room_types)
          attn:        attention weights from cross attention (for debugging)
        """
        token, attn = self.semantic_feature(x, mask=mask)
        room_logits = self.room_branch(token.squeeze(1))
        return room_logits, attn

    def print_trainable_parameters(self):
        """
        Print each parameter's name, shape, and whether it's trainable.
        Then print total vs. trainable param counts.
        """
        total_params = 0
        trainable_params = 0

        print("\n=== Trainable Parameter Check ===")
        for name, param in self.named_parameters():
            p_count = param.numel()
            total_params += p_count
            if param.requires_grad:
                trainable_params += p_count
                print(f"[TRAINABLE] {name:<60} shape={list(param.shape)} count={p_count}")
        print(f"\nTotal params:      {total_params}")
        print(f"Trainable params:  {trainable_params}")
        print("=================================\n")
