import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as fn
import torchvision.transforms as T

from transformers import pipeline
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

##############################################################################
# 1) A simpler ConvBnReLU with kernel_size=1 by default
##############################################################################
class Conv1x1BnReLU(nn.Module):
    """
    Simple 1x1 Conv + BN + ReLU block.
    Using kernel_size=1 drastically reduces parameters 
    compared to a 3x3.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Attention(nn.Module):
    """
    Single-head dot-product attention
    """
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, attn_mask=None):
        # query: (N, Q_len, d)
        # key:   (N, K_len, d)
        # value: (N, K_len, d)
        d = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (d ** 0.5)  # (N, Q_len, K_len)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)  # (N, Q_len, K_len)
        output = torch.matmul(attn, value)  # (N, Q_len, d)
        return output, attn


##############################################################################
# 2) Mask2Former-based feature extractor for Swin-Small (frozen),
#    but we do 768->48 with a 1x1 conv to cut down on trainable params.
##############################################################################
class semantic_feature_mask2former(nn.Module):
    def __init__(self):
        super().__init__()

        # Load Mask2Former Swin-Small (frozen)
        self.pipe = pipeline(
            "image-segmentation",
            model="facebook/mask2former-swin-small-ade-semantic"
        )
        self.mask2former_model: Mask2FormerForUniversalSegmentation = self.pipe.model
        self.mask2former_processor: Mask2FormerImageProcessor = self.pipe.image_processor

        # Freeze all Mask2Former parameters
        for name, param in self.mask2former_model.named_parameters():
            param.requires_grad = False

        # Typically 768 for Swin-Small final hidden dim
        self.backbone_out_channels = 768

        # Map 768 -> 48 with a 1x1 conv
        self.conv = Conv1x1BnReLU(
            in_channels=self.backbone_out_channels,
            out_channels=48
        )

        # Positional MLPs: 2D/1D => 16 dims each
        self.pos_mlp_2d = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh()
        )
        self.pos_mlp_1d = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh()
        )

        # Q/K/V input = 48 + 16 = 64 -> output=48
        self.q_proj = nn.Linear(64, 48, bias=False)
        self.k_proj = nn.Linear(64, 48, bias=False)
        self.v_proj = nn.Linear(64, 48, bias=False)

        self.attn = Attention()
        self.dropout_attn = nn.Dropout(p=0)

        # We'll upsample horizontally from ~12 -> 40
        self.target_width = 40

    def forward(self, x, mask=None):
        """
        x: (N,3,H,W)
        mask: (N,H,W) or None
        Return:
          x_out: (N,40,48)
          attn_w: attention weights
        """
        N, C, H, W = x.shape

        # Convert to PIL for the HF image processor
        pil_images = [T.ToPILImage()(x[i].cpu()) for i in range(N)]
        inputs = self.mask2former_processor(images=pil_images, return_tensors="pt")
        for k, v in inputs.items():
            inputs[k] = v.to(x.device)

        # Frozen Mask2Former forward
        with torch.no_grad():
            outputs = self.mask2former_model(**inputs, output_hidden_states=True)
        encoder_feature = outputs.encoder_last_hidden_state  # (N,768,h,w)

        # 1) 1x1 Conv => (N,48,h,w)
        x_features = self.conv(encoder_feature)
        fH, fW = x_features.shape[2], x_features.shape[3]

        # 2) Summarize vertically => query (N,fW,48)
        query = x_features.mean(dim=2)  # (N,48,fW)
        query = query.permute(0, 2, 1)  # (N,fW,48)

        # Flatten => (N,fH*fW,48)
        x_flat = x_features.view(N, 48, -1).permute(0, 2, 1)

        # 3) Positional encoding
        #   a) 2D for key/value
        pos_x = torch.linspace(0, 1, fW, device=x.device) - 0.5
        pos_y = torch.linspace(0, 1, fH, device=x.device) - 0.5
        pos_grid_x, pos_grid_y = torch.meshgrid(pos_x, pos_y, indexing="ij")
        pos_grid_2d = torch.stack((pos_grid_x, pos_grid_y), dim=-1).reshape(-1, 2)  # (fH*fW,2)

        pos_enc_2d = self.pos_mlp_2d(pos_grid_2d)  # => (fH*fW,16)
        pos_enc_2d = pos_enc_2d.unsqueeze(0).expand(N, -1, -1)  # => (N,fH*fW,16)

        x_cat = torch.cat((x_flat, pos_enc_2d), dim=-1)  # => (N,fH*fW,64)

        #   b) 1D for query
        pos_v = torch.linspace(0, 1, fW, device=x.device) - 0.5
        pos_v = pos_v.unsqueeze(-1)  # => (fW,1)
        pos_enc_1d = self.pos_mlp_1d(pos_v)  # => (fW,16)
        pos_enc_1d = pos_enc_1d.unsqueeze(0).expand(N, -1, -1)  # => (N,fW,16)

        query_cat = torch.cat((query, pos_enc_1d), dim=-1)  # => (N,fW,64)

        # 4) Q/K/V + dropout
        Q = self.dropout_attn(self.q_proj(query_cat))  # (N,fW,48)
        K = self.dropout_attn(self.k_proj(x_cat))      # (N,fH*fW,48)
        V = self.dropout_attn(self.v_proj(x_cat))      # (N,fH*fW,48)

        # 5) Construct attention mask if provided
        attn_mask = None
        if mask is not None:
            # Downsample mask to h,w
            mask_down = fn.resize(mask, (fH, fW), fn.InterpolationMode.NEAREST).bool()
            mask_down = torch.logical_not(mask_down)  # True -> ignore
            mask_down = mask_down.view(N, -1).unsqueeze(1).repeat(1, fW, 1)
            attn_mask = mask_down

        # Single-head attention => (N,fW,48)
        x_out, attn_w = self.attn(Q, K, V, attn_mask=attn_mask)

        # 6) Upsample width from fW->40
        if x_out.shape[1] != self.target_width:
            x_out_ups = x_out.permute(0, 2, 1).unsqueeze(-2)  # (N,48,1,fW)
            x_out_ups = F.interpolate(x_out_ups, size=(1, self.target_width), mode="nearest")
            x_out = x_out_ups.squeeze(-2).permute(0, 2, 1)    # => (N,40,48)

        return x_out, attn_w


##############################################################################
# 3) A main network with a small MLP for classification:
#    (48 -> 64 -> num_classes). This should be ~50k trainable params total.
##############################################################################
class semantic_net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        # Frozen Mask2Former + 1x1 conv => (N,40,48)
        self.semantic_feature = semantic_feature_mask2former()

        # Small MLP classifier
        # (48 -> 64 -> num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(48, 64, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0),
            nn.Linear(64, self.num_classes, bias=True)
        )

        # Debug info
        self.print_trainable_parameters()

    def forward(self, x, mask=None):
        """
        x: (N,3,H,W)
        mask: (N,H,W) or None
        Returns:
          logits: (N,40,num_classes)
          attn:   attention weights
          prob:   softmax distribution
        """
        # 1) Feature extraction => (N,40,48)
        x_feat, attn = self.semantic_feature(x, mask=mask)

        # 2) Classification => (N,40,num_classes)
        logits = self.classifier(x_feat)
        prob = F.softmax(logits, dim=-1)
        return logits, attn, prob

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
