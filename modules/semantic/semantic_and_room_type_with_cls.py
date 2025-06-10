import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as fn
import torchvision.transforms as T

from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import resnet50

# These utilities (ConvBnReLU, Attention) are assumed to be defined in your modules.
from modules.network_utils import ConvBnReLU, Attention

import lightning.pytorch as pl
import torch.optim as optim

##############################################################################
# 1) Cross Attention Module (single-head)
##############################################################################
class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        """
        A single-head cross attention module.
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

        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.embed_dim ** 0.5)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out = self.dropout(torch.matmul(attn, V))
        return out, attn

##############################################################################
# 2) Feature Extractor using ResNet50 Backbone with Separate Attention
##############################################################################
class semantic_feature_res_ca(nn.Module):
    def __init__(self):
        """
        This module extracts features from a ResNet50 backbone.
        The flattened features (with added positional encoding) are attended over
        by two sets of learnable queries:
          - a single CLS token (for room prediction),
          - and 40 ray tokens (for ray predictions).
        """
        super().__init__()
        # Load ResNet50 with dilation in layer4.
        res50 = resnet50(pretrained=True, replace_stride_with_dilation=[False, False, True])
        self.resnet = nn.Sequential(
            IntermediateLayerGetter(res50, return_layers={"layer4": "feat"})
        )
        # Reduce channels from 2048 to 128.
        self.conv = ConvBnReLU(
            in_channels=2048, out_channels=128, kernel_size=3, padding=1, stride=1
        )
        # Project from 128 to 48 dimensions.
        self.proj = nn.Linear(128, 48)
        # Positional embedding from normalized (x,y) coordinates.
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, 48),
            nn.Tanh()
        )
        # Separate learnable queries: one for the room (CLS) and 40 for rays.
        self.cls_token = nn.Parameter(torch.randn(1, 48))
        self.ray_queries = nn.Parameter(torch.randn(40, 48))
        # Single-head cross attention module (shared for both branches).
        self.cross_attn = CrossAttention(embed_dim=48)
        # Register ImageNet normalization constants as buffers.
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, mask=None):
        """
        x: (N, 3, H, W)
        mask: (N, H, W) or None
        Returns:
          ray_tokens: (N, 40, 48)
          room_token: (N, 1, 48)
          ray_attn:   attention weights from ray cross attention
          room_attn:  attention weights from room cross attention
        """
        N, C, H, W = x.shape
        # Normalize input image.
        x_norm = (x - self.mean) / self.std

        # Extract backbone features.
        feat = self.resnet(x_norm)["feat"]  # (N, 2048, fH, fW)
        feat = self.conv(feat)              # (N, 128, fH, fW)
        fH, fW = feat.shape[2], feat.shape[3]

        # Flatten spatial dimensions: (N, 128, fH*fW) -> (N, fH*fW, 128)
        feat_flat = feat.view(N, 128, -1).permute(0, 2, 1)
        # Project features to 48 dimensions.
        feat_flat = self.proj(feat_flat)  # (N, fH*fW, 48)

        # Create a normalized (x,y) grid and compute positional embeddings.
        grid_y = torch.linspace(-0.5, 0.5, fH, device=x.device)
        grid_x = torch.linspace(-0.5, 0.5, fW, device=x.device)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
        pos_coords = torch.stack((grid_x, grid_y), dim=-1).view(-1, 2)  # (fH*fW, 2)
        pos_emb = self.pos_mlp(pos_coords)  # (fH*fW, 48)
        pos_emb = pos_emb.unsqueeze(0).expand(N, -1, -1)
        feat_flat = feat_flat + pos_emb

        # Create attention masks (if a mask is provided).
        if mask is not None:
            mask_down = fn.resize(mask, (fH, fW), interpolation=fn.InterpolationMode.NEAREST).bool()
            mask_down = ~mask_down  # True indicates positions to ignore.
            mask_down = mask_down.view(N, -1)
            attn_mask_room = mask_down.unsqueeze(1).expand(-1, 1, -1)   # For CLS token.
            attn_mask_ray = mask_down.unsqueeze(1).expand(-1, 40, -1)    # For ray tokens.
        else:
            attn_mask_room = None
            attn_mask_ray = None

        # Room branch: use CLS token.
        cls_token = self.cls_token.unsqueeze(0).expand(N, -1, -1)  # (N, 1, 48)
        room_token, room_attn = self.cross_attn(cls_token, feat_flat, feat_flat, attn_mask=attn_mask_room)

        # Ray branch: use ray queries.
        ray_queries = self.ray_queries.unsqueeze(0).expand(N, -1, -1)  # (N, 40, 48)
        ray_tokens, ray_attn = self.cross_attn(ray_queries, feat_flat, feat_flat, attn_mask=attn_mask_ray)

        return ray_tokens, room_token, ray_attn, room_attn

##############################################################################
# 3) Minimal Self-Attention Block (single-head) for each branch
##############################################################################
class _SingleHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        # x: (N, L, embed_dim)
        N, L, D = x.shape
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (D ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        return out

class _SingleHeadSelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim=48, ffn_multiplier=4):
        """
        A minimal self-attention block with residual connections and an FFN.
        """
        super().__init__()
        self.embed_dim = embed_dim
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
        # Self-attention with residual connection.
        x_attn = self.attn_layer(x)
        x = x + x_attn
        x = self.ln1(x)
        # Feed-forward network with residual connection.
        x_ffn = self.ffn(x)
        x = x + x_ffn
        x = self.ln2(x)
        return x

##############################################################################
# 4) Separate Branches for Ray and Room Predictions
##############################################################################
class RayBranch(nn.Module):
    def __init__(self, num_ray_classes, embed_dim=48):
        """
        Processes the 40 ray tokens.
        """
        super().__init__()
        self.self_attn = _SingleHeadSelfAttentionBlock(embed_dim=embed_dim, ffn_multiplier=4)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(64, num_ray_classes)
        )

    def forward(self, x):
        # x: (N, 40, embed_dim)
        x = self.self_attn(x)
        out = self.mlp(x)  # (N, 40, num_ray_classes)
        return out

class RoomBranch(nn.Module):
    def __init__(self, num_room_types, embed_dim=48):
        """
        Processes the single CLS token for room prediction.
        """
        super().__init__()
        self.self_attn = _SingleHeadSelfAttentionBlock(embed_dim=embed_dim, ffn_multiplier=4)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(64, num_room_types)
        )

    def forward(self, x):
        # x: (N, 1, embed_dim) -> apply self-attention and then squeeze.
        x = self.self_attn(x)
        x = x.squeeze(1)
        out = self.mlp(x)  # (N, num_room_types)
        return out

##############################################################################
# 5) Final Multi-task Network
##############################################################################
class semantic_net(nn.Module):
    def __init__(self, num_ray_classes, num_room_types):
        """
        num_ray_classes: number of classes for each of the 40 rays.
        num_room_types: number of room classes (single prediction).
        """
        super().__init__()
        # Feature extractor: returns separate tokens for rays and room.
        self.semantic_feature = semantic_feature_res_ca()
        # Two separate branches after token extraction.
        self.ray_branch = RayBranch(num_ray_classes, embed_dim=48)
        self.room_branch = RoomBranch(num_room_types, embed_dim=48)

    def forward(self, x, mask=None):
        """
        x: (N, 3, H, W)
        mask: (N, H, W) or None
        Returns:
          ray_logits:  (N, 40, num_ray_classes)
          room_logits: (N, num_room_types)
          ray_attn:    attention weights from ray cross attention (for debugging)
          room_attn:   attention weights from room cross attention (for debugging)
        """
        ray_tokens, room_token, ray_attn, room_attn = self.semantic_feature(x, mask=mask)
        ray_logits = self.ray_branch(ray_tokens)
        room_logits = self.room_branch(room_token)
        return ray_logits, room_logits, ray_attn

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
