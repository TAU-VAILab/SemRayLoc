import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

##############################################################################
# 1) ResNet-50 Backbone (Partial)
##############################################################################
class ResNetBackbone(nn.Module):
    """
    A ResNet-50 backbone that extracts features up to layer3
    and then projects them to embed_dim via 1x1 Conv.
    Output shape: (N, embed_dim, H/16, W/16)
    """
    def __init__(self, embed_dim=256, freeze_backbone=True):
        super().__init__()
        # Load a standard ResNet-50 (no pretrained weights or adjust as needed).
        base_resnet = models.resnet50(weights=None)
        
        # Optionally, load pretrained weights:
        # base_resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # We take only the layers up to layer3:
        self.backbone = nn.Sequential(
            base_resnet.conv1,
            base_resnet.bn1,
            base_resnet.relu,
            base_resnet.maxpool,
            base_resnet.layer1,
            base_resnet.layer2,
            base_resnet.layer3
        )
        
        # 1x1 conv to map from 1024 (layer3 output) -> embed_dim
        self.conv_out = nn.Sequential(
            nn.Conv2d(1024, embed_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # Optionally freeze the entire backbone to reduce trainable parameters
        # to around the size you need (~500K).
        # You can unfreeze certain layers if you want more capacity.
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        x: (N, 3, H, W)
        returns: (N, embed_dim, H/16, W/16)
        """
        features = self.backbone(x)       # -> (N, 1024, H/16, W/16)
        out = self.conv_out(features)     # -> (N, embed_dim, H/16, W/16)
        return out

##############################################################################
# 2) Positional Embedding
##############################################################################
class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.Tanh()
        )
        
    def forward(self, feature_map):
        # feature_map: (N, C, H, W)
        N, C, H, W = feature_map.shape
        # Create a normalized coordinate grid of shape (H, W, 2).
        device = feature_map.device
        grid_y = torch.linspace(-0.5, 0.5, H, device=device)
        grid_x = torch.linspace(-0.5, 0.5, W, device=device)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
        pos = torch.stack((grid_x, grid_y), dim=-1)  # (H, W, 2)
        pos = pos.view(-1, 2)  # (H*W, 2)
        pos_emb = self.mlp(pos)  # (H*W, embed_dim)
        pos_emb = pos_emb.unsqueeze(0).expand(N, -1, -1)  # (N, H*W, embed_dim)
        return pos_emb

##############################################################################
# 3) Single-Head Cross-Attention
##############################################################################
class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(p=0.0)
    
    def forward(self, queries, keys, values, attn_mask=None):
        """
        queries: (N, num_queries, embed_dim)
        keys:    (N, seq_len, embed_dim)
        values:  (N, seq_len, embed_dim)
        """
        Q = self.query_proj(queries)
        K = self.key_proj(keys)
        V = self.value_proj(values)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.embed_dim ** 0.5)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out = self.dropout(torch.matmul(attn, V))
        return out, attn

##############################################################################
# 4) Self-Attention Block
##############################################################################
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)
    
    def forward(self, x):
        """
        x: (N, L, embed_dim)
        """
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (x.size(-1) ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        return out

class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, ffn_multiplier=2):
        super().__init__()
        self.self_attn = SelfAttention(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        
        hidden_dim = int(embed_dim * ffn_multiplier)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.ln2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        """
        x: (N, L, embed_dim)
        """
        attn_out = self.self_attn(x)
        x = x + attn_out
        x = self.ln1(x)
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.ln2(x)
        return x

##############################################################################
# 5) Classification Head
##############################################################################
class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)

##############################################################################
# 6) Final Network (with ResNet-50 Backbone)
##############################################################################
class room_type_resnet50_net(nn.Module):
    def __init__(self, num_classes, embed_dim=256):
        """
        num_classes: number of output classes
        embed_dim:   feature dimension used throughout
        """
        super().__init__()
        
        # Replace the custom CNN with a ResNet-50 backbone:
        self.backbone = ResNetBackbone(embed_dim=embed_dim, freeze_backbone=True)
        
        # Positional embedding added to the flattened feature map
        self.pos_emb = PositionalEmbedding(embed_dim)
        
        # Single learnable query token
        self.num_tokens = 1
        self.query = nn.Parameter(torch.randn(self.num_tokens, embed_dim))
        
        # Cross-attention
        self.cross_attn = CrossAttention(embed_dim)
        
        # Self-attention block on the token
        self.self_attn_block = SelfAttentionBlock(embed_dim, ffn_multiplier=2)
        
        # Final classifier
        self.classifier = ClassificationHead(embed_dim, num_classes)
        
        self._print_trainable_parameters()
    
    def forward(self, x, mask=None):
        """
        x: (N,3,H,W)
        returns: (logits, attn)
        """
        # 1) Extract features via ResNet-50 (partial)
        features = self.backbone(x)  # (N, embed_dim, H/16, W/16)
        N, C, H_feat, W_feat = features.shape
        
        # 2) Flatten spatial dimensions: (N, H_feat*W_feat, embed_dim)
        features_flat = features.view(N, C, -1).permute(0, 2, 1).contiguous()  # (N, H*W, C)
        
        # 3) Positional embedding
        pos_emb = self.pos_emb(features)  # (N, H_feat*W_feat, embed_dim)
        features_flat = features_flat + pos_emb
        
        # 4) Prepare query token and expand for batch
        queries = self.query.unsqueeze(0).expand(N, -1, -1)  # (N, 1, embed_dim)
        
        # 5) Cross-attention: token attends over spatial features
        token, attn = self.cross_attn(queries, features_flat, features_flat, attn_mask=None)
        
        # 6) Self-attention block (with single token, it still applies LN + FFN)
        token = self.self_attn_block(token)  # (N, 1, embed_dim)
        
        # 7) Classification head
        logits = self.classifier(token.squeeze(1))  # (N, num_classes)
        
        return logits, attn

    def _print_trainable_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("\n=== Trainable Parameter Check ===")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"[TRAINABLE] {name:<60} shape={list(param.shape)} count={param.numel()}")
        print(f"\nTotal params:      {total_params}")
        print(f"Trainable params:  {trainable_params}")
        print("=================================\n")
