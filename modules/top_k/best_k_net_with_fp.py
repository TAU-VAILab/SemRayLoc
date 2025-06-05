import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter

# -----------------------------
# Utility Modules
# -----------------------------

class PositionalMLP(nn.Module):
    """A small MLP to compute positional embeddings."""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.mlp(x)

class Attention(nn.Module):
    """Scaled dot-product attention."""
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, attn_mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, value)
        return output, attn

# -----------------------------
# Image Feature Extractor
# -----------------------------

class ImageFeatureExtractor(nn.Module):
    """
    Extract features from an input image.
    Uses a ResNet50 backbone (with layer4 features) and a small convolution before adaptive pooling.
    Outputs a fixed-width feature map, to which we add 1D positional encoding.
    """
    def __init__(self, output_channels=160, fixed_width=40):
        super().__init__()
        # Use a pretrained ResNet50; extract features from layer4.
        resnet = resnet50(pretrained=True, replace_stride_with_dilation=[False, False, True])
        self.resnet = IntermediateLayerGetter(resnet, return_layers={"layer4": "feat"})
        self.conv = nn.Conv2d(2048, 128, kernel_size=3, stride=1, padding=1)
        self.fixed_width = fixed_width
        # 1D positional encoding for the width dimension.
        self.pos_mlp = PositionalMLP(in_dim=1, hidden_dim=32, out_dim=32)
        # Project concatenated features to desired output channels.
        self.proj = nn.Linear(128 + 32, output_channels)
    
    def forward(self, x):
        """
        Args:
          x: Tensor of shape (N, 3, 360, 640)
        Returns:
          features: (N, fixed_width, output_channels)
        """
        N = x.size(0)
        # Normalize with ImageNet statistics.
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_norm = (x - mean) / std

        feat_dict = self.resnet(x_norm)
        feat = feat_dict["feat"]  # (N, 2048, H, W)
        feat = self.conv(feat)    # (N, 128, H, W)

        # Adaptive pool along width so that we have a fixed number of columns.
        feat = F.adaptive_avg_pool2d(feat, (feat.size(2), self.fixed_width))  # (N, 128, H, fixed_width)
        # Collapse the height via average pooling.
        feat = feat.mean(dim=2)   # (N, 128, fixed_width)
        feat = feat.permute(0, 2, 1)  # (N, fixed_width, 128)

        # Compute positional encoding along the width.
        pos = torch.linspace(0, 1, self.fixed_width, device=x.device).unsqueeze(-1)  # (fixed_width, 1)
        pos_enc = self.pos_mlp(pos)  # (fixed_width, 32)
        pos_enc = pos_enc.unsqueeze(0).repeat(N, 1, 1)  # (N, fixed_width, 32)

        # Concatenate features with their positional encodings.
        feat = torch.cat([feat, pos_enc], dim=-1)  # (N, fixed_width, 128+32)
        feat = self.proj(feat)  # (N, fixed_width, output_channels)
        return feat

# -----------------------------
# Semantic Feature Extractor
# -----------------------------

class SemanticFeatureExtractor(nn.Module):
    """
    Extract features from the semantic map (floorplan).
    A simple CNN to process the (1, 300, 300) semantic map.
    """
    def __init__(self, output_dim=64):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (N, 32, 300, 300)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (N, 32, 150, 150)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (N, 64, 150, 150)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (N, 64, 75, 75)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # (N, 128, 75, 75)
            nn.ReLU(),
            nn.MaxPool2d(3),  # (N, 128, 25, 25)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # (N, 256, 25, 25)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))  # (N, 256, 1, 1)
        )
        self.fc = nn.Linear(256, output_dim)
    
    def forward(self, x):
        """
        Args:
          x: Tensor of shape (N, 1, 300, 300)
        Returns:
          features: Tensor of shape (N, output_dim)
        """
        feat = self.conv_layers(x)  # (N, 256, 1, 1)
        feat = feat.view(feat.size(0), -1)  # (N, 256)
        feat = self.fc(feat)  # (N, output_dim)
        feat = F.relu(feat)
        return feat

# -----------------------------
# Candidate Encoder
# -----------------------------

class CandidateEncoder(nn.Module):
    """
    Encodes each candidate's vector (e.g., depth or semantic).
    """
    def __init__(self, input_dim=40, output_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        """
        Args:
          x: Tensor of shape (N, num_candidates, input_dim)
        Returns:
          Tensor of shape (N, num_candidates, output_dim)
        """
        return self.mlp(x)

# -----------------------------
# Candidate Position Encoder
# -----------------------------

class CandidatePositionEncoder(nn.Module):
    """
    Encodes each candidate's positional information (x, y, o).
    """
    def __init__(self, input_dim=3, hidden_dim=16, output_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        """
        Args:
          x: Tensor of shape (N, num_candidates, 3)  # (x, y, o)
        Returns:
          Tensor of shape (N, num_candidates, output_dim)
        """
        return self.mlp(x)  # (N, num_candidates, output_dim)

# -----------------------------
# Main Network: best_k_net_with_fp
# -----------------------------

class best_k_net_with_fp(nn.Module):
    """
    The classification network that fuses features from an RGB image,
    a depth vector, a semantic vector, a floorplan, and candidate positions
    to predict the best candidate (k).
    
    Inputs:
      - image: Tensor of shape (N, 3, 360, 640)
      - depth_vec: Tensor of shape (N, 5, 40)
      - sem_vec: Tensor of shape (N, 5, 40)
      - semantic_map: Tensor of shape (N, 1, 300, 300)
      - k_positions: Tensor of shape (N, 5, 3)  # Each candidate's (x, y, o)
      
    Output:
      - logits: Tensor of shape (N, 5); logits for each of the 5 candidates.
    """
    def __init__(self, num_candidates=5,
                 image_feature_dim=160,
                 candidate_feature_dim=64,
                 fusion_dim=128,
                 semantic_feature_dim=64,
                 position_feature_dim=32,
                 num_classes=5):
        super().__init__()
        self.num_candidates = num_candidates

        # Image feature extractor.
        self.image_extractor = ImageFeatureExtractor(output_channels=image_feature_dim, fixed_width=40)
        # Semantic feature extractor.
        self.semantic_extractor = SemanticFeatureExtractor(output_dim=semantic_feature_dim)
        # Candidate (depth and semantic) encoders.
        self.depth_encoder = CandidateEncoder(input_dim=40, output_dim=candidate_feature_dim)
        self.sem_encoder = CandidateEncoder(input_dim=40, output_dim=candidate_feature_dim)
        # Candidate position encoder.
        self.pos_encoder = CandidatePositionEncoder(input_dim=3, hidden_dim=16, output_dim=position_feature_dim)
        # Fuse candidate features with positional features.
        self.candidate_proj = nn.Linear(candidate_feature_dim + candidate_feature_dim + position_feature_dim, fusion_dim)
        # Project image features to the fusion dimension.
        self.image_proj = nn.Linear(image_feature_dim, fusion_dim)
        # Project semantic features to the fusion dimension.
        self.semantic_proj = nn.Linear(semantic_feature_dim, fusion_dim)
        # Attention block: use combined image and semantic embeddings as query and candidate embedding as key/value.
        self.attention = Attention()
        # Final classifier to produce a score (logit) per candidate.
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_dim // 2, 1)
        )
    
    def forward(self, image, depth_vec, sem_vec, semantic_map, k_positions):
        """
        Args:
          image: Tensor of shape (N, 3, 360, 640)
          depth_vec: Tensor of shape (N, 5, 40)
          sem_vec: Tensor of shape (N, 5, 40)
          semantic_map: Tensor of shape (N, 1, 300, 300)
          k_positions: Tensor of shape (N, 5, 3)  # Each candidate's (x, y, o)
        Returns:
          logits: Tensor of shape (N, 5)
        """
        N = image.size(0)
        # 1. Extract image features. The extractor produces (N, fixed_width, image_feature_dim).
        img_features = self.image_extractor(image)  # (N, 40, 160)
        # Average the spatial dimension to obtain a global representation.
        img_features_mean = img_features.mean(dim=1)  # (N, 160)
        img_emb = self.image_proj(img_features_mean)    # (N, fusion_dim)

        # 2. Extract semantic map features.
        sem_map_features = self.semantic_extractor(semantic_map)  # (N, semantic_feature_dim)
        sem_map_emb = self.semantic_proj(sem_map_features)        # (N, fusion_dim)

        # Combine image and semantic map embeddings.
        combined_emb = img_emb + sem_map_emb  # (N, fusion_dim)

        # 3. Encode the candidate vectors.
        depth_enc = self.depth_encoder(depth_vec)   # (N, 5, 64)
        sem_enc = self.sem_encoder(sem_vec)         # (N, 5, 64)
        # Encode the candidate positions.
        pos_enc = self.pos_encoder(k_positions)     # (N, 5, 32)
        pos_enc = pos_enc.permute(1, 0, 2)  # Now [16, 5, 32]

        # Concatenate candidate encodings with positional encodings.
        cand_enc = torch.cat([depth_enc, sem_enc, pos_enc], dim=-1)  # (N, 5, 64 + 64 + 32 = 160)
        # Fuse the encodings via projection.
        cand_emb = self.candidate_proj(cand_enc)    # (N, 5, fusion_dim)

        # 4. Use attention: combined_emb (as query) attends over candidate embeddings.
        query = combined_emb.unsqueeze(1)           # (N, 1, fusion_dim)
        fused, attn_weights = self.attention(query, cand_emb, cand_emb)  # fused: (N, 1, fusion_dim)
        fused = fused.squeeze(1)                     # (N, fusion_dim)

        # 5. For each candidate, fuse the image-guided representation with candidate embedding.
        fused_expand = fused.unsqueeze(1).repeat(1, self.num_candidates, 1)  # (N, 5, fusion_dim)
        fusion_feature = fused_expand * cand_emb                         # element-wise multiplication (N, 5, fusion_dim)
        # Compute a logit per candidate.
        logits = self.classifier(fusion_feature).squeeze(-1)  # (N, 5)
        return logits

# -----------------------------
# For Testing the Network Independently
# -----------------------------
if __name__ == '__main__':
    # Create dummy data
    N = 2
    image = torch.randn(N, 3, 360, 640)
    depth_vec = torch.randn(N, 5, 40)
    sem_vec = torch.randn(N, 5, 40)
    semantic_map = torch.randn(N, 1, 300, 300)
    k_positions = torch.randn(N, 5, 3)  # Each candidate's (x, y, o)
    
    model = best_k_net_with_fp()
    logits = model(image, depth_vec, sem_vec, semantic_map, k_positions)
    print("Logits shape:", logits.shape)  # Expected: (N, 5)
