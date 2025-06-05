import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Image Feature Extractor (Simplified)
# -----------------------------
class ImageFeatureExtractorSimple(nn.Module):
    """
    A simpler CNN backbone that reduces the input image to a global feature vector.
    Input: (N, 3, 360, 640)
    Output: (N, out_dim)
    """
    def __init__(self, out_dim=128):
        super().__init__()
        # A small stack of conv layers with stride to reduce spatial dims.
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),  # -> (N, 32, 180, 320)
            nn.ReLU(),
            nn.MaxPool2d(2),                                       # -> (N, 32, 90, 160)

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> (N, 64, 45, 80)
            nn.ReLU(),
            nn.MaxPool2d(2),                                       # -> (N, 64, 22, 40)

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# -> (N, 128, 11, 20)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))                           # -> (N, 128, 1, 1)
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        """
        x: (N, 3, 360, 640)
        returns: (N, out_dim)
        """
        feats = self.conv_layers(x)         # (N, 128, 1, 1)
        feats = feats.view(feats.size(0), -1)  # (N, 128)
        feats = self.fc(feats)                # (N, out_dim)
        feats = F.relu(feats)
        return feats


# -----------------------------
# Semantic Feature Extractor (Simplified)
# -----------------------------
class SemanticFeatureExtractorSimple(nn.Module):
    """
    A simpler CNN for the floorplan semantic map.
    Input: (N, 1, 300, 300)
    Output: (N, out_dim)
    """
    def __init__(self, out_dim=64):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),  # -> (N, 32, 150, 150)
            nn.ReLU(),
            nn.MaxPool2d(2),                                       # -> (N, 32, 75, 75)

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> (N, 64, 38, 38)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))                           # -> (N, 64, 1, 1)
        )
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        """
        x: (N, 1, 300, 300)
        returns: (N, out_dim)
        """
        feats = self.conv_layers(x)           # (N, 64, 1, 1)
        feats = feats.view(feats.size(0), -1) # (N, 64)
        feats = self.fc(feats)                # (N, out_dim)
        feats = F.relu(feats)
        return feats


# -----------------------------
# Candidate Encoders (Depth, SemVec)
# -----------------------------
class CandidateEncoderSimple(nn.Module):
    """
    Encode each candidate's depth or semantic vector.
    For each candidate: (N, k, in_dim) -> (N, k, out_dim).
    """
    def __init__(self, in_dim=40, out_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        """
        x: (N, k, in_dim)
        returns: (N, k, out_dim)
        """
        return self.mlp(x)


# -----------------------------
# Position Encoder
# -----------------------------
class PositionEncoderSimple(nn.Module):
    """
    Encodes each candidate's position (x, y, o).
    (N, k, 3) -> (N, k, out_dim).
    """
    def __init__(self, in_dim=3, out_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.mlp(x)


# -----------------------------
# Main Network: SimpleKNetWithFP
# -----------------------------
class SimpleKNetWithFP(nn.Module):
    """
    A simpler classification network that fuses:
      - Image features (global)
      - Semantic map features (global)
      - Candidate depth/sem vectors (per candidate)
      - Candidate positions (per candidate)

    Outputs logits of shape (N, 3) for the 3 candidates.
    """
    def __init__(self,
                 k=3,
                 image_out_dim=128,
                 semantic_out_dim=64,
                 candidate_out_dim=64,
                 position_out_dim=32,
                 hidden_dim=128):
        super().__init__()
        self.k = k

        # Global feature extractors
        self.image_extractor = ImageFeatureExtractorSimple(out_dim=image_out_dim)
        self.sem_map_extractor = SemanticFeatureExtractorSimple(out_dim=semantic_out_dim)

        # Candidate encoders
        self.depth_encoder = CandidateEncoderSimple(in_dim=40, out_dim=candidate_out_dim)
        self.sem_encoder   = CandidateEncoderSimple(in_dim=40, out_dim=candidate_out_dim)
        self.pos_encoder   = PositionEncoderSimple(in_dim=3, out_dim=position_out_dim)

        # Projection layers
        # We fuse the global embeddings from image + semantic map => global_emb
        # dimension = image_out_dim + semantic_out_dim => project to hidden_dim
        self.global_proj = nn.Linear(image_out_dim + semantic_out_dim, hidden_dim)

        # We fuse the candidate embeddings => dimension = depth + sem + position
        # => candidate_out_dim + candidate_out_dim + position_out_dim
        # => 64 + 64 + 32 = 160 => project to hidden_dim
        self.candidate_proj = nn.Linear(candidate_out_dim*2 + position_out_dim, hidden_dim)

        # Final classifier: merges the global_emb + candidate_emb (here we do a simple sum)
        #   fused_emb = global_emb + candidate_emb
        # Then we feed that to a small MLP to get 1 logit per candidate
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 1 logit per candidate
        )

    def forward(self, image, depth_vec, sem_vec, semantic_map, k_positions):
        """
        Args:
          image: (N, 3, 360, 640)
          depth_vec: (N, 3, 40)
          sem_vec: (N, 3, 40)
          semantic_map: (N, 1, 300, 300)
          k_positions: (N, 3, 3)
        Returns:
          logits: (N, 3)
        """
        N = image.size(0)

        # 1) Global features from image and semantic map
        img_feats = self.image_extractor(image)         # (N, image_out_dim)
        sem_map_feats = self.sem_map_extractor(semantic_map)  # (N, semantic_out_dim)

        global_feats = torch.cat([img_feats, sem_map_feats], dim=1)  # (N, image_out_dim + semantic_out_dim)
        global_emb = self.global_proj(global_feats)                 # (N, hidden_dim)
        global_emb = F.relu(global_emb)

        # 2) Per-candidate features
        depth_enc = self.depth_encoder(depth_vec)   # (N, 3, candidate_out_dim)
        sem_enc   = self.sem_encoder(sem_vec)       # (N, 3, candidate_out_dim)
        pos_enc   = self.pos_encoder(k_positions)   # (N, 3, position_out_dim)

        # Combine them
        cand_feats = torch.cat([depth_enc, sem_enc, pos_enc], dim=-1)  # (N, 3, 160)
        cand_emb = self.candidate_proj(cand_feats)                     # (N, 3, hidden_dim)
        cand_emb = F.relu(cand_emb)

        # 3) Fuse global_emb with each candidate
        #    We'll do an elementwise sum: fused_emb = cand_emb + global_emb (broadcasted)
        global_emb_expanded = global_emb.unsqueeze(1).expand(-1, self.k, -1)  # (N, 3, hidden_dim)
        fused_emb = cand_emb + global_emb_expanded  # (N, 3, hidden_dim)

        # 4) Classify each candidate => 1 logit per candidate => (N, 3)
        logits = self.classifier(fused_emb).squeeze(-1)
        return logits


# -----------------------------
# Quick test
# -----------------------------
if __name__ == "__main__":
    # Dummy test
    N = 2
    k = 3
    image = torch.randn(N, 3, 360, 640)
    depth_vec = torch.randn(N, k, 40)
    sem_vec   = torch.randn(N, k, 40)
    semantic_map = torch.randn(N, 1, 300, 300)
    k_positions = torch.randn(N, k, 3)

    model = SimpleKNetWithFP()
    out = model(image, depth_vec, sem_vec, semantic_map, k_positions)
    print("Logits shape:", out.shape)  # (2, 3)
