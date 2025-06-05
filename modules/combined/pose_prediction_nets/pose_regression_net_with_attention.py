# modules/combined/pose_regression_net_with_attention.py

import torch
import torch.nn as nn

class ResidualBlock2D(nn.Module):
    """
    A 2D Residual Block with optional downsampling.
    """
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock2D, self).__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out

class AttentionFusion(nn.Module):
    """
    Attention Fusion Module that applies attention weights to combined features.
    """
    def __init__(self, in_channels):
        super(AttentionFusion, self).__init__()
        self.attention_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
            
    def forward(self, depth_feat, semantic_feat):
        # Ensure features are on the same device
        device = depth_feat.device
        semantic_feat = semantic_feat.to(device)
            
        # Concatenate features
        combined_feat = torch.cat([depth_feat, semantic_feat], dim=1)  # [B, C_d + C_s, H, W]
        # Compute attention weights
        attention_weights = self.attention_conv(combined_feat)  # [B, C, H, W]
        # Apply attention weights
        attended_feat = combined_feat * attention_weights
        return attended_feat

class PoseRegressionNetWithAttention(nn.Module):
    def __init__(self):
        super(PoseRegressionNetWithAttention, self).__init__()
        # Depth Branch
        self.depth_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            ResidualBlock2D(32, 64, downsample=True),
            ResidualBlock2D(64, 128, downsample=True),
            ResidualBlock2D(128, 256, downsample=True),
            nn.AdaptiveAvgPool2d(4)  # Adjust pooling size as needed
        )
            
        # Semantic Branch
        self.semantic_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            ResidualBlock2D(32, 64, downsample=True),
            ResidualBlock2D(64, 128, downsample=True),
            ResidualBlock2D(128, 256, downsample=True),
            nn.AdaptiveAvgPool2d(4)  # Adjust pooling size as needed
        )
            
        # Attention Fusion Layer
        self.attention_fusion = AttentionFusion(in_channels=512)  # 256 depth + 256 semantic
            
        # Fusion Convolutional Layers
        self.fusion_conv = nn.Sequential(
            ResidualBlock2D(512, 512),
            nn.AdaptiveAvgPool2d(1)  # Pool to [B, C, 1, 1]
        )
            
        # Fully Connected Layers for Pose Regression
        self.fc = nn.Sequential(
            nn.Flatten(),  # Flatten the tensor
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(128, 2)  # Output x, y
        )
            
    def forward(self, depth_prob_vol, semantic_prob_vol):
        """
        Args:
            depth_prob_vol (torch.Tensor): Tensor of shape [B, H, W]
            semantic_prob_vol (torch.Tensor): Tensor of shape [B, H, W]
        
        Returns:
            torch.Tensor: Predicted pose tensor of shape [B, 2]
        """
        B = depth_prob_vol.shape[0]

        # Add channel dimension
        depth = depth_prob_vol.unsqueeze(1)       # [B, C=1, H, W]
        semantic = semantic_prob_vol.unsqueeze(1)  # [B, C=1, H, W]

        # Ensure tensors are on the correct device and of type float32
        device = next(self.parameters()).device
        depth = depth.to(device=device, dtype=torch.float32)
        semantic = semantic.to(device=device, dtype=torch.float32)

        # Process each branch
        depth_feat = self.depth_branch(depth)           # [B, C=256, H', W']
        semantic_feat = self.semantic_branch(semantic)  # [B, C=256, H', W']

        # Fuse features with attention
        combined_feat = self.attention_fusion(depth_feat, semantic_feat)  # [B, C=512, H', W']

        # Further process with fusion convolution
        fused_feat = self.fusion_conv(combined_feat)  # [B, C=512, 1, 1]

        # Predict pose
        pose_pred = self.fc(fused_feat).squeeze(-1).squeeze(-1)  # [B, 2]

        return pose_pred  # [B, 2]
