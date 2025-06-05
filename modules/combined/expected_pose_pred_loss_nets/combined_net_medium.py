import torch
import torch.nn as nn

class ResidualBlock3D(nn.Module):
    """
    A 3D Residual Block consisting of two Conv3D layers with BatchNorm and ReLU.
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # If the input and output channels differ, use a 1x1x1 convolution to match dimensions
        if in_channels != out_channels:
            self.downsample = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out

class CombinedProbVolNet_medium(nn.Module):
    def __init__(self):
        super(CombinedProbVolNet_medium, self).__init__()
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=3, padding=1),  # Increased from 8 to 16 channels
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            ResidualBlock3D(16, 32),
            ResidualBlock3D(32, 32),
            ResidualBlock3D(32, 64),
            ResidualBlock3D(64, 64),
            ResidualBlock3D(64, 32),
            ResidualBlock3D(32, 16)
        )

        # Final convolution to reduce channels to 1
        self.final_conv = nn.Sequential(
            nn.Conv3d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 1, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, depth_prob_vol, semantic_prob_vol):
        """
        Args:
            depth_prob_vol (torch.Tensor): Tensor of shape [B, H, W, O=36]
            semantic_prob_vol (torch.Tensor): Tensor of shape [B, H, W, O=36]

        Returns:
            torch.Tensor: Combined probability volume of shape [B, H, W, O=36]
        """
        # Permute to [B, O, H, W]
        depth = depth_prob_vol.permute(0, 3, 1, 2)  # [B, O, H, W]
        semantic = semantic_prob_vol.permute(0, 3, 1, 2)  # [B, O, H, W]

        # Stack along the channel dimension: [B, C=2, O, H, W]
        x = torch.stack((depth, semantic), dim=1)  # [B, 2, O, H, W]

        # Ensure the tensor is float and on the correct device
        x = x.float().to(self.initial_conv[0].weight.device).contiguous()

        # Apply initial convolution
        x = self.initial_conv(x)  # [B, 16, O, H, W]

        # Apply residual blocks
        x = self.residual_blocks(x)  # [B, 16, O, H, W]

        # Apply final convolution layers
        x = self.final_conv(x)  # [B, 1, O, H, W]

        # Remove channel dimension: [B, O, H, W]
        x = x.squeeze(1)  # [B, O, H, W]

        # Permute back to [B, H, W, O]
        x = x.permute(0, 2, 3, 1)  # [B, H, W, O]

        return x  # [B, H, W, O]
