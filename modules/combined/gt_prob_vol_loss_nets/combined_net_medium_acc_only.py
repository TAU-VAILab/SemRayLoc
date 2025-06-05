import torch
import torch.nn as nn

class ResidualBlock2D(nn.Module):
    """
    A 2D Residual Block consisting of two Conv2D layers with BatchNorm and ReLU.
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If the input and output channels differ, use a 1x1 convolution to match dimensions
        if in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1)
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
            nn.Conv2d(2, 16, kernel_size=3, padding=1),  # Start with 2 input channels
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            ResidualBlock2D(16, 32),
            ResidualBlock2D(32, 32),
            ResidualBlock2D(32, 64),
            ResidualBlock2D(64, 64),
            ResidualBlock2D(64, 32),
            ResidualBlock2D(32, 16)
        )

        # Final convolution to reduce channels to 1
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, depth_prob_vol, semantic_prob_vol):
        """
        Args:
            depth_prob_vol (torch.Tensor): Tensor of shape [B, H, W]
            semantic_prob_vol (torch.Tensor): Tensor of shape [B, H, W]

        Returns:
            torch.Tensor: Combined probability map of shape [B, H, W]
        """
        # Stack depth and semantic probability volumes along the channel dimension
        x = torch.stack((depth_prob_vol, semantic_prob_vol), dim=1)  # [B, 2, H, W]

        # Ensure the tensor is float and on the correct device
        x = x.float().to(self.initial_conv[0].weight.device).contiguous()

        # Apply initial convolution
        x = self.initial_conv(x)  # [B, 16, H, W]

        # Apply residual blocks
        x = self.residual_blocks(x)  # [B, 16, H, W]

        # Apply final convolution layers
        x = self.final_conv(x)  # [B, 1, H, W]

        # Squeeze out the channel dimension to get [B, H, W]
        x = x.squeeze(1)  # [B, H, W]

        return x  # [B, H, W]
