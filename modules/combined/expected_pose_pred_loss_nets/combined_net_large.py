import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for lightweight channel-wise attention with increased complexity.
    """
    def __init__(self, in_channels, reduction=4):  # Reduced reduction factor for more parameters
        super(SqueezeExcitation, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, C, H, W = x.size()
        # Squeeze operation
        y = x.view(batch_size, C, -1).mean(dim=2)  # Global average pooling
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch_size, C, 1, 1)
        # Excitation operation
        return x * y

class ResidualBlock(nn.Module):
    """
    Residual block with an optional Squeeze-and-Excitation block.
    """
    def __init__(self, in_channels, out_channels, add_attention=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.attention = SqueezeExcitation(out_channels) if add_attention else None

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

        # Apply SE attention if available
        if self.attention is not None:
            out = self.attention(out)

        return out

class CombinedProbVolNet_large(nn.Module):
    def __init__(self):
        super(CombinedProbVolNet_large, self).__init__()
        # Increase initial convolution channels
        self.initial_conv = nn.Sequential(
            nn.Conv2d(2, 128, kernel_size=3, padding=1),  # Increase to 128 channels
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Add more residual blocks with larger channels
        self.residual_blocks = nn.Sequential(
            ResidualBlock(128, 256, add_attention=True),  # Increase channels in first block
            ResidualBlock(256, 256),
            ResidualBlock(256, 512, add_attention=True),  # Further increase channels
            ResidualBlock(512, 1024),
            ResidualBlock(1024, 1024),
            ResidualBlock(1024, 512),
            ResidualBlock(512, 256, add_attention=True),
            ResidualBlock(256, 256),
            ResidualBlock(256, 128)
        )

        # Final convolution layers with larger channels
        self.final_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # Increase to 64 channels
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, depth_prob_vol, semantic_prob_vol):
        x = torch.stack((depth_prob_vol, semantic_prob_vol), dim=1)
        x = x.float().to(self.initial_conv[0].weight.device).contiguous()

        # Apply initial convolution
        x = self.initial_conv(x)

        # Apply residual blocks with SE attention
        x = self.residual_blocks(x)

        # Apply final convolution layers
        x = self.final_conv(x)

        return x.squeeze(1)  # [B, H, W]
