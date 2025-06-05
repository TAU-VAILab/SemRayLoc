import torch
import torch.nn as nn

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channel, D, H, W = x.size()
        y = x.view(batch_size, channel, -1).mean(dim=2)  # Global Average Pooling
        y = self.fc(y).view(batch_size, channel, 1, 1, 1)
        return x * y

class ResidualBlock3D_SE(nn.Module):
    """3D Residual Block with SE Attention"""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock3D_SE, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.se = SEBlock(out_channels)

        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out

class CombinedProbVolNet_Attention(nn.Module):
    def __init__(self):
        super(CombinedProbVolNet_Attention, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )

        self.residual_blocks = nn.Sequential(
            ResidualBlock3D_SE(16, 32),
            ResidualBlock3D_SE(32, 64),
            ResidualBlock3D_SE(64, 128),
            ResidualBlock3D_SE(128, 128),
            ResidualBlock3D_SE(128, 64),
            ResidualBlock3D_SE(64, 32)
        )

        self.final_conv = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 1, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, depth_prob_vol, semantic_prob_vol):
        # Stack inputs
        x = torch.stack([depth_prob_vol, semantic_prob_vol], dim=1)  # [B, 2, H, W, D]
        x = x.permute(0, 1, 4, 2, 3)  # [B, 2, D, H, W]
        x = x.float().contiguous()

        x = self.initial_conv(x)
        x = self.residual_blocks(x)
        x = self.final_conv(x)

        x = x.squeeze(1).permute(0, 2, 3, 1)  # [B, H, W, D]
        return x
