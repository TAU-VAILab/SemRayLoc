import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """[Conv3D => BatchNorm => ReLU] x 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2_d, x2_s):
        x1 = self.up(x1)
        x = torch.cat([x2_d, x2_s, x1], dim=1)
        return self.conv(x)

class CombinedProbVolNet_UNet(nn.Module):
    def __init__(self, n_channels=1):
        super(CombinedProbVolNet_UNet, self).__init__()
        self.n_channels = n_channels

        # Depth Encoder
        self.inc_depth = DoubleConv(1, 16)
        self.down1_depth = Down(16, 32)
        self.down2_depth = Down(32, 64)

        # Semantic Encoder
        self.inc_semantic = DoubleConv(1, 16)
        self.down1_semantic = Down(16, 32)
        self.down2_semantic = Down(32, 64)

        # Bottleneck
        self.bottleneck = DoubleConv(128, 128)

        # Decoder
        self.up1 = Up(256, 64)
        self.up2 = Up(128, 32)
        self.up3 = Up(64, 16)
        self.outc = nn.Conv3d(16, n_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, depth_prob_vol, semantic_prob_vol):
        # Prepare inputs
        depth = depth_prob_vol.unsqueeze(1)  # [B, 1, H, W, D]
        semantic = semantic_prob_vol.unsqueeze(1)  # [B, 1, H, W, D]

        # Permute to [B, C, D, H, W]
        depth = depth.permute(0, 1, 4, 2, 3)
        semantic = semantic.permute(0, 1, 4, 2, 3)

        # Move to the correct device and data type
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        depth = depth.to(device=device, dtype=dtype)
        semantic = semantic.to(device=device, dtype=dtype)

        # Depth Encoder
        x1_d = self.inc_depth(depth)
        x2_d = self.down1_depth(x1_d)
        x3_d = self.down2_depth(x2_d)

        # Semantic Encoder
        x1_s = self.inc_semantic(semantic)
        x2_s = self.down1_semantic(x1_s)
        x3_s = self.down2_semantic(x2_s)

        # Bottleneck
        x_bottleneck = torch.cat([x3_d, x3_s], dim=1)
        x_bottleneck = self.bottleneck(x_bottleneck)

        # Decoder
        x = self.up1(x_bottleneck, x2_d, x2_s)
        x = self.up2(x, x1_d, x1_s)
        x = self.up3(x, depth, semantic)
        x = self.outc(x)
        x = self.relu(x)

        # Permute back to [B, H, W, D]
        x = x.squeeze(1).permute(0, 2, 3, 1)

        return x
