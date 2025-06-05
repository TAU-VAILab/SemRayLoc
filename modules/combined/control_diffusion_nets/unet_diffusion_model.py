# unet_diffusion_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class UNetModel(nn.Module):
    def __init__(self, in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions, dropout, channel_mult, conv_resample, dims):
        super(UNetModel, self).__init__()
        self.in_channels = in_channels  # Should be 3
        self.out_channels = out_channels
        self.model_channels = model_channels

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(model_channels, model_channels * 4),
            nn.GELU(),
            nn.Linear(model_channels * 4, model_channels)
        )

        # Encoder
        self.down1 = nn.Conv3d(self.in_channels, model_channels, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(2)
        self.down2 = nn.Conv3d(model_channels, model_channels * 2, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(2)

        # Decoder
        self.up1 = nn.Conv3d(model_channels * 4, model_channels * 2, kernel_size=3, padding=1)
        self.up2 = nn.Conv3d(model_channels * 3, model_channels, kernel_size=3, padding=1)
        self.conv_final = nn.Conv3d(model_channels, out_channels, kernel_size=1)
        
    def forward(self, x, t, cond):
        # x: [B, 1, H, W, D]
        # t: [B]
        # cond: [B, 2, H, W, D]
        
        # Concatenate x and cond along the channel dimension
        x = torch.cat([x, cond], dim=1)  # x: [B, 3, H, W, D]

        # Get time embedding and pass it through an MLP
        t_embed = self.time_embedding(t)  # [B, model_channels]
        t_embed = self.time_mlp(t_embed)  # [B, model_channels]
        t_embed = t_embed[:, :, None, None, None]  # [B, model_channels, 1, 1, 1]

        # Encoder
        x1 = self.down1(x)  # [B, model_channels, H, W, D]
        x1 = x1 + t_embed  # Add time embedding
        x1 = F.relu(x1)
        x1_pooled = self.pool1(x1)

        x2 = self.down2(x1_pooled)  # [B, model_channels * 2, H/2, W/2, D/2]
        x2 = F.relu(x2)
        x2_pooled = self.pool2(x2)

        x_up1 = F.interpolate(x2_pooled, size=x2.shape[2:], mode='trilinear', align_corners=False)
        x_up1 = torch.cat([x_up1, x2], dim=1)  # Concatenate along channel dimension
        x_up1 = F.relu(self.up1(x_up1))

        x_up2 = F.interpolate(x_up1, size=x1.shape[2:], mode='trilinear', align_corners=False)
        x_up2 = torch.cat([x_up2, x1], dim=1)  # Concatenate along channel dimension
        x_up2 = F.relu(self.up2(x_up2))

        x_final = self.conv_final(x_up2)

        return x_final

    def time_embedding(self, t):
        half_dim = self.model_channels // 2
        emb_log = torch.log(torch.tensor(10000.0, device=t.device))
        emb = emb_log / (half_dim - 1)
        emb = torch.exp(-emb * torch.arange(half_dim, device=t.device, dtype=torch.float32))
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb
