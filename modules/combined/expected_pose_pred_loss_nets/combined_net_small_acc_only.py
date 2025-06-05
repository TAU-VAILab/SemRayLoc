import torch
import torch.nn as nn

class CombinedProbVolNet_small_acc_only(nn.Module):
    def __init__(self):
        super(CombinedProbVolNet_small_acc_only, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(2, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, depth_prob_vol, semantic_prob_vol):
        """
        Args:
            depth_prob_vol (torch.Tensor): Tensor of shape [B, H, W, D]
            semantic_prob_vol (torch.Tensor): Tensor of shape [B, H, W, D]

        Returns:
            torch.Tensor: Combined probability volume of shape [B, H, W]
        """
        # Stack depth and semantic probability volumes along channel dimension
        x = torch.stack([depth_prob_vol, semantic_prob_vol], dim=1)  # [B, 2, H, W, D]

        # Permute to [B, C=2, D, H, W] for Conv3D operation
        x = x.permute(0, 1, 4, 2, 3)  # [B, 2, D, H, W]

        # Ensure the tensor is float and on the correct device
        x = x.float().to(self.conv[0].weight.device).contiguous()

        # Apply Conv3D layers
        x = self.conv(x)  # [B, 1, D, H, W]

        # Squeeze out the channel dimension to get [B, D, H, W]
        x = x.squeeze(1)  # [B, D, H, W]

        # Aggregate the depth dimension by taking the mean along the D-axis
        x = x.mean(dim=1)  # [B, H, W]

        return x  # [B, H, W]
