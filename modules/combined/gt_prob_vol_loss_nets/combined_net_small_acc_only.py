import torch
import torch.nn as nn

class CombinedProbVolNet_small(nn.Module):
    def __init__(self):
        super(CombinedProbVolNet_small, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, depth_prob_vol, semantic_prob_vol):
        """
        Args:
            depth_prob_vol (torch.Tensor): Tensor of shape [B, H, W]
            semantic_prob_vol (torch.Tensor): Tensor of shape [B, H, W]

        Returns:
            torch.Tensor: Combined probability map of shape [B, H, W]
        """
        # Stack depth and semantic probability volumes along channel dimension
        x = torch.stack([depth_prob_vol, semantic_prob_vol], dim=1)  # [B, 2, H, W]

        # Ensure the tensor is float and on the correct device
        x = x.float().to(self.conv[0].weight.device).contiguous()

        # Apply Conv2D layers
        x = self.conv(x)  # [B, 1, H, W]

        # Squeeze out the channel dimension to get [B, H, W]
        x = x.squeeze(1)  # [B, H, W]

        return x  # [B, H, W]
