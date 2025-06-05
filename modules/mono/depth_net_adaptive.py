"""
This module predicts the structural ray scan from a perspective image.
It returns a fixed number (40) of rays for any input image size.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as fn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import resnet50

from modules.network_utils import ConvBnReLU, Attention  # Assumed to be defined elsewhere

FIXED_WIDTH = 40

class AdaptivePoolWidth(nn.Module):
    """
    AdaptivePoolWidth pools only along the width dimension to a fixed value (FIXED_WIDTH)
    while leaving the height dimension unchanged.
    """
    def __init__(self, width):
        super().__init__()
        self.width = width

    def forward(self, x):
        # x: (N, C, H, W)
        H = x.shape[2]
        return F.adaptive_avg_pool2d(x, (H, self.width))


class depth_net(nn.Module):
    def __init__(self, d_min=0.1, d_max=15.0, d_hyp=-0.2, D=128) -> None:
        super().__init__()
        """
        The network extracts depth features and predicts a depth probability volume.
        For an input image, it returns a fixed number (40) of rays.
        
        - depth_feature extracts features and produces a tensor of shape (N, 40, D).
        - A softmax is applied to create a probability volume, and a weighted sum yields depth.
        """
        self.d_min = d_min
        self.d_max = d_max
        self.d_hyp = d_hyp
        self.D = D
        self.depth_feature = depth_feature_res()

    def forward(self, x, mask=None):
        # Extract depth features (rays) and attention weights.
        x, attn = self.depth_feature(x, mask)  # x: (N, 40, D)

        # Create a set of D disparity (or depth) hypotheses.
        d_vals = torch.linspace(
            self.d_min**self.d_hyp, self.d_max**self.d_hyp, self.D, device=x.device
        ) ** (1 / self.d_hyp)  # (D,)

        # Use softmax over the disparity dimension to obtain a probability volume.
        prob = F.softmax(x, dim=-1)  # (N, 40, D)

        # Weighted average to compute final depth.
        d = torch.sum(prob * d_vals, dim=-1)  # (N, 40)

        return d, attn, prob


class depth_feature_res(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        """
        The network extracts features using a ResNet backbone and applies an attention module.
        
        Flow:
        1. Input image is normalized and passed through ResNet50 (with dilation) to get a feature map.
        2. A convolution layer reduces the channels.
        3. Adaptive pooling fixes the width of the feature map to FIXED_WIDTH (40) regardless of input size.
        4. Vertical averaging produces a tensor of shape (N, 40, 128) which serves as query.
        5. 2D and 1D positional encodings are added.
        6. The query, key, and value are projected and passed through an attention module.
        """
        res50 = resnet50(pretrained=True, replace_stride_with_dilation=[False, False, True])
        self.resnet = nn.Sequential(
            IntermediateLayerGetter(res50, return_layers={"layer4": "feat"})
        )
        self.conv = ConvBnReLU(
            in_channels=2048, out_channels=128, kernel_size=3, padding=1, stride=1
        )

        # Adaptive pooling to force the feature map width to FIXED_WIDTH (40)
        self.adaptive_pool = AdaptivePoolWidth(FIXED_WIDTH)

        # 2D positional encoding MLP for spatial (x, y) positions.
        self.pos_mlp_2d = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh()
        )
        # 1D positional encoding MLP for the horizontal (ray) positions.
        self.pos_mlp_1d = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh()
        )

        # Attention block projection layers.
        self.q_proj = nn.Linear(160, 128, bias=False)
        self.k_proj = nn.Linear(160, 128, bias=False)
        self.v_proj = nn.Linear(160, 128, bias=False)
        self.attn = Attention()

    def forward(self, x, mask=None):
        # Normalize the input image.
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_normalized = (x - mean) / std
        
        # Extract features from the ResNet backbone.
        x = self.resnet(x_normalized)["feat"]  # (N, 1024, H, W)
        x = self.conv(x)  # (N, 128, H, W)
        
        # Apply adaptive pooling to set the width to FIXED_WIDTH.
        x = self.adaptive_pool(x)  # (N, 128, H, FIXED_WIDTH)
        
        # Get the spatial dimensions.
        H, W = x.shape[2], x.shape[3]  # W should now be FIXED_WIDTH (40)
        N = x.shape[0]

        # Reduce vertically: average over the height dimension to produce 40 rays.
        query = x.mean(dim=2)  # (N, 128, FIXED_WIDTH)
        query = query.permute(0, 2, 1)  # (N, FIXED_WIDTH, 128)

        # Reshape the feature map for positional encoding.
        x_reshaped = x.view(N, 128, H * W).permute(0, 2, 1)  # (N, H*W, 128)

        # 2D positional encoding: Create a grid based on the pooled feature map dimensions.
        pos_x = torch.linspace(0, 1, W, device=x.device) - 0.5  # (W,)
        pos_y = torch.linspace(0, 1, H, device=x.device) - 0.5  # (H,)
        pos_grid_y, pos_grid_x = torch.meshgrid(pos_y, pos_x, indexing='ij')  # each: (H, W)
        pos_grid_2d = torch.stack((pos_grid_x, pos_grid_y), dim=-1)  # (H, W, 2)
        pos_enc_2d = self.pos_mlp_2d(pos_grid_2d)  # (H, W, 32)
        pos_enc_2d = pos_enc_2d.view(1, H * W, 32).repeat(N, 1, 1)  # (N, H*W, 32)

        # Concatenate 2D positional encoding with the reshaped features.
        key_value = torch.cat((x_reshaped, pos_enc_2d), dim=-1)  # (N, H*W, 128+32)

        # 1D positional encoding for query along the width dimension.
        pos_v = torch.linspace(0, 1, W, device=x.device) - 0.5  # (W,)
        pos_enc_1d = self.pos_mlp_1d(pos_v.unsqueeze(-1))  # (W, 32)
        pos_enc_1d = pos_enc_1d.unsqueeze(0).repeat(N, 1, 1)  # (N, W, 32)
        query = torch.cat((query, pos_enc_1d), dim=-1)  # (N, W, 128+32)

        # Project query, key, and value.
        query = self.q_proj(query)  # (N, W, 128)
        key = self.k_proj(key_value)  # (N, H*W, 128)
        value = self.v_proj(key_value)  # (N, H*W, 128)

        # If a mask is provided, resize and adjust it to match key dimensions.
        if mask is not None:
            mask = fn.resize(mask, (H, W), fn.InterpolationMode.NEAREST).type(torch.bool)
            mask = torch.logical_not(mask)  # True indicates locations to mask out.
            mask = mask.view(mask.shape[0], 1, -1)  # (N, 1, H*W)
            mask = mask.repeat(1, W, 1)  # (N, W, H*W)

        # Compute attention.
        out, attn_w = self.attn(query, key, value, attn_mask=mask)  # (N, W, 128)
        return out, attn_w
