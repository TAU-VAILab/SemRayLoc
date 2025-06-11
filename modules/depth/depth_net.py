import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as fn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import *

from modules.network_utils import *

class depth_net(nn.Module):
    def __init__(self, d_min=0.1, d_max=15.0, d_hyp=-0.2, D=128) -> None:
        super().__init__()

        self.d_min = d_min
        self.d_max = d_max
        self.d_hyp = d_hyp
        self.D = D
        self.depth_feature = depth_feature_res()

    def forward(self, x, mask=None):
        x, attn = self.depth_feature(x, mask)

        d_vals = torch.linspace(
            self.d_min**self.d_hyp, self.d_max**self.d_hyp, self.D, device=x.device
        ) ** (
            1 / self.d_hyp
        )

        prob = F.softmax(x, dim=-1)

        d = torch.sum(prob * d_vals, dim=-1)

        return d, attn, prob


class depth_feature_res(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        res50 = resnet50(
            pretrained=True, replace_stride_with_dilation=[False, False, True]
        )
        self.resnet = nn.Sequential(
            IntermediateLayerGetter(res50, return_layers={"layer4": "feat"})
        )
        self.conv = ConvBnReLU(
            in_channels=2048, out_channels=128, kernel_size=3, padding=1, stride=1
        )

        self.pos_mlp_2d = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh()
        )
        self.pos_mlp_1d = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh()
        )

        self.q_proj = nn.Linear(160, 128, bias=False)
        self.k_proj = nn.Linear(160, 128, bias=False)
        self.v_proj = nn.Linear(160, 128, bias=False)
        self.attn = Attention()

    def forward(self, x, mask=None):
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_normalized = (x - mean) / std
        
        x = self.resnet(x_normalized)["feat"]
        x = self.conv(x)
        
        fH, fW = list(x.shape[2:])
        N = x.shape[0]

        query = x.mean(dim=2)

        query = query.permute(0, 2, 1)

        x = x.view(list(x.shape[:2]) + [-1])
        x = x.permute(0, 2, 1)

        pos_x = torch.linspace(0, 1, fW, device=x.device) - 0.5
        pos_y = torch.linspace(0, 1, fH, device=x.device) - 0.5
        pos_grid_2d_x, pos_grid_2d_y = torch.meshgrid(pos_x, pos_y)
        pos_grid_2d = torch.stack((pos_grid_2d_x, pos_grid_2d_y), dim=-1)
        pos_enc_2d = self.pos_mlp_2d(pos_grid_2d)
        pos_enc_2d = pos_enc_2d.reshape((1, -1, 32))
        pos_enc_2d = pos_enc_2d.repeat((N, 1, 1))
        x = torch.cat((x, pos_enc_2d), dim=-1)

        pos_v = torch.linspace(0, 1, fW, device=x.device) - 0.5
        pos_enc_1d = self.pos_mlp_1d(pos_v.reshape((-1, 1)))
        pos_enc_1d = pos_enc_1d.reshape((1, -1, 32)).repeat((N, 1, 1))
        query = torch.cat((query, pos_enc_1d), dim=-1)

        query = self.q_proj(query)
        key = self.k_proj(x)
        value = self.v_proj(x)

        if mask is not None:
            mask = fn.resize(mask, (fH, fW), fn.InterpolationMode.NEAREST).type(
                torch.bool
            )
            mask = torch.logical_not(
                mask
            )
            mask = mask.reshape((mask.shape[0], 1, -1))
            mask = mask.repeat(1, fW, 1)
        x, attn_w = self.attn(query, key, value, attn_mask=mask)

        return x, attn_w
