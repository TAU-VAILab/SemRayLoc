import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, Q, K, V, attn_mask=None):
        """
        one head attention
        Input:
            Q: queries, (N, L, D)
            K: keys, (N, S, D)
            V: values, (N, S, D)
            attn_mask: mask on the KV, (N, L, S)
        Output:
            queried_values: gathered values, (N, L, D)
            attn_weights: weights of the attention, (N, L, S)
        """
        # dot product
        QK = torch.einsum("nld,nsd->nls", Q, K)  # (N, L, S)
        if attn_mask is not None:
            QK[attn_mask] = -torch.inf  # this lead to 0 after softmax

        # softmax
        D = Q.shape[2]
        attn_weights = torch.softmax(QK / (D**2), dim=2)  # (N, L, S)

        # weighted average
        x = torch.einsum("nsd,nls->nld", V, attn_weights)  # (N, L, D)
        return x, attn_weights


class ELU_Plus(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        """
        Make the ELU > 0
        """
        return F.elu(x) + 1


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # conv + bn
        self.convbn = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
            nn.BatchNorm2d(self.out_channels),
        )

    def forward(self, x):
        x = self.convbn(x)
        return x


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.convbn = ConvBn(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

    def forward(self, x):
        x = self.convbn(x)
        x = F.relu(x)
        return x


class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # conv + relu
        self.convrelu = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.convrelu(x)
        return x


class ConvTransBn(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, output_padding
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        # convtrans + bn
        self.convtransbn = nn.Sequential(
            nn.ConvTranspose2d(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                output_padding=self.output_padding,
            ),
            nn.BatchNorm2d(self.out_channels),
        )

    def forward(self, x):
        x = self.convtransbn(x)
        return x


class ConvTransBnReLU(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, output_padding
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.convtransbn = ConvTransBn(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_padding=output_padding,
        )

    def forward(self, x):
        x = self.convtransbn(x)
        x = F.relu(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True) -> None:
        """
        This is a single residual block
        if downsample:
            x ----> conv(3x3, stride=2, in_ch, out_ch) -> bn -> relu ----> conv(3x3, out_ch, out_ch) -> bn ---(+)---> relu
                |------------------ conv(1x1, stride=2, in_ch, out_ch) -> bn ----------------------------------|
        else:
            x ----> conv(3x3, in_ch, out_ch) -> bn -> relu ----> conv(3x3, out_ch, out_ch) -> bn ---(+)---> relu
                |------------------------------------------------------------------------------------|
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample

        # first conv
        self.conv1 = ConvBnReLU(
            self.in_channels,
            self.out_channels,
            3,
            stride=2 if self.downsample else 1,
            padding=1,
        )

        # second conv
        self.conv2 = ConvBn(
            self.out_channels, self.out_channels, 3, stride=1, padding=1
        )

        if self.downsample:
            self.conv3 = ConvBn(
                self.in_channels, self.out_channels, 1, stride=2, padding=0
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            out = F.relu(out + self.conv3(x))
        else:
            out = F.relu(out + x)
        return out

