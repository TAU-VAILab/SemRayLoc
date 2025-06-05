# unet_diffusion_model_acc_only.py

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math

# Define necessary utility functions and classes
def conv_nd(dims, in_channels, out_channels, kernel_size, stride=1, padding=0):
    if dims == 1:
        return nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
    elif dims == 2:
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    else:
        raise NotImplementedError(f"conv_nd not implemented for dims={dims}")

def linear(in_features, out_features):
    return nn.Linear(in_features, out_features)

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

def normalization(channels):
    return nn.GroupNorm(32, channels)

def timestep_embedding(timesteps, dim):
    # Create sinusoidal timestep embeddings
    half_dim = dim // 2
    emb_scale = math.log(10000) / (half_dim - 1)
    # Ensure emb is on the same device as timesteps
    emb = th.exp(-emb_scale * th.arange(half_dim, device=timesteps.device, dtype=th.float32))
    emb = timesteps[:, None].float() * emb[None, :]  # Now emb and timesteps are on the same device
    emb = th.cat([th.sin(emb), th.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

class TimestepBlock(nn.Module):
    def forward(self, x, emb):
        pass

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class ResBlock(TimestepBlock):
    def __init__(self, channels, emb_channels, dropout, out_channels=None, 
                 use_conv=False, use_scale_shift_norm=False, dims=2, 
                 use_checkpoint=False, up=False, down=False):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, out_channels or channels, 3, padding=1),
        )
        self.updown = up or down
        if up:
            self.h_upd = nn.Upsample(scale_factor=2, mode='nearest')
            self.x_upd = nn.Upsample(scale_factor=2, mode='nearest')
        elif down:
            self.h_upd = nn.AvgPool2d(kernel_size=2, stride=2)
            self.x_upd = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.h_upd = self.x_upd = nn.Identity()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, 2 * (out_channels or channels) if use_scale_shift_norm else (out_channels or channels)),
        )
        self.out_layers = nn.Sequential(
            normalization(out_channels or channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, out_channels or channels, out_channels or channels, 3, padding=1)),
        )
        if out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, out_channels or channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, out_channels or channels, 1)

    def forward(self, x, emb):
        return self._forward(x, emb)

    def _forward(self, x, emb):
        if self.updown:
            h = self.in_layers[:-1](x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = self.in_layers[-1](h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None, None]
        if self.use_scale_shift_norm:
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = self.out_layers[0](h) * (1 + scale) + shift
            h = self.out_layers[1:](h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        out = self.skip_connection(x) + h
        return out

class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        dims=2,  # Overall model dimensions
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"channels {channels} not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        # Use dims=1 for the convolutions in attention
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))
        self.dims = dims  # Store dims for potential use in forward

    def forward(self, x):
        return self._forward(x)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x_in = x
        x = x.reshape(b, c, -1)
        x = self.norm(x)
        qkv = self.qkv(x)
        h = self.attention(qkv)
        h = self.proj_out(h)
        h = h.reshape(b, c, *spatial)
        out = x_in + h
        return out

class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.view(bs * self.n_heads, ch, length))
        out = a.view(bs, -1, length)
        return out



class UNetModel(nn.Module):
    """
    Adjusted UNet model for (B, H, W) inputs with conditional input of shape (B, H, W).
    """
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, self.in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            dims=dims,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else conv_nd(dims, ch, out_ch, 3, stride=2, padding=1)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                dims=dims,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])

        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            dims=dims,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else nn.Upsample(scale_factor=2, mode='nearest')
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, self.out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps, cond_input):
        """
        Apply the model to an input batch.

        :param x: an [N x H x W] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param cond_input: an [N x H x W] Tensor of conditional inputs.
        :return: an [N x H x W] Tensor of outputs.
        """
        # Expand dimensions to match expected input for convolution layers
        x = x.unsqueeze(1)  # Now x has shape [N, 1, H, W]
        cond_input = cond_input.unsqueeze(1)  # cond_input has shape [N, 1, H, W]

        # Concatenate conditions to the input
        x = th.cat([x, cond_input], dim=1)  # Now x has shape [N, 2, H, W]
        timesteps = timesteps.to(x.device)
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        hs = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h_skip = hs.pop()
            # Adjust h to match h_skip's spatial dimensions
            if h.shape[2:] != h_skip.shape[2:]:
                h = F.interpolate(h, size=h_skip.shape[2:], mode='nearest')
            h = th.cat([h, h_skip], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        out = self.out(h)

        # Squeeze out the channel dimension to return shape [N, H, W]
        return out.squeeze(1)
