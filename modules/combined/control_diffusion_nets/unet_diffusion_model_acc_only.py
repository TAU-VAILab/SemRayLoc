import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + shortcut)

class UNetModel_acc_only(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout,
        channel_mult,
        conv_resample,
        dims,
    ):
        super(UNetModel_acc_only, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(model_channels, model_channels * 4),
            nn.GELU(),
            nn.Linear(model_channels * 4, model_channels),
        )

        # Time embedding projections for different channel sizes
        self.time_projections = nn.ModuleDict()
        self.all_channels = set()

        # Encoder with stacked residual blocks
        self.encoder_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.encoder_channels = []  # Keep track of encoder output channels
        input_channels = in_channels
        for i in range(2):  # Two stages for downsampling
            out_channels_stage = model_channels * (2 ** i)
            blocks = nn.Sequential(
                *[
                    ResidualBlock(
                        input_channels if j == 0 else out_channels_stage,
                        out_channels_stage,
                    )
                    for j in range(num_res_blocks)
                ]
            )
            self.encoder_blocks.append(blocks)
            self.pools.append(nn.MaxPool2d(2))
            input_channels = out_channels_stage
            self.encoder_channels.append(out_channels_stage)
            self.all_channels.add(out_channels_stage)

        # Decoder with stacked residual blocks
        self.decoder_blocks = nn.ModuleList()
        decoder_in_channels = []
        decoder_out_channels = []

        # Compute decoder channels based on encoder channels
        x_channels = self.encoder_channels[-1]  # Start with the last encoder channel size
        for i in reversed(range(len(self.encoder_channels))):
            encoder_channel = self.encoder_channels[i]

            in_channels_stage = x_channels + encoder_channel
            out_channels_stage = encoder_channel

            decoder_in_channels.append(in_channels_stage)
            decoder_out_channels.append(out_channels_stage)
            self.all_channels.update([in_channels_stage, out_channels_stage])

            x_channels = out_channels_stage  # Update x_channels for the next iteration

        # Do NOT reverse the lists
        # decoder_in_channels = decoder_in_channels[::-1]
        # decoder_out_channels = decoder_out_channels[::-1]

        # Initialize decoder blocks with correct channels
        for in_ch, out_ch in zip(decoder_in_channels, decoder_out_channels):
            blocks = nn.Sequential(
                *[
                    ResidualBlock(in_ch if j == 0 else out_ch, out_ch)
                    for j in range(num_res_blocks)
                ]
            )
            self.decoder_blocks.append(blocks)

        # Define time projections for all unique channel sizes
        for channels in self.all_channels:
            self.time_projections[str(channels)] = nn.Linear(
                model_channels, channels
            )

        # Final convolution layer
        self.conv_final = nn.Conv2d(self.encoder_channels[0], out_channels, kernel_size=1)

    def forward(self, x, t, cond):
        # x: [B, 1, H, W]
        # t: [B]
        # cond: [B, 2, H, W]

        # Concatenate x and cond along the channel dimension
        x = torch.cat([x, cond], dim=1)  # x: [B, 3, H, W]

        # Get time embedding and pass it through an MLP
        t_embed = self.time_embedding(t)  # [B, model_channels]
        t_embed = self.time_mlp(t_embed)  # [B, model_channels]

        # Encoder
        encoder_outputs = []
        for i, (block, pool) in enumerate(zip(self.encoder_blocks, self.pools)):
            x = block(x)  # Apply residual blocks

            # Project t_embed to match x's channel dimension
            current_channels = x.size(1)
            t_proj = self.time_projections[str(current_channels)]
            t_embed_proj = t_proj(t_embed)  # [B, current_channels]
            t_embed_proj = t_embed_proj[:, :, None, None]  # [B, current_channels, 1, 1]

            x = x + t_embed_proj  # Add time embedding after each stage
            encoder_outputs.append(x)
            x = pool(x)

        # Decoder
        for i, block in enumerate(self.decoder_blocks):
            x = F.interpolate(
                x,
                size=encoder_outputs[-(i + 1)].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            x = torch.cat([x, encoder_outputs[-(i + 1)]], dim=1)  # Concatenate with corresponding encoder output
            x = block(x)  # Apply residual blocks

            # Project t_embed to match x's channel dimension
            current_channels = x.size(1)
            t_proj = self.time_projections[str(current_channels)]
            t_embed_proj = t_proj(t_embed)  # [B, current_channels]
            t_embed_proj = t_embed_proj[:, :, None, None]  # [B, current_channels, 1, 1]

            x = x + t_embed_proj  # Add time embedding after each stage

        x_final = self.conv_final(x)
        return x_final

    def time_embedding(self, t):
        half_dim = self.model_channels // 2
        emb_log = torch.log(torch.tensor(10000.0, device=t.device))
        emb = emb_log / (half_dim - 1)
        emb = torch.exp(-emb * torch.arange(half_dim, device=t.device, dtype=torch.float32))
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb
