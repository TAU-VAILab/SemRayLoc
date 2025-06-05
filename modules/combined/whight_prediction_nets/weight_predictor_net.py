import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttention(nn.Module):
    """ Self-Attention Layer """
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.channel_in = in_dim
        intermediate_dim = max(1, in_dim // 8)

        # Convolution layers to generate query, key, and value
        self.query_conv = nn.Conv2d(in_dim, intermediate_dim, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_dim, intermediate_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        
        # Softmax for attention map
        self.softmax = nn.Softmax(dim=-1)
        
        # Scaling parameter
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        """
        Inputs:
            x: Input feature maps (B, C, H, W)
        Returns:
            out: Self-attention value + input feature
        """
        batch_size, C, width, height = x.size()
        
        # Generate query, key, and value matrices
        proj_query = self.query_conv(x).view(batch_size, -1, width * height)  # (B, C', N)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)      # (B, C', N)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)  # (B, C, N)
        
        # Compute attention scores
        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)             # (B, N, N)
        attention = self.softmax(energy)                                      # (B, N, N)
        
        # Apply attention to the values
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))               # (B, C, N)
        out = out.view(batch_size, C, width, height)
        
        # Apply scaling and residual connection
        out = self.gamma * out + x
        return out

class WeightPredictorNet(nn.Module):
    def __init__(self, net_size='medium'):
        super(WeightPredictorNet, self).__init__()
        self.net_size = net_size

        if net_size == 'small':
            # Existing small network (unchanged)
            self.conv_layers = nn.Sequential(
                nn.Conv2d(2, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            self.output_layer = nn.Conv2d(16, 2, kernel_size=1)
        
        if net_size == 'medium':
            # Initial convolution layers
            self.conv_layers_before_att = nn.Sequential(
                nn.Conv2d(2, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),  # Downsample by a factor of 2
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),  # Downsample by another factor of 2 (Total downsample x4)
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            
            # Apply self-attention at lower resolution
            self.attention = SelfAttention(in_dim=64)
            
            # Additional convolution layers after attention
            self.conv_layers_after_att = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            
            # Upsample back to original resolution
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            
            # Final output layer
            self.output_layer = nn.Conv2d(32, 2, kernel_size=1)
        
        elif net_size == 'large':
            # Initial convolution layers
            self.layers = nn.Sequential(
                nn.Conv2d(2, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),  # Downsample by a factor of 2
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),  # Downsample by another factor of 2 (Total downsample x4)
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                SelfAttention(in_dim=64),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                SelfAttention(in_dim=128),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            
            # Final output layer
            self.output_layer = nn.Conv2d(32, 2, kernel_size=1)
        elif net_size == 'x-large':
            # Initial convolution layers
            self.layers = nn.Sequential(
                nn.Conv2d(2, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),  # Downsample by a factor of 2
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),  # Downsample by another factor of 2 (Total downsample x4)
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                SelfAttention(in_dim=64),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                SelfAttention(in_dim=128),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                SelfAttention(in_dim=256),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

            
            # Final output layer
            self.output_layer = nn.Conv2d(32, 2, kernel_size=1)
        elif net_size == 'xx-large':
            # Initial convolution layers
            self.layers = nn.Sequential(
                nn.Conv2d(2, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),  # Downsample by a factor of 2
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),  # Downsample by another factor of 2 (Total downsample x4)
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                SelfAttention(in_dim=64),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                SelfAttention(in_dim=128),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                SelfAttention(in_dim=256),
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                SelfAttention(in_dim=512),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

            
            # Final output layer
            self.output_layer = nn.Conv2d(32, 2, kernel_size=1)
        else:
            raise ValueError(f"Unsupported net_size: {net_size}")
    
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Use Kaiming (He) initialization
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, SelfAttention):
                # SelfAttention parameters are initialized by default
                pass
                
    def forward(self, depth_map, semantic_map):
        x = torch.stack([depth_map, semantic_map], dim=1)  # [B, 2, H, W]
        
        if self.net_size == 'medium':
            # Initial convolutions and downsampling
            x = self.conv_layers_before_att(x)
            # Apply self-attention at lower resolution
            x = self.attention(x)
            # Additional convolutions
            x = self.conv_layers_after_att(x)
            # Upsample back to original resolution
            x = self.upsample(x)
        elif self.net_size == 'large':
            # Initial convolutions and downsampling
            x = self.layers(x)
        elif self.net_size == 'x-large':
            # Initial convolutions and downsampling
            x = self.layers(x)
        elif self.net_size == 'xx-large':
            # Initial convolutions and downsampling
            x = self.layers(x)
        else:
            x = self.conv_layers(x)
        
        # Output layer
        weight_map = self.output_layer(x)  # [B, 2, H, W]
        
        # Apply softmax over the channel dimension
        weight_map = F.softmax(weight_map, dim=1)
        
        # Split the weight maps
        weight_map1 = weight_map[:, 0, :, :]
        weight_map2 = weight_map[:, 1, :, :]
        return weight_map1, weight_map2

