import torch
import torch.nn as nn

# Self-Attention Layer for 3D data
class SelfAttention3D(nn.Module):
    """ Self-Attention Layer for 3D data """
    def __init__(self, in_dim):
        super(SelfAttention3D, self).__init__()
        self.channel_in = in_dim
        intermediate_dim = max(1, in_dim // 8)
    
        # Convolution layers to generate query, key, and value
        self.query_conv = nn.Conv3d(in_dim, intermediate_dim, kernel_size=1)
        self.key_conv   = nn.Conv3d(in_dim, intermediate_dim, kernel_size=1)
        self.value_conv = nn.Conv3d(in_dim, in_dim, kernel_size=1)
        
        # Softmax for attention map
        self.softmax = nn.Softmax(dim=-1)
        
        # Scaling parameter
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        """
        Inputs:
            x: Input feature maps (B, C, D, H, W)
        Returns:
            out: Self-attention value + input feature
        """
        batch_size, C, D, H, W = x.size()
        
        # Generate query, key, and value matrices
        proj_query = self.query_conv(x).view(batch_size, -1, D * H * W)  # (B, C', N)
        proj_key = self.key_conv(x).view(batch_size, -1, D * H * W)      # (B, C', N)
        proj_value = self.value_conv(x).view(batch_size, -1, D * H * W)  # (B, C, N)
        
        # Compute attention scores
        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)        # (B, N, N)
        attention = self.softmax(energy)                                 # (B, N, N)
        
        # Apply attention to the values
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))          # (B, C, N)
        out = out.view(batch_size, C, D, H, W)
        
        # Apply scaling and residual connection
        out = self.gamma * out + x
        return out

class MapPredictorNet(nn.Module):
    def __init__(self, net_size='medium'):
        super(MapPredictorNet, self).__init__()
        self.net_size = net_size
        
        if net_size == 'small':
            # Small network configuration
            self.conv_layers = nn.Sequential(
                nn.Conv3d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            # Global average pooling and output layer
            self.global_pool = nn.AdaptiveAvgPool3d(1)  # Output size (B, C, 1, 1, 1)
            self.fc = nn.Linear(16, 3)  # Output logits over 6 classes
        
        elif net_size == 'medium':
            # Medium network configuration with adjusted pooling
            self.conv_layers_before_att = nn.Sequential(
                nn.Conv3d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
                nn.Conv3d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
                nn.Conv3d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
                nn.Conv3d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
                nn.Conv3d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            
            # Self-attention layers
            self.attention = nn.Sequential(
                SelfAttention3D(in_dim=64),
                nn.Conv3d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                SelfAttention3D(in_dim=64)
            )
            
            # Additional convolution layers after attention
            self.conv_layers_after_att = nn.Sequential(
                nn.Conv3d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            
            # Upsampling layers
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False),
                nn.Conv3d(128, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False),
                nn.Conv3d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False),
                nn.Conv3d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False),
                nn.Conv3d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            
            # Global average pooling and output layer
            self.global_pool = nn.AdaptiveAvgPool3d(1)  # Output size (B, C, 1, 1, 1)
            self.fc = nn.Linear(32, 3)  # Output logits over 6 classes
                        
        elif net_size == 'large':
            # Large network configuration (adjust as needed)
            self.conv_layers_before_att = nn.Sequential(
                nn.Conv3d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
                nn.Conv3d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
                nn.Conv3d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
                nn.Conv3d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
                nn.Conv3d(128, 256, kernel_size=3, padding=1),    
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
                nn.Conv3d(256, 256, kernel_size=3, padding=1),    
                nn.ReLU(inplace=True),
            )
            
            # Self-attention layers
            self.attention = nn.Sequential(
                SelfAttention3D(in_dim=256),
                nn.Conv3d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                SelfAttention3D(in_dim=256)
            )
            
            # Additional convolution layers after attention
            self.conv_layers_after_att = nn.Sequential(
                nn.Conv3d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            
            # Upsampling layers
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False),
                nn.Conv3d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False),
                nn.Conv3d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False),
                nn.Conv3d(128, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False),
                nn.Conv3d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False),
                nn.Conv3d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            
            # Global average pooling and output layer
            self.global_pool = nn.AdaptiveAvgPool3d(1)  # Output size (B, C, 1, 1, 1)
            self.fc = nn.Linear(32, 3)  # Output logits over 6 classes
        
        else:
            raise ValueError(f"Unsupported net_size: {net_size}")
            
    def forward(self, x):
        # x is [B, 2, H, W, O]
        B, C, H, W, O = x.shape

        # Permute to match Conv3D input shape: [B, C, D, H, W]
        x = x.permute(0, 1, 4, 2, 3)  # x is [B, 2, O, H, W]

        if self.net_size in ['medium', 'large']:
            # Initial convolutions and downsampling
            x = self.conv_layers_before_att(x)
            # Apply self-attention at lower resolution
            x = self.attention(x)
            # Additional convolutions
            x = self.conv_layers_after_att(x)
            # Upsample back to original resolution
            x = self.upsample(x)
        else:
            x = self.conv_layers(x)
        
        # Global average pooling
        x = self.global_pool(x)  # [B, C, 1, 1, 1]
        x = x.view(B, -1)  # Flatten to [B, C]
        logits = self.fc(x)  # [B, 6]

        return logits
