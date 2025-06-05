import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleWeightPredictorNet(nn.Module):
    def __init__(self, net_size='small'):
        super(SingleWeightPredictorNet, self).__init__()
        self.net_size = net_size

        if net_size == 'small':
            self.conv_layers = nn.Sequential(
                nn.Conv2d(2, 16, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(32, 16, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True)
            )
        elif net_size == 'medium':
            self.conv_layers = nn.Sequential(
                nn.Conv2d(2, 32, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
            )
        elif net_size == 'large':
            self.conv_layers = nn.Sequential(
                nn.Conv2d(2, 64, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True)
            )
        else:
            raise ValueError(f"Unsupported net_size: {net_size}")

        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.fc = nn.Linear(16 if net_size == 'small' else 32 if net_size == 'medium' else 512, 2)

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.constant_(module.weight, 0)  # Set weights to zero
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)  # Set biases to zero
                    
    def forward(self, depth_map, semantic_map):
        x = torch.stack([depth_map, semantic_map], dim=1)  # Shape: [B, 2, H, W]
        x = self.conv_layers(x)  # Convolutional layers
        x = self.global_pool(x)  # Global average pooling to [B, C, 1, 1]
        x = torch.flatten(x, 1)  # Flatten to [B, C]
        x = self.fc(x)  # Fully connected layer to [B, 2]
        weights = F.softmax(x, dim=1)  # Ensure the outputs sum to 1
        return weights
