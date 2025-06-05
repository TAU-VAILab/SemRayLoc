# diffusion_utils.py
import torch

class DiffusionScheduler:
    def __init__(self, num_timesteps):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        beta = torch.linspace(1e-4, 0.02, num_timesteps, dtype=torch.float32, device=device)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.alpha_bar = alpha_bar
        self.beta = beta
        self.alpha = alpha
