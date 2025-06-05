# diffusion_utils.py

import torch

class DiffusionScheduler:
    def __init__(self, num_timesteps, beta_start=0.0001, beta_end=0.02, schedule='linear'):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule = schedule

        self.beta = self.get_beta_schedule()
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def get_beta_schedule(self):
        if self.schedule == 'linear':
            return torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        elif self.schedule == 'cosine':
            return self.cosine_beta_schedule()
        else:
            raise NotImplementedError(f"Unknown beta schedule: {self.schedule}")

    def cosine_beta_schedule(self):
        """
        Implements the cosine schedule as proposed in
        https://openreview.net/forum?id=-NEXDKk8gZ
        """
        steps = self.num_timesteps + 1
        s = 0.008
        x = torch.linspace(0, self.num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
