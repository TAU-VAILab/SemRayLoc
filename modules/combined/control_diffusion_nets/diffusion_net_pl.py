# diffusion_net_pl.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
import numpy as np

from modules.combined.control_diffusion_nets.unet_diffusion_model import UNetModel
from modules.combined.control_diffusion_nets.unet_diffusion_model_acc_only import UNetModel_acc_only
from modules.combined.control_diffusion_nets.diffusion_utils import DiffusionScheduler

class ConditionalDiffusionNetPL(LightningModule):
    def __init__(self, config, lr=1e-4, log_dir='logs', net_size="small", acc_only = False):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters for checkpointing

        # Initialize the U-Net diffusion model
        if acc_only:
            self.model = UNetModel_acc_only(
                in_channels=3,   # Depth and semantic inputs
                model_channels=config.model_channels,  # Adjust based on net_size
                out_channels=1,  # Predicting noise for the target probability volume
                num_res_blocks=config.num_res_blocks,
                attention_resolutions=[],
                dropout=config.dropout,
                channel_mult=(1, 2, 4, 8),
                conv_resample=config.conv_resample,
                dims=config.dims
            )
        else:
            self.model = UNetModel(
                in_channels=3,   # Depth and semantic inputs
                model_channels=config.model_channels,  # Adjust based on net_size
                out_channels=1,  # Predicting noise for the target probability volume
                num_res_blocks=config.num_res_blocks,
                attention_resolutions=[],
                dropout=config.dropout,
                channel_mult=(1, 2, 4, 8),
                conv_resample=config.conv_resample,
                dims=config.dims
            )


        self.lr = lr
        self.log_dir = log_dir  # Directory to save results
        self.config = config  # Store config for use
        self.acc_only = config.acc_only

        # Diffusion parameters
        self.num_timesteps = config.num_timesteps
        self.scheduler = DiffusionScheduler(self.num_timesteps)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        return optimizer        

    def forward(self, x, t, cond_input):
        return self.model(x, t, cond_input)

    def training_step(self, batch, batch_idx):    
        if self.acc_only:
            prob_vol_depth = batch['prob_vol_depth']  # [B, H, W, D]
            prob_vol_semantic = batch['prob_vol_semantic']  # [B, H, W, D]
            prob_vol_gt = batch['prob_vol_gt']

            B = prob_vol_gt.shape[0]
        
            cond_input = torch.stack([prob_vol_depth, prob_vol_semantic], dim=1)  # [B, 2, H, W, D]

            prob_vol_gt = prob_vol_gt.unsqueeze(1)  # [B, 1, H, W, D]

            # Sample random timestep for each instance in the batch
            t = torch.randint(0, self.num_timesteps, (B,), device=self.device).long()

            # Forward diffusion (adding noise)
            noise = torch.randn_like(prob_vol_gt)
            alpha_bar_t = self.scheduler.alpha_bar.to(self.device)[t].view(-1, 1, 1, 1)
            noisy_prob_vol = torch.sqrt(alpha_bar_t) * prob_vol_gt + torch.sqrt(1 - alpha_bar_t) * noise

            # Predict the noise
            noise_pred = self(noisy_prob_vol, t, cond_input)

        else:
            prob_vol_depth = batch['prob_vol_depth']  # [B, H, W, D]
            prob_vol_semantic = batch['prob_vol_semantic']  # [B, H, W, D]
            prob_vol_gt = batch['prob_vol_gt']

            B = prob_vol_gt.shape[0]

            # Prepare conditional input
            cond_input = torch.stack([prob_vol_depth, prob_vol_semantic], dim=1)  # [B, 2, H, W, D]

            prob_vol_gt = prob_vol_gt.unsqueeze(1)  # [B, 1, H, W, D]

            # Sample random timestep for each instance in the batch
            t = torch.randint(0, self.num_timesteps, (B,), device=self.device).long()

            # Forward diffusion (adding noise)
            noise = torch.randn_like(prob_vol_gt)
            alpha_bar_t = self.scheduler.alpha_bar.to(self.device)[t].view(-1, 1, 1, 1, 1)
            noisy_prob_vol = torch.sqrt(alpha_bar_t) * prob_vol_gt + torch.sqrt(1 - alpha_bar_t) * noise

            # Predict the noise
            noise_pred = self(noisy_prob_vol, t, cond_input)

        # Compute loss
        loss = F.mse_loss(noise_pred, noise)

        self.log('loss_train', loss.detach().cpu(), on_step=True, on_epoch=True, prog_bar=True, batch_size=B)

        return loss

    def validation_step(self, batch, batch_idx):
        if self.acc_only:
            prob_vol_depth = batch['prob_vol_depth']  # [B, H, W, D]
            prob_vol_semantic = batch['prob_vol_semantic']  # [B, H, W, D]
            prob_vol_gt = batch['prob_vol_gt']

            B = prob_vol_gt.shape[0]
        
            cond_input = torch.stack([prob_vol_depth, prob_vol_semantic], dim=1)  # [B, 2, H, W, D]

            prob_vol_gt = prob_vol_gt.unsqueeze(1)  # [B, 1, H, W, D]

            # Sample random timestep for each instance in the batch
            t = torch.randint(0, self.num_timesteps, (B,), device=self.device).long()

            # Forward diffusion (adding noise)
            noise = torch.randn_like(prob_vol_gt)
            alpha_bar_t = self.scheduler.alpha_bar.to(self.device)[t].view(-1, 1, 1, 1)
            noisy_prob_vol = torch.sqrt(alpha_bar_t) * prob_vol_gt + torch.sqrt(1 - alpha_bar_t) * noise

            # Predict the noise
            noise_pred = self(noisy_prob_vol, t, cond_input)

        else:
            prob_vol_depth = batch['prob_vol_depth']  # [B, H, W, D]
            prob_vol_semantic = batch['prob_vol_semantic']  # [B, H, W, D]
            prob_vol_gt = batch['prob_vol_gt']

            B = prob_vol_gt.shape[0]

            # Prepare conditional input
            cond_input = torch.stack([prob_vol_depth, prob_vol_semantic], dim=1)  # [B, 2, H, W, D]

            prob_vol_gt = prob_vol_gt.unsqueeze(1)  # [B, 1, H, W, D]

            # Sample random timestep for each instance in the batch
            t = torch.randint(0, self.num_timesteps, (B,), device=self.device).long()

            # Forward diffusion (adding noise)
            noise = torch.randn_like(prob_vol_gt)
            alpha_bar_t = self.scheduler.alpha_bar.to(self.device)[t].view(-1, 1, 1, 1, 1)
            noisy_prob_vol = torch.sqrt(alpha_bar_t) * prob_vol_gt + torch.sqrt(1 - alpha_bar_t) * noise

            # Predict the noise
            noise_pred = self(noisy_prob_vol, t, cond_input)

        # Compute loss
        loss = F.mse_loss(noise_pred, noise)

        self.log('loss-valid', loss.detach().cpu(), on_step=False, on_epoch=True, prog_bar=True, batch_size=B)

        return loss

    
    def ddim_sampling(self, cond_input, num_steps=50, acc_only=False, number_of_evaluations=1):
        """
        Perform DDIM sampling to generate the combined probability map.

        Args:
            cond_input (torch.Tensor): Conditional input tensor 
            num_steps (int): Number of sampling steps
            acc_only (bool): Flag to indicate if only accuracy is considered
            average (int): Number of runs to average over (default is 1)

        Returns:
            torch.Tensor: Averaged generated combined probability map 
        """
        self.model.eval()
        results = []

        for _ in range(number_of_evaluations):
            with torch.no_grad():
                if acc_only:
                    B, _, H, W = cond_input.shape
                    device = cond_input.device

                    # Initialize with random noise
                    x = torch.randn(B, 1, H, W, device=device)

                    # Prepare timesteps
                    timesteps = torch.linspace(self.num_timesteps - 1, 0, num_steps, device=device).long()
                    alphas_cumprod = self.scheduler.alpha_bar.to(device)
                    alphas_cumprod_prev = torch.cat([alphas_cumprod[0:1], alphas_cumprod[:-1]])

                    for i, t in enumerate(timesteps):
                        t = t.long()
                        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

                        # Predict noise
                        noise_pred = self.model(x, t_tensor, cond_input)

                        # Compute parameters
                        alpha_t = alphas_cumprod[t].view(-1, 1, 1, 1)
                        alpha_t_prev = alphas_cumprod_prev[t].view(-1, 1, 1, 1)
                        beta_t = self.scheduler.beta[t].view(-1, 1, 1, 1)

                        sqrt_alpha_t = torch.sqrt(alpha_t)
                        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
                        sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
                        sqrt_one_minus_alpha_t_prev = torch.sqrt(1 - alpha_t_prev)

                        # Update x using the DDIM update rule
                        x_prev = sqrt_alpha_t_prev * (x - sqrt_one_minus_alpha_t * noise_pred / sqrt_alpha_t) + sqrt_one_minus_alpha_t_prev * noise_pred

                        x = x_prev

                    results.append(x)

                else:
                    with torch.no_grad():
                        B, _, H, W, D = cond_input.shape
                        device = cond_input.device

                        # Initialize with random noise
                        x = torch.randn(B, 1, H, W, D, device=device)

                        # Create a list of timesteps
                        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_steps, device=device).long()

                        for i, t in enumerate(timesteps):
                            t = t.long()
                            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

                            # Predict noise
                            noise_pred = self.model(x, t_tensor, cond_input)

                            # Compute parameters
                            beta_t = self.scheduler.beta[t].view(-1, 1, 1, 1, 1).type_as(x)
                            sqrt_alpha_t = torch.sqrt(self.scheduler.alpha[t]).view(-1, 1, 1, 1, 1).type_as(x)
                            sqrt_one_minus_alpha_t = torch.sqrt(1 - self.scheduler.alpha[t]).view(-1, 1, 1, 1, 1).type_as(x)

                            # Update x
                            if i < num_steps - 1:
                                x = (1 / sqrt_alpha_t) * (x - (beta_t / sqrt_one_minus_alpha_t) * noise_pred)
                            else:
                                x = noise_pred  # At the last step, output the prediction

                        results.append(x)

        # Average the results over the specified number of runs
        return torch.stack(results).mean(dim=0) if number_of_evaluations > 1 else results[0]

