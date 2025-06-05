# diffusion_net_pl.py

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from modules.combined.control_diffusion_nets.crowed_diff_unet.unet_diffusion_crowed_diff import UNetModel_acc_only
from modules.combined.control_diffusion_nets.crowed_diff_unet.diffusion_utils import DiffusionScheduler
from utils.localization_utils import finalize_localization_acc_only

class ConditionalDiffusionNetPLCrowedDiff(LightningModule):
    def __init__(self, config, lr=1e-4, log_dir='logs', net_size="small", acc_only = True):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters for checkpointing

        # Initialize the U-Net diffusion model for acc_only case
        if config.acc_only:
            self.model = UNetModel_acc_only(
                in_channels=3,   # Input channels: x (1) + cond_input (2)
                model_channels=config.model_channels,  
                out_channels=1,  # Predicting noise for the target probability map
                num_res_blocks=config.num_res_blocks,
                attention_resolutions=[],
                dropout=config.dropout,
                channel_mult=config.channel_mult,
                conv_resample=config.conv_resample,
                dims=2,  # Ensure dims=2 for 2D data
                use_checkpoint=config.use_checkpoint,
                use_fp16=config.use_fp16,
            )
        else:
            # Handle the non-acc_only case (3D data)
            # Make sure to import the appropriate UNetModel for 3D data
            pass  # Implement as needed

        self.lr = lr
        self.log_dir = log_dir  # Directory to save results
        self.config = config  # Store config for use
        # self.loss_type = config.loss_type #mse/cross-entropy
        # Diffusion parameters
        self.num_timesteps = config.num_timesteps
        self.scheduler = DiffusionScheduler(self.num_timesteps)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        return optimizer        

    def forward(self, cond_input, num_steps=50, number_of_evaluations=1):
        return self.ddim_sampling(cond_input, num_steps=num_steps, number_of_evaluations=number_of_evaluations)
    
    def training_step(self, batch, batch_idx):    
        if self.config.acc_only:
            # Load inputs
            prob_vol_depth = batch['prob_vol_depth'].to(self.device)  # [B, H, W]
            prob_vol_semantic = batch['prob_vol_semantic'].to(self.device)  # [B, H, W]
            prob_vol_gt = batch['prob_vol_gt'].to(self.device)  # [B, H, W]
            ref_pose = batch['ref_pose'].to(self.device)  # [B, N], where N >= 2 (x, y, ...)

            B, H, W = prob_vol_gt.shape

            cond_input = torch.stack([prob_vol_depth, prob_vol_semantic], dim=1)  # [B, 2, H, W]
            prob_vol_gt = prob_vol_gt.unsqueeze(1)  # [B, 1, H, W]

            # Sample random timestep for each instance in the batch
            t = torch.randint(0, self.num_timesteps, (B,), device=self.device).long()

            # Forward diffusion (adding noise)
            noise = torch.randn_like(prob_vol_gt)
            alpha_bar_t = self.scheduler.alpha_bar.to(self.device)[t].view(-1, 1, 1, 1)
            noisy_prob_vol = torch.sqrt(alpha_bar_t) * prob_vol_gt + torch.sqrt(1 - alpha_bar_t) * noise

            # Predict the noise
            noise_pred = self.model(noisy_prob_vol, t, cond_input)

            # Reconstruct the predicted map
            pred_prob_vol = (noisy_prob_vol - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)

            # Ensure the predicted map is non-negative and normalized
            pred_prob_vol = F.relu(pred_prob_vol)
            pred_prob_vol = pred_prob_vol / (pred_prob_vol.sum(dim=(2, 3), keepdim=True) + 1e-8)

            # Compute expected coordinates
            pred_x, pred_y = self.compute_expected_coordinates(pred_prob_vol)
            pred_x /= 10
            pred_y /= 10
            
            # Get ground truth coordinates
            gt_x = ref_pose[:, 0] 
            gt_y = ref_pose[:, 1] 

            # Compute positional loss
            positional_loss = torch.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2).mean()

            # Compute MSE loss on noise prediction
            mse_loss = F.mse_loss(noise_pred, noise)

            # Total loss
            total_loss = 1 * mse_loss + self.config.positional_loss_weight * positional_loss

            # Logging
            self.log('loss_train', total_loss.detach().cpu(), on_step=True, on_epoch=True, prog_bar=True, batch_size=B)
            self.log('mse_loss', mse_loss.detach().cpu(), on_step=True, on_epoch=True, batch_size=B)
            self.log('positional_loss', positional_loss.detach().cpu(), on_step=True, on_epoch=True, batch_size=B)

            return total_loss

        else:
            # Handle the non-acc_only case (3D data)
            pass  # Implement as needed

    def compute_expected_coordinates(self, pred_map):
        B, _, H, W = pred_map.shape
        device = pred_map.device

        # Create coordinate grids
        y_coords = torch.arange(0, H, device=device).view(1, 1, H, 1)  # Shape: [1, 1, H, 1]
        x_coords = torch.arange(0, W, device=device).view(1, 1, 1, W)  # Shape: [1, 1, 1, W]

        # Compute expected coordinates
        expected_y = (pred_map * y_coords).sum(dim=(2, 3))  # Shape: [B, 1]
        expected_x = (pred_map * x_coords).sum(dim=(2, 3))  # Shape: [B, 1]

        return expected_x.squeeze(1), expected_y.squeeze(1)
    
    def recall_validation(self, batch, cond_input):
        acc_records = []
        acc_orn_records = []
        pred_positions = []
        
        ref_pose = batch['ref_pose']
        B = ref_pose.shape[0]
        combined_prob_vol_pred = self.ddim_sampling(cond_input, num_steps=self.config.num_sampling_steps)
        combined_prob_vol_pred = combined_prob_vol_pred.squeeze(1)
        for i in range(B):
            # For each sample in the batch
            combined_prob_vol_pred_i = combined_prob_vol_pred[i].cpu()
            ref_pose_i = ref_pose[i].cpu()

            prob_dist_pred, _, pose_pred = finalize_localization_acc_only(combined_prob_vol_pred_i)

            # Compute accuracy and orientation error
            pose_pred = torch.tensor(pose_pred, dtype=torch.float32)
            pose_pred[:2] = pose_pred[:2] / 10  # Scale poses
            acc = torch.norm(pose_pred[:2] - ref_pose_i[:2], p=2).item()
            acc_orn = 0  # No orientation error calculation yet
            acc_records.append(acc)
            acc_orn_records.append(acc_orn)
            pred_positions.append(pose_pred[:2])  # Append only the positional part
        
        # Stack the list of tensors into a single tensor
        pred_positions = torch.stack(pred_positions)  # [B, 2]
        
        # Compute positional error
        acc_record = torch.norm(pred_positions - ref_pose[:, :2].cpu(), p=2, dim=1)  # [B]
        acc_mean = acc_record.mean().item()

        recalls = {
            "10m": (acc_record < 10).float().mean().item(),
            "2m": (acc_record < 2).float().mean().item(),
            "1m": (acc_record < 1).float().mean().item(),
            "0.5m": (acc_record < 0.5).float().mean().item(),
            "0.1m": (acc_record < 0.1).float().mean().item(),
        }
        
        for recall_name, recall_value in recalls.items():
            self.log(f"recall_{recall_name}", recall_value, on_step=False, on_epoch=True, prog_bar=True)
        
        self.log('positional_error', acc_mean, on_step=False, on_epoch=True, prog_bar=True)


    def validation_step(self, batch, batch_idx):
        if self.config.acc_only:
            # Load inputs
            prob_vol_depth = batch['prob_vol_depth'].to(self.device)
            prob_vol_semantic = batch['prob_vol_semantic'].to(self.device)
            prob_vol_gt = batch['prob_vol_gt'].to(self.device)
            ref_pose = batch['ref_pose'].to(self.device)

            B, H, W = prob_vol_gt.shape

            cond_input = torch.stack([prob_vol_depth, prob_vol_semantic], dim=1)
            prob_vol_gt = prob_vol_gt.unsqueeze(1)

            # Sample random timesteps
            t = torch.randint(0, self.num_timesteps, (B,), device=self.device).long()

            # Forward diffusion
            noise = torch.randn_like(prob_vol_gt)
            alpha_bar_t = self.scheduler.alpha_bar.to(self.device)[t].view(-1, 1, 1, 1)
            noisy_prob_vol = torch.sqrt(alpha_bar_t) * prob_vol_gt + torch.sqrt(1 - alpha_bar_t) * noise

            # Predict the noise
            noise_pred = self.model(noisy_prob_vol, t, cond_input)

            # Reconstruct the predicted map
            pred_prob_vol = (noisy_prob_vol - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)

            # Ensure the predicted map is non-negative and normalized
            pred_prob_vol = F.relu(pred_prob_vol)
            pred_prob_vol = pred_prob_vol / (pred_prob_vol.sum(dim=(2, 3), keepdim=True) + 1e-8)

            # Compute expected coordinates
            pred_x, pred_y = self.compute_expected_coordinates(pred_prob_vol)
            pred_x /= 10
            pred_y /= 10
            
            # Get ground truth coordinates
            gt_x = ref_pose[:, 0] 
            gt_y = ref_pose[:, 1] 

            # Compute positional loss
            positional_loss = torch.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2).mean()

            # Compute MSE loss
            mse_loss = F.mse_loss(noise_pred, noise)

            # Total loss
            total_loss = 1 *mse_loss + self.config.positional_loss_weight * positional_loss

            # Logging
            self.log('loss-valid', total_loss.detach().cpu(), on_step=False, on_epoch=True, prog_bar=True, batch_size=B)
            self.log('mse_loss_valid', mse_loss.detach().cpu(), on_step=False, on_epoch=True, batch_size=B)
            self.log('positional_loss_valid', positional_loss.detach().cpu(), on_step=False, on_epoch=True, batch_size=B)

            # if batch_idx < (200 // B):
            #     self.recall_validation(batch, cond_input)
            
            return total_loss

        else:
            # Handle the non-acc_only case (3D data)
            pass  # Implement as needed

    def ddim_sampling(self, cond_input, num_steps=50, number_of_evaluations=1):
        """
        Perform DDIM sampling to generate the combined probability map.

        Args:
            cond_input (torch.Tensor): Conditional input tensor [B, 2, H, W]
            num_steps (int): Number of sampling steps
            number_of_evaluations (int): Number of runs to average over

        Returns:
            torch.Tensor: Averaged generated combined probability map 
        """
        self.model.eval()
        results = []

        with torch.no_grad():
            B, _, H, W = cond_input.shape
            device = cond_input.device

            for _ in range(number_of_evaluations):
                # Initialize with random noise
                x = torch.randn(B, 1, H, W, device=device)

                # Prepare timesteps
                timesteps = torch.linspace(self.num_timesteps - 1, 0, num_steps, device=device).long()
                alphas_cumprod = self.scheduler.alpha_bar.to(device)

                # Prepare alphas for the current and next timesteps
                t = timesteps
                t_next = torch.cat([timesteps[1:], torch.tensor([0], device=device)])

                alphas_cumprod_t = alphas_cumprod[t]  # Shape: [num_steps]
                alphas_cumprod_t_next = alphas_cumprod[t_next]  # Shape: [num_steps]

                for i, t_i in enumerate(timesteps):
                    t_i = t_i.long().item()  # Get scalar timestep
                    t_tensor = torch.full((B,), t_i, device=device, dtype=torch.long)

                    # Predict noise
                    noise_pred = self.model(x, t_tensor, cond_input)  # x shape: [B, 1, H, W], noise_pred shape: [B, 1, H, W]

                    # Compute parameters with corrected alphas
                    alpha_t = alphas_cumprod_t[i].view(1, 1, 1, 1)  # Shape: [1,1,1,1]
                    alpha_t_next = alphas_cumprod_t_next[i].view(1, 1, 1, 1)  # Shape: [1,1,1,1]

                    # Compute x0_pred
                    sqrt_alpha_t = torch.sqrt(alpha_t)
                    sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
                    x0_pred = (x - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t

                    # Update x using the DDIM update rule
                    sqrt_alpha_t_next = torch.sqrt(alpha_t_next)
                    sqrt_one_minus_alpha_t_next = torch.sqrt(1 - alpha_t_next)
                    x = sqrt_alpha_t_next * x0_pred + sqrt_one_minus_alpha_t_next * noise_pred

                results.append(x)

        # Average the results over the specified number of runs
        return torch.stack(results).mean(dim=0) if number_of_evaluations > 1 else results[0]

    
    def ddpm_sampling(self, cond_input, num_steps=1000):
        """
        Perform DDPM sampling to generate the combined probability map.

        Args:
            cond_input (torch.Tensor): Conditional input tensor [B, 2, H, W]
            num_steps (int, optional): Number of sampling steps. If None, uses the number of timesteps defined in the scheduler.

        Returns:
            torch.Tensor: Generated combined probability map 
        """
        self.model.eval()
        with torch.no_grad():
            B, _, H, W = cond_input.shape
            device = cond_input.device
            x = torch.randn(B, 1, H, W, device=device)

            num_steps = num_steps or self.num_timesteps
            timesteps = torch.arange(self.num_timesteps - 1, -1, -1, device=device)
            alphas = self.scheduler.alpha.to(device)
            alphas_cumprod = self.scheduler.alpha_bar.to(device)
            betas = self.scheduler.beta.to(device)

            for t in timesteps:
                t = t.long()
                t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
                # Predict noise
                noise_pred = self.model(x, t_tensor, cond_input)

                # Compute coefficients
                alpha_t = alphas[t].view(-1, 1, 1, 1)
                beta_t = betas[t].view(-1, 1, 1, 1)
                alpha_cumprod_t = alphas_cumprod[t].view(-1, 1, 1, 1)
                sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod_t)
                sqrt_recip_alpha_t = torch.sqrt(1 / alpha_t)

                # Equation to compute the mean (denoised estimate)
                x0_pred = (x - sqrt_one_minus_alpha_cumprod_t * noise_pred) / torch.sqrt(alpha_cumprod_t)
                # Mean of the posterior
                mean = sqrt_recip_alpha_t * (x - beta_t / sqrt_one_minus_alpha_cumprod_t * noise_pred)

                if t > 0:
                    # Sample from the posterior
                    noise = torch.randn_like(x)
                    x = mean + torch.sqrt(beta_t) * noise
                else:
                    # For t == 0, no noise is added
                    x = mean
        return x