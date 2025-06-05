import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
import numpy as np

from modules.combined.expected_pose_pred_loss_nets.combined_net_small import CombinedProbVolNet_small
from modules.combined.expected_pose_pred_loss_nets.combined_net_medium import CombinedProbVolNet_medium
from modules.combined.expected_pose_pred_loss_nets.combined_net_large import CombinedProbVolNet_large
from modules.combined.expected_pose_pred_loss_nets.combined_net_medium_small import CombinedProbVolNet_medium_small

from utils.localization_utils import finalize_localization

class ProbVolNetPL(LightningModule):
    def __init__(self, config, lr=1e-4, log_dir='logs', net_size="small", loss_type='mse'):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters for checkpointing

        # Initialize the combined model
        if net_size == 'small':  # ~0.6K parameters
            self.model = CombinedProbVolNet_small()
        if net_size == '6k':  # ~0.6K parameters
            self.model = CombinedProbVolNet_6k()
        elif net_size == 'medium-small':  # ~60K parameters
            self.model = CombinedProbVolNet_medium_small()
        elif net_size == 'medium':  # ~600K parameters
            self.model = CombinedProbVolNet_medium()
        elif net_size == 'large':  # ~2.5M parameters
            self.model = CombinedProbVolNet_large()
        else:
            raise ValueError(f"Unsupported net_size: {net_size}")

        self.lr = lr
        self.log_dir = log_dir  # Directory to save results
        self.config = config  # Store config for use    
        self.loss_type = loss_type

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def forward(self, depth_prob_vol, semantic_prob_vol):
        return self.model(depth_prob_vol, semantic_prob_vol)
    
    def _create_gt_prob_vol(self, gt_pose, H, W, O, sigma=1.0):
        """
        Create a ground truth probability volume with a Gaussian peak at the ground truth location.

        Args:
            gt_pose (torch.Tensor): Ground truth pose tensor of shape (B, 3) with [x, y, theta].
            H, W, O: Spatial and orientation dimensions of the volume.
            sigma (float): Standard deviation for the Gaussian distribution.

        Returns:
            torch.Tensor: Ground truth probability volume of shape (B, H, W, O).
        """
        B = gt_pose.shape[0]
        gt_prob_vol = torch.zeros((B, H, W, O), device=gt_pose.device)

        # Extract ground truth x, y, theta
        x, y, theta = gt_pose[:, 0], gt_pose[:, 1], gt_pose[:, 2]
        theta = theta % (2 * np.pi)  # Normalize theta to [0, 2pi)

        # Convert theta to the closest orientation bin index
        theta_idx = (theta / (2 * np.pi) * O).long()

        # Gaussian kernel centered at ground truth
        x_coords = torch.arange(W, device=gt_pose.device).view(1, 1, W).repeat(B, H, 1).float()
        y_coords = torch.arange(H, device=gt_pose.device).view(1, H, 1).repeat(B, 1, W).float()

        # Compute Gaussian in spatial dimensions (H, W)
        spatial_dist = torch.exp(-(((x_coords - x.view(B, 1, 1))**2 + 
                                    (y_coords - y.view(B, 1, 1))**2) / (2 * sigma**2)))
        
        for b in range(B):
            # Set the probability distribution in orientation bins for each batch item
            orientation_dist = torch.zeros(O, device=gt_pose.device)
            orientation_dist[theta_idx[b]] = 1.0  # Peak at ground truth orientation
            
            # Apply Gaussian smoothing in the orientation axis
            orientation_dist_np = orientation_dist.cpu().numpy()  # Convert to numpy for scipy compatibility
            orientation_dist_np = gaussian_filter1d(orientation_dist_np, sigma=sigma, mode='wrap')  # Smooth
            orientation_dist = torch.tensor(orientation_dist_np, device=gt_pose.device)  # Convert back to torch tensor

            gt_prob_vol[b] = spatial_dist[b].unsqueeze(-1) * orientation_dist  # Multiply spatial and orientation dist

        # Normalize to make it a probability distribution
        # gt_prob_vol = gt_prob_vol / gt_prob_vol.sum(dim=(1, 2, 3), keepdim=True)
        return gt_prob_vol

    def calc_loss(self, pred_prob_vol, gt_prob_vol):
        if self.loss_type == 'mse':
            loss = F.mse_loss(pred_prob_vol, gt_prob_vol)
        elif self.loss_type == 'cross-entropy':
            # Flatten the volumes
            B = pred_prob_vol.shape[0]
            pred_probs = pred_prob_vol.reshape(B, -1) + 1e-8  # Add epsilon to avoid log(0)
            gt_probs = gt_prob_vol.reshape(B, -1)
            # Normalize predicted probabilities
            pred_probs = pred_probs / pred_probs.sum(dim=1, keepdim=True)
            # Cross-entropy loss
            loss = -(gt_probs * pred_probs.log()).sum(dim=1).mean()
        elif self.loss_type == 'weighted-mse':
            # Create weight map based on gt_prob_vol
            weight_map = gt_prob_vol / (gt_prob_vol.max() + 1e-8)
            loss = ((pred_prob_vol - gt_prob_vol) ** 2 * weight_map).mean()
        elif self.loss_type == 'focal':
            gamma = 2.0  # Focusing parameter
            alpha = 0.25  # Balancing parameter
            pt = torch.where(gt_prob_vol > 0, pred_prob_vol, 1 - pred_prob_vol)
            # Avoid log(0)
            pt = pt + 1e-8
            focal_loss = -alpha * ((1 - pt) ** gamma) * torch.log(pt)
            loss = focal_loss.mean()
        elif self.loss_type == 'nll':  # Negative Log-Likelihood Loss
            # Get the indices of the ground truth positions
            B, H, W, O = pred_prob_vol.shape
            device = pred_prob_vol.device
            # Assume gt_pose in grid coordinates
            gt_pose = self.current_gt_pose  # Stored during training_step
            x_indices = gt_pose[:, 0].long().clamp(0, W -1)
            y_indices = gt_pose[:, 1].long().clamp(0, H -1)
            theta = gt_pose[:, 2] % (2 * np.pi)
            theta_indices = ((theta / (2 * np.pi)) * O).long() % O
            # Gather predicted probabilities at GT locations
            pred_probs_at_gt = pred_prob_vol[torch.arange(B), y_indices, x_indices, theta_indices] + 1e-8
            loss = -torch.log(pred_probs_at_gt).mean()
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")
        return loss


    def training_step(self, batch, batch_idx):
        combined_prob_vol = self(batch['prob_vol_depth'], batch['prob_vol_semantic'])

        # Generate ground truth probability volume
        B, H, W, O = combined_prob_vol.shape
        gt_prob_vol = self._create_gt_prob_vol(batch['ref_pose'], H, W, O)

        # Store current gt_pose for use in loss calculation (if needed)
        self.current_gt_pose = batch['ref_pose']

        # Calculate loss
        loss = self.calc_loss(combined_prob_vol, gt_prob_vol)
        self.log("loss_train", loss.detach(), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        combined_prob_vol = self(batch['prob_vol_depth'], batch['prob_vol_semantic'])

        # Generate ground truth probability volume
        B, H, W, O = combined_prob_vol.shape
        gt_prob_vol = self._create_gt_prob_vol(batch['ref_pose'], H, W, O)

        # Store current gt_pose for use in loss calculation (if needed)
        self.current_gt_pose = batch['ref_pose']

        # Calculate loss
        loss = self.calc_loss(combined_prob_vol, gt_prob_vol)
        
        # Get predicted pose using finalize_localization
        _, _, _, pose_pred = finalize_localization(combined_prob_vol[0])  # Assuming batch size of 1

        # Convert pose_pred to a torch tensor and adjust scaling
        device = combined_prob_vol.device
        pose_pred_tensor = torch.tensor(pose_pred, device=device, dtype=torch.float32)
        pose_pred_tensor[:2] /= 10  # Scale positions

        # Calculate positional and orientation errors
        ref_pose = batch['ref_pose'][0]  # Assuming batch size of 1
        acc = torch.norm(pose_pred_tensor[:2] - ref_pose[:2], p=2).item()
        acc_orn = ((pose_pred_tensor[2] - ref_pose[2]) % (2 * np.pi)).item()
        acc_orn = min(acc_orn, 2 * np.pi - acc_orn) / np.pi * 180  # Convert to degrees

        # Log metrics
        self.log('loss-valid', loss.detach(), on_step=False, on_epoch=True, prog_bar=True, batch_size=B)
        self.log('positional_error', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('orientation_error', acc_orn, on_step=False, on_epoch=True, prog_bar=True)

        return {'loss': loss, 'positional_error': acc, 'orientation_error': acc_orn}
