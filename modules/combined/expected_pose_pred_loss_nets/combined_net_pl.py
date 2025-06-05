import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
import numpy as np

from modules.combined.expected_pose_pred_loss_nets.combined_net_small import CombinedProbVolNet_small
from modules.combined.expected_pose_pred_loss_nets.combined_net_small_acc_only import CombinedProbVolNet_small_acc_only
from modules.combined.expected_pose_pred_loss_nets.combined_net_medium import CombinedProbVolNet_medium
from modules.combined.expected_pose_pred_loss_nets.combined_net_large import CombinedProbVolNet_large
from modules.combined.expected_pose_pred_loss_nets.combined_net_medium_small import CombinedProbVolNet_medium_small
from modules.combined.expected_pose_pred_loss_nets.combined_net_6k import CombinedProbVolNet_6k
from modules.combined.expected_pose_pred_loss_nets.combined_net_unet import CombinedProbVolNet_UNet
from modules.combined.expected_pose_pred_loss_nets.combined_net_attention import CombinedProbVolNet_Attention

from utils.localization_utils import (
    finalize_localization,
    finalize_localization_acc_only
)
class CombinedProbVolNetPL(LightningModule):
    def __init__(self, config, lr=1e-4, log_dir='logs', net_size="small", acc_only = False):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters for checkpointing

        # Initialize the combined model
        if net_size == 'small':  # ~0.6K parameters
            self.model = CombinedProbVolNet_small()
        elif net_size == 'small_acc_only':  # ~0.6K parameters
            self.model = CombinedProbVolNet_small_acc_only()
        elif net_size == '6k':  # ~0.6K parameters
            self.model = CombinedProbVolNet_6k()
        elif net_size == 'medium':  # ~600K parameters
            self.model = CombinedProbVolNet_medium()
        elif net_size == 'large':  # ~2.5M parameters
            self.model = CombinedProbVolNet_large()
        elif net_size == 'medium-small':  # ~60K parameters
            self.model = CombinedProbVolNet_medium_small()   
        elif net_size == 'UNet':  # ~60K parameters
            self.model = CombinedProbVolNet_UNet()     
        elif net_size == 'Attention':  # ~60K parameters
            self.model = CombinedProbVolNet_Attention()              
        else:
            raise ValueError(f"Unsupported net_size: {net_size}")

        self.lr = lr
        self.log_dir = log_dir  # Directory to save results
        self.config = config  # Store config for use
        self.acc_only = acc_only

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer        
    
    def forward(self, depth_prob_vol, semantic_prob_vol):
        return self.model(depth_prob_vol, semantic_prob_vol)

    def finalize_localization(self, combined_prob_vol: torch.Tensor):
        """
        Differentiable finalize localization using the combined probability volume.
        Input:
            prob_vol: combined probability volume [B, H, W, O]
        Output:
            position_probs: probability distribution over positions [B, H, W]
            expected_pose: [B, 3] predicted states [x, y, theta]
        """
        if self.acc_only:
            B, H, W = combined_prob_vol.shape  # Input shape is [B, H, W]

            # Compute softmax over the combined probability volume
            probs = F.softmax(combined_prob_vol.view(B, -1), dim=1).view(B, H, W)

            # Create grids for x and y coordinates
            device = combined_prob_vol.device
            xs = torch.arange(W, device=device).float() / 10.0  # Scale back to original coordinates
            ys = torch.arange(H, device=device).float() / 10.0

            xs = xs.view(1, 1, W).expand(B, H, -1)  # Shape: [B, H, W]
            ys = ys.view(1, H, 1).expand(B, -1, W)  # Shape: [B, H, W]

            # Compute expected positions
            expected_x = torch.sum(probs * xs, dim=(1, 2))  # Shape: [B]
            expected_y = torch.sum(probs * ys, dim=(1, 2))  # Shape: [B]

            expected_pose = torch.stack([expected_x, expected_y], dim=1)  # Shape: [B, 2]

            return expected_pose
        else:
            B, H, W, D = combined_prob_vol.shape  # Input shape is now [B, H, W, D]

            # Compute softmax over the combined probability volume
            probs = F.softmax(combined_prob_vol.reshape(B, -1), dim=1).reshape(B, H, W, D)

            # Create grids for x, y, theta
            device = combined_prob_vol.device
            xs = torch.arange(W, device=device).float() / 10.0  # Scale back to original coordinates
            ys = torch.arange(H, device=device).float() / 10.0
            thetas = torch.arange(D, device=device).float() * (2 * np.pi / D) 

            xs = xs.view(1, 1, W, 1).expand(B, H, -1, D)  # Shape: [B, H, W, D]
            ys = ys.view(1, H, 1, 1).expand(B, -1, W, D)  # Shape: [B, H, W, D]
            thetas = thetas.view(1, 1, 1, D).expand(B, H, W, -1)  # Shape: [B, H, W, D]

            # Compute expected positions
            expected_x = torch.sum(probs * xs, dim=(1, 2, 3))  # Shape: [B]
            expected_y = torch.sum(probs * ys, dim=(1, 2, 3))  # Shape: [B]
            expected_theta = torch.sum(probs * thetas, dim=(1, 2, 3))  # Shape: [B]

            expected_pose = torch.stack([expected_x, expected_y, expected_theta], dim=1)  # Shape: [B, 3]

            return expected_pose


    def training_step(self, batch, batch_idx):    
        prob_vol_depth = batch['prob_vol_depth']
        prob_vol_semantic = batch['prob_vol_semantic']
        gt_pose = batch['ref_pose']
        
        combined_prob_vol = self(prob_vol_depth, prob_vol_semantic)
        B = combined_prob_vol.shape[0]

        expected_pose = self.finalize_localization(combined_prob_vol)
        
        loss = self.compute_loss(expected_pose, gt_pose)    

        self.log('loss_train', loss.detach().cpu(), on_step=True, on_epoch=True, prog_bar=True, batch_size=B)

        return loss


    def validation_step(self, batch, batch_idx):
        prob_vol_depth = batch['prob_vol_depth']
        prob_vol_semantic = batch['prob_vol_semantic']
        gt_pose = batch['ref_pose']
        
        combined_prob_vol = self(prob_vol_depth, prob_vol_semantic)
    
        expected_pose = self.finalize_localization(combined_prob_vol)
        
        loss = self.compute_loss(expected_pose, gt_pose)    

        B = combined_prob_vol.shape[0]
        pred_positions = []
        
        for i in range(B):
            prob_map_np = combined_prob_vol[i].detach().cpu()
            if self.acc_only:
                _, _,pred = finalize_localization_acc_only(prob_map_np)
            else:
                _, _, _,pred = finalize_localization(prob_map_np)
            pred_positions.append(pred)
        pred_positions = torch.tensor(pred_positions, device=combined_prob_vol.device).float()
        
        # Calculate positional error
        ref_pose = batch['ref_pose'][:, :2]  # [B, 2]
        acc_record = torch.norm(pred_positions[:, :2]/10 - ref_pose[:, :2], p=2, dim=1)  # [B]
        acc_mean = acc_record.mean().item()
        
        if not self.acc_only:
            # Orientation error calculation
            ref_pose = batch['ref_pose'][:, 2:]
            device = combined_prob_vol.device
            pi_tensor = torch.tensor(np.pi, device=device)  # Define pi as a tensor on the same device

            # Calculate orientation error with device compatibility
            acc_orn = (pred_positions[:, 2:] - ref_pose.to(device)) % (2 * pi_tensor)
            acc_orn = torch.min(acc_orn, 2 * pi_tensor - acc_orn) / pi_tensor * 180

            # Move acc_orn to CPU for NumPy compatibility and convert to numpy array
            acc_orn = acc_orn.cpu().numpy()
        else:
            acc_orn = None

        # Calculate recalls based on distance thresholds
        recalls = {
            "10m": (acc_record < 10).float().mean().item(),
            "2m": (acc_record < 2).float().mean().item(),
            "1m": (acc_record < 1).float().mean().item(),
            "0.5m": (acc_record < 0.5).float().mean().item(),
            "0.1m": (acc_record < 0.1).float().mean().item(),
            "1m 30 deg": np.sum(np.logical_and(acc_record.cpu().numpy() < 1, acc_orn < 30)) / acc_record.shape[0] if acc_orn is not None else 0,
        }

        # Log metrics
        self.log('loss-valid', loss.detach(), on_step=False, on_epoch=True, prog_bar=True, batch_size=B)
        self.log('positional_error', acc_mean, on_step=False, on_epoch=True, prog_bar=True)

        # Log recall metrics
        for recall_name, recall_value in recalls.items():
            self.log(f"recall_{recall_name}", recall_value, on_step=False, on_epoch=True, prog_bar=True)

        return {'loss': loss, 'positional_error': acc_mean, **recalls}

    


    def compute_loss(self, expected_poses, gt_poses, pos_weight=0.9, theta_weight=0.1):
        """
        Compute the loss between the expected poses and ground truth poses.

        Args:
            expected_poses (torch.Tensor): Expected poses tensor of shape [B, 3]
            gt_poses (torch.Tensor): Ground truth pose tensor of shape [B, 3]

        Returns:
            torch.Tensor: Computed loss
        """
        loss_pos = F.mse_loss(expected_poses[:, :2], gt_poses[:, :2])
        if self.acc_only:
        # Compute positional loss (Mean Squared Error)
            return loss_pos
        
        # Compute orientation loss using cosine similarity
        delta_theta = expected_poses[:, 2] - gt_poses[:, 2]
        loss_theta = torch.mean(1 - torch.cos(delta_theta))  # Angular difference

        # Total loss
        # loss = pos_weight * loss_pos + theta_weight * loss_theta
        loss = loss_pos 
        return loss
