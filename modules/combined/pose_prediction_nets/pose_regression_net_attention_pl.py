# modules/combined/pose_regression_net_attention_pl.py

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from modules.combined.pose_prediction_nets.pose_regression_net_with_attention import PoseRegressionNetWithAttention

class PoseRegressionNetAttentionPL(LightningModule):
    """
    PyTorch Lightning Module for the pose regression network with attention.
    """
    def __init__(self, config, lr=1e-4, log_dir='logs', net_size = 'small', loss_type='mse'):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters for checkpointing

        self.lr = lr
        self.log_dir = log_dir  # Directory to save results
        self.config = config  # Store config for use
        self.loss_type = loss_type
        
        # Initialize the combined model
        if net_size == 'small':  # ~0.6K parameters
            self.model = PoseRegressionNetWithAttention()
        if net_size == 'medium':  # ~0.6K parameters
            pass
        if net_size == 'large':  # ~0.6K parameters
            pass        
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def forward(self, depth_prob_vol, semantic_prob_vol):
        return self.model(depth_prob_vol, semantic_prob_vol)
    
    def training_step(self, batch, batch_idx):
        gt_pose = batch['ref_pose']
        
        depth_prob_vol = batch['prob_vol_depth']  # [B, H, W]
        semantic_prob_vol = batch['prob_vol_semantic']  # [B, H, W]
        
        pose_pred = self(
            depth_prob_vol,
            semantic_prob_vol
        )  
        
        # Compute loss
        loss = self.compute_loss(pose_pred, gt_pose)
        self.log('loss_train', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=gt_pose.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        gt_pose = batch['ref_pose']
        
        depth_prob_vol = batch['prob_vol_depth']  # [B, H, W]
        semantic_prob_vol = batch['prob_vol_semantic']  # [B, H, W]
        
        pose_pred = self(
            depth_prob_vol,
            semantic_prob_vol
        )  
        
        # Compute loss
        loss = self.compute_loss(pose_pred, gt_pose)        
        self.log('loss-valid', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=gt_pose.size(0))

        # Get predicted pose
        # Find the peak of the combined_prob_map
        B = gt_pose.shape[0]

        pose_pred = pose_pred
        ref_pose = batch['ref_pose'][:, :2]
        # Calculate positional error
        acc_record = torch.norm(pose_pred - ref_pose, p=2, dim=1)  # [B]
        acc_mean = acc_record.mean().item()

        # Calculate recalls based on distance thresholds
        recalls = {
            "10m": (acc_record < 10).float().mean().item(),
            "2m": (acc_record < 2).float().mean().item(),
            "1m": (acc_record < 1).float().mean().item(),
            "0.5m": (acc_record < 0.5).float().mean().item(),
            "0.1m": (acc_record < 0.1).float().mean().item(),
        }

        # Log metrics
        self.log('loss-valid', loss.detach(), on_step=False, on_epoch=True, prog_bar=True, batch_size=B)
        self.log('positional_error', acc_mean, on_step=False, on_epoch=True, prog_bar=True)

        # Log recall metrics
        for recall_name, recall_value in recalls.items():
            self.log(f"recall_{recall_name}", recall_value, on_step=False, on_epoch=True, prog_bar=True)

        return {'loss': loss, 'positional_error': acc_mean, **recalls}



    def compute_loss(self, pose_pred, gt_pose):
        x_pred = pose_pred[:, 0]
        y_pred = pose_pred[:, 1]
       
        x_gt = gt_pose[:, 0]
        y_gt = gt_pose[:, 1]
        
        # Position loss (MSE)
        pos_loss = F.mse_loss(torch.stack([x_pred, y_pred], dim=1), torch.stack([x_gt, y_gt], dim=1))

        # Total loss
        total_loss = pos_loss 

        return total_loss
