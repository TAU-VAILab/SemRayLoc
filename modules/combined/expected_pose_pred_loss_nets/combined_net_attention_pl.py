# modules/combined/combined_net_attention_pl.py

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
import numpy as np

from modules.combined.expected_pose_pred_loss_nets.combined_net_with_attention import DualBranchCombinedNetWithAttention
from modules.mono.depth_net_pl import depth_net_pl
from modules.semantic.semantic_net_pl import semantic_net_pl
from utils.localization_utils import (
    localize, 
    get_ray_from_depth, 
    get_ray_from_semantics, 
    finalize_localization
)

class CombinedProbVolNetAttentionPL(LightningModule):
    """
    PyTorch Lightning Module for the combined probability volume network with attention.
    """
    def __init__(self, config, lr=1e-4, log_dir='logs'):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters for checkpointing

        self.lr = lr
        self.log_dir = log_dir  # Directory to save results
        self.config = config  # Store config for use

        # Initialize the combined model with attention
        self.model = DualBranchCombinedNetWithAttention()

        # Initialize depth_net and semantic_net
        self.depth_net = depth_net_pl.load_from_checkpoint(
            checkpoint_path=self.config.depth_net_checkpoint,
            d_min=self.config.d_min,
            d_max=self.config.d_max,
            d_hyp=self.config.d_hyp,
            D=self.config.D,
        )
        self.depth_net.eval()
        self.depth_net.freeze()

        self.semantic_net = semantic_net_pl.load_from_checkpoint(
            checkpoint_path=self.config.semantic_net_checkpoint,
            num_classes=self.config.num_classes,
        )
        self.semantic_net.eval()
        self.semantic_net.freeze()

    def forward(self, depth_prob_vol, semantic_prob_vol):
        return self.model(depth_prob_vol, semantic_prob_vol)

    def _process_batch(self, batch):
        """
        Processes the input batch to compute combined probability volume and ground truth pose.

        Args:
            batch (dict): A batch of data.

        Returns:
            combined_prob_vol (torch.Tensor): Combined probability volume tensor.
            gt_pose (torch.Tensor): Ground truth pose tensor.
        """
        gt_pose = torch.tensor(batch[0]['ref_pose'], device=self.device, dtype=torch.float32)  # [3]
        ref_imgs = batch[0]['ref_img']
        desdf = batch[0]['desdf']
        color = batch[0]['color']

        # Compute depth rays using depth_net
        with torch.no_grad():
            ref_img_torch = torch.tensor(ref_imgs, device=self.device).unsqueeze(0)
            pred_depths, _, _ = self.depth_net.encoder(ref_img_torch, None) 
            pred_depths = pred_depths.squeeze(0).cpu().numpy()
            pred_rays_depth = get_ray_from_depth(pred_depths)

        # Compute semantic rays using semantic_net
        with torch.no_grad():
            ref_img_torch = torch.tensor(ref_imgs, device=self.device).unsqueeze(0)
            _, _, prob = self.semantic_net.encoder(ref_img_torch, None)
            prob_squeezed = prob.squeeze(dim=0)
            sampled_indices = torch.multinomial(
                prob_squeezed, num_samples=1, replacement=True
            ).squeeze(dim=1)
            sampled_indices_np = sampled_indices.cpu().numpy()
            pred_rays_semantic = get_ray_from_semantics(sampled_indices_np)

        # Compute probability volumes
        prob_vol_pred_depth, _, _, _ = localize(
            torch.tensor(desdf["desdf"], device=self.device),
            torch.tensor(pred_rays_depth, device=self.device),
            return_np=False,
        )
        prob_vol_pred_semantic, _, _, _ = localize(
            torch.tensor(color["desdf"], device=self.device),
            torch.tensor(pred_rays_semantic, device=self.device),
            return_np=False,
        )

        # Ensure same shape
        min_shape = [min(d, s) for d, s in zip(prob_vol_pred_depth.shape, prob_vol_pred_semantic.shape)]
        slices = tuple(slice(0, min_shape[i]) for i in range(len(min_shape)))
        prob_vol_pred_depth = prob_vol_pred_depth[slices]
        prob_vol_pred_semantic = prob_vol_pred_semantic[slices]

        # Forward pass through combined network
        combined_prob_vol = self(
            prob_vol_pred_depth,
            prob_vol_pred_semantic
        )

        return combined_prob_vol, gt_pose

    def training_step(self, batch, batch_idx):
        combined_prob_vol, gt_pose = self._process_batch(batch)

        # Prepare combined_prob_vol for loss computation
        combined_prob_vol = combined_prob_vol.unsqueeze(0)  # [B=1, H, W, D]
        combined_prob_vol = combined_prob_vol.permute(0, 3, 1, 2)  # [B=1, D, H, W]

        # Compute loss
        loss = self.compute_loss(combined_prob_vol, gt_pose)
        self.log('loss_train', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        return loss

    def validation_step(self, batch, batch_idx):
        combined_prob_vol, gt_pose = self._process_batch(batch)

        # Prepare combined_prob_vol for loss computation
        combined_prob_vol_tensor = combined_prob_vol.unsqueeze(0).permute(0, 3, 1, 2)  # [B=1, D, H, W]

        # Perform localization to get predicted pose
        _, _, _, pose_pred = finalize_localization(combined_prob_vol.detach().cpu())

        pose_pred = torch.tensor(pose_pred, device=self.device, dtype=torch.float32)  

        # Compute accuracy metrics
        pose_pred[:2] = pose_pred[:2] / 10  # Scale back to original coordinates

        acc = torch.norm(pose_pred[:2] - gt_pose[:2], p=2).item()

        acc_orn = (pose_pred[2] - gt_pose[2]) % (2 * np.pi)
        acc_orn = min(acc_orn, 2 * np.pi - acc_orn) / np.pi * 180        

        # Log metrics
        self.log('acc_valid', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        self.log('acc_orn_valid', acc_orn, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        
        # Compute loss
        loss = self.compute_loss(combined_prob_vol_tensor, gt_pose)
        self.log('loss-valid', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)

        return {'loss': loss, 'acc': acc, 'acc_orn': acc_orn}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def get_ground_truth_indices(self, gt_pose, resolution, H, W, D):
        """
        Convert ground truth pose to indices in the discretized grid.

        Args:
            gt_pose (torch.Tensor): Ground truth pose tensor of shape [3] (x, y, theta)
            resolution (float): Spatial resolution (e.g., 0.1)
            H (int): Grid height
            W (int): Grid width
            D (int): Number of orientation bins

        Returns:
            int: Flattened index corresponding to the ground truth pose
        """
        x, y, theta = gt_pose.cpu().numpy()
        x_idx = int(x / resolution)
        y_idx = int(y / resolution)
        theta_idx = int(theta / (2 * np.pi) * D)
        x_idx = np.clip(x_idx, 0, W - 1)
        y_idx = np.clip(y_idx, 0, H - 1)
        theta_idx = np.clip(theta_idx, 0, D - 1)
        # Flattened index
        gt_index = theta_idx * H * W + y_idx * W + x_idx
        return int(gt_index)

    def compute_loss(self, combined_prob_vol, gt_pose):
        """
        Compute the cross-entropy loss between the combined probability volume and ground truth pose indices.

        Args:
            combined_prob_vol (torch.Tensor): Tensor of shape [B=1, D, H, W]
            gt_pose (torch.Tensor): Ground truth pose tensor of shape [3]

        Returns:
            torch.Tensor: Computed loss
        """
        B, D, H, W = combined_prob_vol.shape  # B=1

        # Reshape output to [B, C] where C = D * H * W
        combined_prob_vol_flat = combined_prob_vol.view(B, -1)  # [B=1, C=D*H*W]
        
        # Get ground truth index
        gt_index = self.get_ground_truth_indices(gt_pose, resolution=0.1, H=H, W=W, D=D)
        gt_index = torch.tensor([gt_index], dtype=torch.long, device=self.device)  # [B=1]

        # Compute cross-entropy loss
        loss = F.cross_entropy(combined_prob_vol_flat, gt_index)
        return loss
