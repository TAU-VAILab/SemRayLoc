import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
import numpy as np

from modules.combined.gt_prob_vol_loss_nets.combined_net_small_acc_only import CombinedProbVolNet_small
from modules.combined.gt_prob_vol_loss_nets.combined_net_medium_acc_only import CombinedProbVolNet_medium
from modules.combined.gt_prob_vol_loss_nets.combined_net_large_acc_only import CombinedProbVolNet_large

class ProbVolNetPLAccOnly(LightningModule):
    def __init__(self, config, lr=1e-4, log_dir='logs', net_size="small", loss_type='mse'):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters for checkpointing

        # Initialize the combined model
        if net_size == 'small':  # ~0.6K parameters
            self.model = CombinedProbVolNet_small()
        if net_size == 'medium':  # ~0.6K parameters
            self.model = CombinedProbVolNet_medium()
        if net_size == 'large':  # ~0.6K parameters
            self.model = CombinedProbVolNet_large()
        # Add other net sizes if needed.

        self.lr = lr
        self.log_dir = log_dir  # Directory to save results
        self.config = config  # Store config for use    
        self.loss_type = loss_type

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def forward(self, depth_prob_vol, semantic_prob_vol):
        return self.model(depth_prob_vol, semantic_prob_vol)
    

    def _create_gt_prob_map(self, gt_pose, H, W, decay=0.6, threshold=4):
        """
        Create a ground truth map where the central ground truth cell is set to 1,
        and surrounding cells have decayed values based on Manhattan distance from the center.

        Args:
            gt_pose (torch.Tensor): Ground truth pose tensor of shape (B, 3) with [x, y, theta].
            H, W: Spatial dimensions of the map.
            decay (float): Decay factor for each unit of Manhattan distance from the ground truth cell.
            threshold (int): Maximum distance for decay to be applied.

        Returns:
            torch.Tensor: Ground truth map of shape (B, H, W) with decayed values around the GT location.
        """
        B = gt_pose.shape[0]
        
        # Create a grid of coordinates (H, W)
        y_coords = torch.arange(H, device=gt_pose.device).view(1, H, 1).expand(B, H, W)
        x_coords = torch.arange(W, device=gt_pose.device).view(1, 1, W).expand(B, H, W)
        
        # Extract the x and y positions of the ground truth points for each batch
        gt_x = (gt_pose[:, 0].view(B, 1, 1) * 10).int()
        gt_y = (gt_pose[:, 1].view(B, 1, 1) * 10).int()
        
        # Compute Manhattan distance from each grid point to the ground truth point
        manhattan_dist = torch.abs(x_coords - gt_x) + torch.abs(y_coords - gt_y)
        
        # Apply the decay based on distance, limited by the threshold
        gt_map = torch.clamp(1.0 - (manhattan_dist.float() * (1 - decay)), min=0)
        gt_map[manhattan_dist > threshold] = 0  # Zero out values beyond the threshold
        gt_map = gt_map
        return gt_map


    def calc_loss(self, pred_prob_map, gt_prob_map):
        if self.loss_type == 'masked-mse':
            epsilon = 1e-8 
            # Mask to focus only on non-zero GT locations
            mask = (gt_prob_map > 0).float()
            
            # Calculate masked MSE loss for each sample in the batch
            masked_mse_loss = ((pred_prob_map - gt_prob_map) ** 2 * mask).sum(dim=[1, 2]) / (mask.sum(dim=[1, 2]) + epsilon)
            
            # Penalize non-zero values in pred_prob_map outside the mask for each sample in the batch
            outside_mask = (1 - mask)
            outside_zero_loss = (pred_prob_map ** 2 * outside_mask).sum(dim=[1, 2]) / (outside_mask.sum(dim=[1, 2]) + epsilon)

            # Combine the two losses and average across the batch
            loss = (masked_mse_loss + outside_zero_loss).mean()        
            return loss
        elif self.loss_type == 'l1':
            epsilon = 1e-8
            # Mask to focus only on non-zero GT locations
            mask = (gt_prob_map > 0).float()
            
            # Calculate masked L1 loss for each sample in the batch
            masked_l1_loss = (torch.abs(pred_prob_map - gt_prob_map) * mask).sum(dim=[1, 2]) / (mask.sum(dim=[1, 2]) + epsilon)
            
            # Penalize non-zero values in pred_prob_map outside the mask for each sample in the batch
            outside_mask = (1 - mask)
            outside_zero_loss = (torch.abs(pred_prob_map) * outside_mask).sum(dim=[1, 2]) / (outside_mask.sum(dim=[1, 2]) + epsilon)
            
            # Combine the two losses and average across the batch
            loss = (masked_l1_loss + outside_zero_loss).mean()
            return loss
        elif self.loss_type == 'cross-entropy':
            # Flatten the maps
            B = pred_prob_map.shape[0]
            pred_probs = pred_prob_map.reshape(B, -1) + 1e-8  # Add epsilon to avoid log(0)
            gt_probs = gt_prob_map.reshape(B, -1)
            # Normalize predicted probabilities
            pred_probs = pred_probs / pred_probs.sum(dim=1, keepdim=True)
            # Cross-entropy loss
            loss = -(gt_probs * pred_probs.log()).sum(dim=1).mean()
        elif self.loss_type == 'weighted-mse':
            # Create weight map based on gt_prob_map
            weight_map = gt_prob_map / (gt_prob_map.max() + 1e-8)
            loss = ((pred_prob_map - gt_prob_map) ** 2 * weight_map).mean()
        elif self.loss_type == 'focal':
            gamma = 2.0  # Focusing parameter
            alpha = 0.25  # Balancing parameter
            pt = torch.where(gt_prob_map > 0, pred_prob_map, 1 - pred_prob_map)
            # Avoid log(0)
            pt = pt + 1e-8
            focal_loss = -alpha * ((1 - pt) ** gamma) * torch.log(pt)
            loss = focal_loss.mean()
        elif self.loss_type == 'nll':  # Negative Log-Likelihood Loss
            # Get the indices of the ground truth positions
            B, H, W = pred_prob_map.shape
            # Assume gt_pose in grid coordinates
            gt_pose = self.current_gt_pose * 10   # Stored during training_step
            x_indices = gt_pose[:, 0].long().clamp(0, W -1)
            y_indices = gt_pose[:, 1].long().clamp(0, H -1)
            # Gather predicted probabilities at GT locations
            pred_probs_at_gt = pred_prob_map[torch.arange(B), y_indices, x_indices] + 1e-8
            loss = -torch.log(pred_probs_at_gt).mean()
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")
        return loss

    def training_step(self, batch, batch_idx):
        depth_prob_vol = batch['prob_vol_depth']  # [B, H, W]
        semantic_prob_vol = batch['prob_vol_semantic']  # [B, H, W]

        combined_prob_map = self(depth_prob_vol, semantic_prob_vol)  # [B, H, W]

        # Generate ground truth probability map
        B, H, W = combined_prob_map.shape
        gt_prob_map = self._create_gt_prob_map(batch['ref_pose'], H, W)

        # Store current gt_pose for use in loss calculation (if needed)
        self.current_gt_pose = batch['ref_pose']

        # Calculate loss
        loss = self.calc_loss(combined_prob_map, gt_prob_map)
        self.log("loss_train", loss.detach(), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        depth_prob_vol = batch['prob_vol_depth']  # [B, H, W]
        semantic_prob_vol = batch['prob_vol_semantic']  # [B, H, W]

        combined_prob_map = self(depth_prob_vol, semantic_prob_vol)  # [B, H, W]

        # Generate ground truth probability map
        B, H, W = combined_prob_map.shape
        gt_prob_map = self._create_gt_prob_map(batch['ref_pose'], H, W)

        # Store current gt_pose for use in loss calculation (if needed)
        self.current_gt_pose = batch['ref_pose']

        # Calculate loss
        loss = self.calc_loss(combined_prob_map, gt_prob_map)
        
        # Get predicted pose
        # Find the peak of the combined_prob_map
        B = combined_prob_map.shape[0]
        pred_positions = []
        for i in range(B):
            prob_map_np = combined_prob_map[i].detach().cpu().numpy()
            y_idx, x_idx = np.unravel_index(prob_map_np.argmax(), prob_map_np.shape)
            pred_positions.append([x_idx, y_idx])
        pred_positions = torch.tensor(pred_positions, device=combined_prob_map.device).float() / 10

        # Calculate positional error
        ref_pose = batch['ref_pose'][:, :2]  # [B, 2]
        acc_record = torch.norm(pred_positions - ref_pose[:, :2], p=2, dim=1)  # [B]
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