import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
import numpy as np
from utils.localization_utils import finalize_localization_acc_only

from modules.combined.whight_prediction_nets.weight_predictor_net import WeightPredictorNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
import numpy as np
from utils.localization_utils import finalize_localization_acc_only

class WeightPredictorPL(LightningModule):
    def __init__(self, config, lr=1e-4, log_dir='logs', net_size='small', acc_only=True):
        super().__init__()
        self.save_hyperparameters()
        self.model = WeightPredictorNet(net_size=net_size)
        self.lr = lr
        self.log_dir = log_dir
        self.config = config
        self.acc_only = acc_only

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def forward(self, depth_map, semantic_map):
        output = self.model(depth_map, semantic_map)
        return output

    def compute_expected_location(self, combined_map):
        B, H, W = combined_map.shape
        # Flatten combined_map and normalize to sum to 1
        probs = combined_map.view(B, -1)
        probs = probs / probs.sum(dim=1, keepdim=True)
        # Create coordinate grids
        device = combined_map.device
        xs = torch.arange(W, device=device).float() / 10.0  # Convert indices to coordinates
        ys = torch.arange(H, device=device).float() / 10.0

        xs = xs.view(1, 1, W).expand(B, H, -1)  # [B, H, W]
        ys = ys.view(1, H, 1).expand(B, -1, W)  # [B, H, W]

        xs = xs.contiguous().view(B, -1)  # [B, H*W]
        ys = ys.contiguous().view(B, -1)

        expected_x = torch.sum(probs * xs, dim=1)
        expected_y = torch.sum(probs * ys, dim=1)

        expected_pose = torch.stack([expected_x, expected_y], dim=1)  # [B, 2]
        return expected_pose

    def training_step(self, batch, batch_idx):
        depth_map = batch['prob_vol_depth'].to(self.device)  # [B, H, W]
        semantic_map = batch['prob_vol_semantic'].to(self.device)  # [B, H, W]
        gt_pose = batch['ref_pose'].to(self.device)  # [B, 2]

        w_depth, w_semantic = self(depth_map, semantic_map)  # Outputs are weights

        # Stack weights and apply Softmax over channel dimension
        w = torch.stack([w_depth, w_semantic], dim=1)  # [B, 2, H, W]
        w = F.softmax(w, dim=1)
        w_depth = w[:, 0, :, :]  # [B, H, W]
        w_semantic = w[:, 1, :, :]  # [B, H, W]
        
        self.log('w_depth_mean', w_depth.mean().item(), on_step=True)
        self.log('w_depth_std', w_depth.std().item(), on_step=True)
        self.log('w_semantic_mean', w_semantic.mean().item(), on_step=True)
        self.log('w_semantic_std', w_semantic.std().item(), on_step=True)
        # Combine the maps using the weights
        combined_map = w_depth * depth_map + w_semantic * semantic_map  # [B, H, W]
        
        self.log('combined_map_mean', combined_map.mean().item(), on_step=True)
        self.log('combined_map_std', combined_map.std().item(), on_step=True)
        self.log('combined_map_max', combined_map.max().item(), on_step=True)
        self.log('combined_map_min', combined_map.min().item(), on_step=True)

        # Choose loss function based on config
        if self.config.loss_type == "mse":
            # Compute expected location
            expected_pose = self.compute_expected_location(combined_map)

            # Compute MSE loss between expected pose and ground truth pose
            loss = F.mse_loss(expected_pose, gt_pose[:, :2])

        elif self.config.loss_type == "nll":
            B, H, W = depth_map.shape
            # Normalize combined_map to sum to 1 over H x W
            combined_map = combined_map.view(B, -1)
            combined_map = combined_map / combined_map.sum(dim=1, keepdim=True)

            # Map ground truth positions to indices
            gt_x = (gt_pose[:, 0] * 10).long()  # Assuming resolution is 0.1m per pixel
            gt_y = (gt_pose[:, 1] * 10).long()

            # Clamp indices to be within valid range
            gt_x = gt_x.clamp(0, W - 1)
            gt_y = gt_y.clamp(0, H - 1)

            # Compute the index in the flattened combined_map
            idx = gt_y * W + gt_x  # [B]

            # Extract the probability at the ground truth position
            probs_at_gt = combined_map[torch.arange(B), idx]  # [B]

            # Compute negative log-likelihood loss
            loss = -torch.log(probs_at_gt + 1e-8).mean()
        elif self.config.loss_type == "nll_region":
            B, H, W = depth_map.shape
            # Normalize combined_map to sum to 1 over H x W
            combined_map = combined_map.view(B, -1)
            combined_map = combined_map / combined_map.sum(dim=1, keepdim=True)
            combined_map = combined_map.view(B, H, W)

            # Map ground truth positions to indices
            gt_x = (gt_pose[:, 0] * 10).long()  # Assuming resolution is 0.1m per pixel
            gt_y = (gt_pose[:, 1] * 10).long()

            # Create a circular mask for cells within 1m radius (10 cells in each direction)
            radius = 10  # 1m / 0.1m per pixel
            grid_y, grid_x = torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device))
            
            # Ensure the grid dimensions match the batch
            grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
            grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]

            # Compute the mask for each batch
            region_masks = ((grid_x - gt_x.unsqueeze(1).unsqueeze(2)).pow(2) + 
                            (grid_y - gt_y.unsqueeze(1).unsqueeze(2)).pow(2)) <= radius**2

            # Compute probabilities in the region for each batch
            probs_in_region = torch.zeros(B, device=self.device)  # Initialize tensor for batch sums
            for b in range(B):
                probs_in_region[b] = combined_map[b][region_masks[b]].sum()

            # Compute negative log-likelihood loss for the region
            loss = -torch.log(probs_in_region + 1e-8).mean()
        elif self.config.loss_type == "kl_divergence":
            B, H, W = depth_map.shape

            # Normalize combined_map to sum to 1 over H x W
            combined_map = combined_map.view(B, -1)
            combined_map = combined_map / combined_map.sum(dim=1, keepdim=True)
            combined_map = combined_map.view(B, H, W)

            # Map ground truth positions to indices
            gt_x = (gt_pose[:, 0] * 10).long()  # Assuming resolution is 0.1m per pixel
            gt_y = (gt_pose[:, 1] * 10).long()

            # Clamp indices to ensure they're within bounds
            gt_x = gt_x.clamp(0, W - 1)
            gt_y = gt_y.clamp(0, H - 1)

            # Create a Gaussian target distribution
            sigma = 10  # Standard deviation in pixels
            grid_y, grid_x = torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device), indexing='ij')
            grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
            grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]

            target_dist = torch.exp(-((grid_x - gt_x.unsqueeze(1).unsqueeze(2)).pow(2) +
                                    (grid_y - gt_y.unsqueeze(1).unsqueeze(2)).pow(2)) / (2 * sigma**2))

            # Normalize the target distribution
            target_dist = target_dist / target_dist.sum(dim=(1, 2), keepdim=True)

            # Compute KL divergence
            kl_div = F.kl_div(combined_map.log(), target_dist, reduction='batchmean')
            loss = kl_div


        self.log('loss_train', loss.detach(), on_step=True, on_epoch=True, prog_bar=True, batch_size=depth_map.size(0))

        mean_w_depth = w_depth.mean().item()
        mean_w_semantic = w_semantic.mean().item()
        self.log('mean_w_depth_train', mean_w_depth, on_step=True, on_epoch=True, prog_bar=False)
        self.log('mean_w_semantic_train', mean_w_semantic, on_step=True, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        depth_map = batch['prob_vol_depth'].to(self.device)  # [B, H, W]
        semantic_map = batch['prob_vol_semantic'].to(self.device)  # [B, H, W]
        gt_pose = batch['ref_pose'].to(self.device)  # [B, 2]

        w_depth, w_semantic = self(depth_map, semantic_map)

        # Stack weights and apply Softmax over channel dimension
        w = torch.stack([w_depth, w_semantic], dim=1)
        w = F.softmax(w, dim=1)
        w_depth = w[:, 0, :, :]
        w_semantic = w[:, 1, :, :]

        # Combine the maps
        combined_map = w_depth * depth_map + w_semantic * semantic_map

        pred_positions = []
        for i in range(depth_map.size(0)):
            prob_map_np = combined_map[i].detach().cpu()
            if self.acc_only:
                _, _, pred = finalize_localization_acc_only(prob_map_np)
            else:
                continue
            pred_positions.append(pred)
        pred_positions = torch.tensor(pred_positions, device=self.device).float()

        acc_record = torch.norm(pred_positions[:, :2] / 10 - gt_pose[:, :2], p=2, dim=1)
        acc_mean = acc_record.mean().item()
        loss = acc_mean


        recalls = {
            "10m": (acc_record < 10).float().mean().item(),
            "2m": (acc_record < 2).float().mean().item(),
            "1m": (acc_record < 1).float().mean().item(),
            "0.5m": (acc_record < 0.5).float().mean().item(),
            "0.1m": (acc_record < 0.1).float().mean().item(),
        }

        for recall_name, recall_value in recalls.items():
            self.log(f"recall_{recall_name}", recall_value, on_step=False, on_epoch=True, prog_bar=True)
            
        mean_w_depth = w_depth.mean().item()
        mean_w_semantic = w_semantic.mean().item()
        self.log('mean_w_depth_valid', mean_w_depth, on_step=False, on_epoch=True, prog_bar=False)
        self.log('mean_w_semantic_valid', mean_w_semantic, on_step=False, on_epoch=True, prog_bar=False)

        # Log combined map statistics
        self.log('combined_map_mean_valid', combined_map.mean().item(), on_step=False, on_epoch=True)
        self.log('combined_map_std_valid', combined_map.std().item(), on_step=False, on_epoch=True)
        
          # Log validation loss and positional error
        self.log('loss-valid', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('positional_error', acc_mean, on_step=False, on_epoch=True, prog_bar=True)

        return {'loss': loss, 'positional_error': acc_mean, **recalls}
