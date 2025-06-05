import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
import numpy as np

from modules.combined.whight_prediction_nets.full_output.full_weight_predictor_net import FullWeightPredictorNet
from utils.localization_utils import finalize_localization

class FullWeightPredictorPL(LightningModule):
    def __init__(self, config, lr=1e-4, log_dir='logs', net_size='small', acc_only=True):
        super().__init__()
        self.save_hyperparameters()
        self.model = FullWeightPredictorNet(net_size=net_size)
        self.lr = lr
        self.log_dir = log_dir
        self.config = config
        self.acc_only = acc_only

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def forward(self, depth_map, semantic_map):
        # Stack depth and semantic maps along channel dimension
        x = torch.stack([depth_map, semantic_map], dim=1)  # x is [B, 2, H, W, O]
        w_depth, w_semantic = self.model(x)
        return w_depth, w_semantic 

    def compute_expected_location(self, combined_map):
        B, H, W, O = combined_map.shape
        # Flatten combined_map and normalize to sum to 1
        probs = combined_map.view(B, -1)
        probs = probs / probs.sum(dim=1, keepdim=True)
        # Create coordinate grids
        device = combined_map.device
        xs = torch.arange(W, device=device).float() / 10.0  # Convert indices to coordinates
        ys = torch.arange(H, device=device).float() / 10.0
        os = torch.arange(O, device=device).float() * 10.0  # Assuming orientation increments of 10 degrees

        xs = xs.view(1, 1, W, 1).expand(B, H, -1, O)  # [B, H, W, O]
        ys = ys.view(1, H, 1, 1).expand(B, -1, W, O)  # [B, H, W, O]
        os = os.view(1, 1, 1, O).expand(B, H, W, -1)  # [B, H, W, O]

        xs = xs.contiguous().view(B, -1)  # [B, H*W*O]
        ys = ys.contiguous().view(B, -1)
        os = os.contiguous().view(B, -1)
        probs = probs.contiguous().view(B, -1)

        expected_x = torch.sum(probs * xs, dim=1)
        expected_y = torch.sum(probs * ys, dim=1)
        expected_o = torch.sum(probs * os, dim=1)

        expected_pose = torch.stack([expected_x, expected_y, expected_o], dim=1)  # [B, 3]
        return expected_pose
    

    def compute_loss_as_classification(self, combined_map, gt_pose):
        """
        Reformulate the pose estimation as a classification problem.

        Args:
            combined_map (torch.Tensor): Combined probability map of shape [B, H, W, O].
            gt_pose (torch.Tensor): Ground truth pose of shape [B, 3] with (x, y, orientation).

        Returns:
            torch.Tensor: Computed cross-entropy loss.
        """
        B, H, W, O = combined_map.shape
        
        # #print statistics of the combined_map before normalization
        # #print(f"Combined_map stats (before softmax):")
        # #print(f"  Max: {combined_map.max().item()}, Min: {combined_map.min().item()}")
        # #print(f"  Mean: {combined_map.mean().item()}, Std: {combined_map.std().item()}")
        
        # Normalize the combined_map along the last dimension
        combined_map = F.log_softmax(combined_map.view(B, -1), dim=-1).view(B, H, W, O)
        
        # #print statistics of the combined_map after normalization
        # #print(f"Combined_map stats (after softmax):")
        # #print(f"  Max: {combined_map.max().item()}, Min: {combined_map.min().item()}")
        # #print(f"  Mean: {combined_map.mean().item()}, Std: {combined_map.std().item()}")
        
        # Flatten combined_map to [B, H*W*O]
        logits = combined_map.view(B, -1)  # Shape: [B, H*W*O]
        # #print(f"Logits shape after flattening: {logits.shape}")

        # Convert ground truth pose (x, y, orientation) into class indices
        gt_x, gt_y, gt_orientation = gt_pose[:, 0], gt_pose[:, 1], gt_pose[:, 2]
        # #print(f"Ground truth x: {gt_x}, y: {gt_y}, orientation: {gt_orientation}")

        # Ensure indices are integers (e.g., rounding or scaling as necessary)
        gt_x = (gt_x * 10).long().clamp(0, H - 1)  # Map to [0, H-1]
        gt_y = (gt_y * 10).long().clamp(0, W - 1)  # Map to [0, W-1]
        gt_orientation = (torch.round(torch.rad2deg(gt_orientation) // 10).long() % 36)  # Map to [0, O-1]
        # #print(f"Processed gt_x: {gt_x}, gt_y: {gt_y}, gt_orientation: {gt_orientation}")

        # Compute class indices: class = x * (W * O) + y * O + orientation
        gt_class_indices = (gt_x * (W * O) + gt_y * O + gt_orientation)  # Shape: [B]
        # #print(f"Ground truth class indices: {gt_class_indices}")

        # Compute the loss
        loss = F.nll_loss(logits, gt_class_indices)
        # #print(f"Computed loss: {loss.item()}")

        return loss


    def compute_loss_with_range(self, combined_map, gt_pose, x_tolerance=1.0, y_tolerance=1.0, o_tolerance_deg=30.0):
        """
        Compute the loss by accepting predictions within a specified range around the ground truth.

        Args:
            combined_map (torch.Tensor): Combined probability map of shape [B, H, W, O].
            gt_pose (torch.Tensor): Ground truth pose of shape [B, 3] with (x, y, orientation).
            x_tolerance (float): Tolerance in meters for the x-coordinate.
            y_tolerance (float): Tolerance in meters for the y-coordinate.
            o_tolerance_deg (float): Tolerance in degrees for the orientation.

        Returns:
            torch.Tensor: Computed loss.
        """
        B, H, W, O = combined_map.shape

        # #print combined_map statistics before log_softmax
        #print(f"Combined_map stats (before log_softmax):")
        #print(f"  Max: {combined_map.max().item()}, Min: {combined_map.min().item()}")
        #print(f"  Mean: {combined_map.mean().item()}, Std: {combined_map.std().item()}")

        # Flatten combined_map to [B, H*W*O]
        logits = combined_map.view(B, -1)  # Shape: [B, H*W*O]

        # Apply log_softmax to get log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # #print log_probs statistics after log_softmax
        #print(f"log_probs stats (after log_softmax):")
        #print(f"  Max: {log_probs.max().item()}, Min: {log_probs.min().item()}")
        #print(f"  Mean: {log_probs.mean().item()}, Std: {log_probs.std().item()}")

        # Grid and orientation resolution
        pos_resolution = 0.1  # Adjust according to your actual grid resolution
        ori_bin_size = 10     # Degrees per orientation bin

        # Convert ground truth pose (x, y, orientation) into indices
        gt_x = (gt_pose[:, 0] / pos_resolution).long().clamp(0, H - 1)
        gt_y = (gt_pose[:, 1] / pos_resolution).long().clamp(0, W - 1)
        gt_o = (torch.round(torch.rad2deg(gt_pose[:, 2]) / ori_bin_size).long() % O)

        #print(f"gt_x: {gt_x}")
        #print(f"gt_y: {gt_y}")
        #print(f"gt_o: {gt_o}")

        # Calculate the range of acceptable indices
        x_tolerance_cells = int(x_tolerance / pos_resolution)
        y_tolerance_cells = int(y_tolerance / pos_resolution)
        o_tolerance_bins = int(o_tolerance_deg / ori_bin_size)

        #print(f"x_tolerance_cells: {x_tolerance_cells}")
        #print(f"y_tolerance_cells: {y_tolerance_cells}")
        #print(f"o_tolerance_bins: {o_tolerance_bins}")

        # Compute ranges for all samples
        x_min = (gt_x - x_tolerance_cells).clamp(0, H - 1)
        x_max = (gt_x + x_tolerance_cells).clamp(0, H - 1)
        y_min = (gt_y - y_tolerance_cells).clamp(0, W - 1)
        y_max = (gt_y + y_tolerance_cells).clamp(0, W - 1)
        o_min = gt_o - o_tolerance_bins
        o_max = gt_o + o_tolerance_bins

        #print(f"x_min: {x_min}")
        #print(f"x_max: {x_max}")
        #print(f"y_min: {y_min}")
        #print(f"y_max: {y_max}")
        #print(f"o_min: {o_min}")
        #print(f"o_max: {o_max}")

        # Initialize a list to hold the number of valid indices per sample
        num_valid_indices = []

        # Initialize a mask for valid predictions
        valid_mask = torch.zeros_like(log_probs, dtype=torch.bool)  # Shape: [B, H*W*O]

        for b in range(B):
            x_indices = torch.arange(x_min[b], x_max[b] + 1, device=gt_x.device)
            y_indices = torch.arange(y_min[b], y_max[b] + 1, device=gt_y.device)
            o_indices = torch.arange(o_min[b], o_max[b] + 1, device=gt_o.device) % O  # Wrap around

            # Create grid of indices
            grid = torch.stack(torch.meshgrid(x_indices, y_indices, o_indices, indexing='ij'), dim=-1)
            grid = grid.reshape(-1, 3)

            # Compute flat indices
            flat_indices = grid[:, 0] * (W * O) + grid[:, 1] * O + grid[:, 2]
            valid_mask[b, flat_indices.long()] = True

            # Store the number of valid indices
            num_valid_indices.append(flat_indices.numel())

            # #print debug information per sample
            #print(f"Sample {b}:")
            #print(f"  x_indices range: {x_indices.min().item()} to {x_indices.max().item()} (Total: {x_indices.numel()})")
            #print(f"  y_indices range: {y_indices.min().item()} to {y_indices.max().item()} (Total: {y_indices.numel()})")
            #print(f"  o_indices range: {o_indices.min().item()} to {o_indices.max().item()} (Total: {o_indices.numel()})")
            #print(f"  Number of valid indices: {flat_indices.numel()}")

        # #print total number of valid indices across batch
        #print(f"Total valid indices per sample: {num_valid_indices}")

        # Compute the loss over valid indices
        loss = -log_probs[valid_mask].mean()

        #print(f"Computed loss: {loss.item()}")

        return loss


    def training_step(self, batch, batch_idx):
        depth_map = batch['prob_vol_depth'].to(self.device)  # [B, H, W, O]
        semantic_map = batch['prob_vol_semantic'].to(self.device)  # [B, H, W, O]
        gt_pose = batch['ref_pose'].to(self.device)  # [B, 3]

        w_depth, w_semantic = self(depth_map, semantic_map)  # Outputs are weights [B, H, W, O]

        # Combine the maps using the weights
        combined_map = w_depth * depth_map + w_semantic * semantic_map  # [B, H, W, O]
        
        # # Compute expected location
        # expected_pose = self.compute_expected_location(combined_map)

        # # Compute MSE loss between expected pose and ground truth pose
        # loss = F.mse_loss(expected_pose[:, :2], gt_pose[:, :2])
        # loss = self.compute_loss_as_classification(combined_map, gt_pose)
        loss = self.compute_loss_with_range(combined_map, gt_pose)

        self.log('loss_train', loss.detach(), on_step=True, on_epoch=True, prog_bar=True, batch_size=depth_map.size(0))

        mean_w_depth = w_depth.mean().item()
        mean_w_semantic = w_semantic.mean().item()
        self.log('mean_w_depth_train', mean_w_depth, on_step=True, on_epoch=True, prog_bar=False)
        self.log('mean_w_semantic_train', mean_w_semantic, on_step=True, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        depth_map = batch['prob_vol_depth'].to(self.device)  # [B, H, W, O]
        semantic_map = batch['prob_vol_semantic'].to(self.device)  # [B, H, W, O]
        gt_pose = batch['ref_pose'].to(self.device)  # [B, 3]

        w_depth, w_semantic = self(depth_map, semantic_map)

        # Combine the maps
        combined_map = w_depth * depth_map + w_semantic * semantic_map  # [B, H, W, O]

        pred_positions = []
        for i in range(depth_map.size(0)):
            _,_,_,pred = finalize_localization(combined_map[i])  # Should return [x, y, orientation]
            pred_positions.append(pred)
        pred_positions = torch.tensor(pred_positions, device=self.device).float()  # [B, 3]

        # Compute positional error
        acc_record = torch.norm(pred_positions[:, :2] / 10 - gt_pose[:, :2], p=2, dim=1)
        acc_mean = acc_record.mean().item()

        # Orientation error (optional)
        orientation_error = ((pred_positions[:, 2] - gt_pose[:, 2] + np.pi) % (2 * np.pi)) - np.pi
        orientation_error_degrees = orientation_error.abs() * (180 / np.pi)  # Convert to degrees
        orientation_error_mean = orientation_error_degrees.mean().item()

        loss = acc_mean   # Scale orientation error

        recalls = {
            "10m": (acc_record < 10).float().mean().item(),
            "2m": (acc_record < 2).float().mean().item(),
            "1m": (acc_record < 1).float().mean().item(),
            "0.5m": (acc_record < 0.5).float().mean().item(),
            "0.1m": (acc_record < 0.1).float().mean().item(),
            "0.1m": (acc_record < 0.1).float().mean().item(),
            "1m 30 deg": torch.mean(torch.logical_and(acc_record < 1, orientation_error_degrees < 30).float()).item() if orientation_error is not None else 0,
        }

        for recall_name, recall_value in recalls.items():
            self.log(f"recall_{recall_name}", recall_value, on_step=False, on_epoch=True, prog_bar=True)

        mean_w_depth = w_depth.mean().item()
        mean_w_semantic = w_semantic.mean().item()
        self.log('mean_w_depth_valid', mean_w_depth, on_step=False, on_epoch=True, prog_bar=False)
        self.log('mean_w_semantic_valid', mean_w_semantic, on_step=False, on_epoch=True, prog_bar=False)

        # Log validation loss and positional error
        self.log('loss-valid', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('positional_error', acc_mean, on_step=False, on_epoch=True, prog_bar=True)
        self.log('orientation_error', orientation_error_mean, on_step=False, on_epoch=True, prog_bar=True)

        return {'loss': loss, 'positional_error': acc_mean, 'orientation_error': orientation_error_mean, **recalls}