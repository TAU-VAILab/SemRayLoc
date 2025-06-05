import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
import numpy as np

from modules.combined.map_prediction_nets.map_predictor_net import MapPredictorNet
from utils.localization_utils import finalize_localization

class MapPredictorPL(LightningModule):
    def __init__(self, config, lr=1e-4, log_dir='logs', net_size='small'):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = MapPredictorNet(net_size=net_size)
        self.log_dir = log_dir
        self.lr = lr
        self.weight_combinations = self.config.weight_combinations  # List of weight combinations

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def forward(self, depth_map, semantic_map):
        # Stack depth and semantic maps along channel dimension
        x = torch.stack([depth_map, semantic_map], dim=1)  # x is [B, 2, H, W, O]
        logits = self.model(x)  # Outputs are [B, 6]
        return logits

    def training_step(self, batch, batch_idx):
        depth_map = batch['prob_vol_depth'].to(self.device)  # [B, H, W, O]
        semantic_map = batch['prob_vol_semantic'].to(self.device)  # [B, H, W, O]
        gt_labels = batch['best_pred_map_vector'].to(self.device)  # [B] LongTensor with class indices (0-5)

        logits = self(depth_map, semantic_map)  # [B, 6]

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, gt_labels)

        self.log('loss_train', loss.detach(), on_step=True, on_epoch=True, prog_bar=True, batch_size=depth_map.size(0))

        return loss

    def validation_step(self, batch, batch_idx):
        depth_map = batch['prob_vol_depth'].to(self.device)  # [B, H, W, O]
        semantic_map = batch['prob_vol_semantic'].to(self.device)  # [B, H, W, O]
        gt_pose = batch['ref_pose'].to(self.device)  # [B, 3]

        logits = self(depth_map, semantic_map)  # [B, 6]
        pred_class_indices = logits.argmax(dim=1)  # [B]

        pred_positions = []
        for i in range(depth_map.size(0)):
            depth_weight, semantic_weight = self.weight_combinations[pred_class_indices[i].item()]
            combined_map = depth_weight * depth_map[i] + semantic_weight * semantic_map[i]  # [H, W, O]
            _, _, _, pred = finalize_localization(combined_map)
            pred_positions.append(pred)
        pred_positions = torch.tensor(pred_positions, device=self.device).float()  # [B, 3]

        # Compute positional error
        acc_record = torch.norm(pred_positions[:, :2] / 10 - gt_pose[:, :2], p=2, dim=1)
        acc_mean = acc_record.mean().item()

        # Orientation error
        orientation_error = ((pred_positions[:, 2] - gt_pose[:, 2] + np.pi) % (2 * np.pi)) - np.pi
        orientation_error_degrees = orientation_error.abs() * (180 / np.pi)
        orientation_error_mean = orientation_error_degrees.mean().item()

        loss = acc_mean

        recalls = {
            "10m": (acc_record < 10).float().mean().item(),
            "2m": (acc_record < 2).float().mean().item(),
            "1m": (acc_record < 1).float().mean().item(),
            "0.5m": (acc_record < 0.5).float().mean().item(),
            "0.1m": (acc_record < 0.1).float().mean().item(),
            "1m 30 deg": torch.mean(
                torch.logical_and(acc_record < 1, orientation_error_degrees < 30).float()
            ).item(),
        }

        # Log chosen map combinations
        map_combination_counts = torch.bincount(pred_class_indices, minlength=len(self.weight_combinations)).float()
        map_combination_counts_normalized = map_combination_counts / pred_class_indices.size(0)
        for i, count in enumerate(map_combination_counts_normalized):
            self.log(f"map_combination_{i}", count.item(), on_step=False, on_epoch=True, prog_bar=True)

        for recall_name, recall_value in recalls.items():
            self.log(f"recall_{recall_name}", recall_value, on_step=False, on_epoch=True, prog_bar=True)

        # Log validation loss and positional error
        self.log('loss_valid', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('positional_error', acc_mean, on_step=False, on_epoch=True, prog_bar=True)
        self.log('orientation_error', orientation_error_mean, on_step=False, on_epoch=True, prog_bar=True)

        return {'loss': loss, 'positional_error': acc_mean, 'orientation_error': orientation_error_mean, **recalls}

