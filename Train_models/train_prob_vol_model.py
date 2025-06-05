# train_prob_vol_model.py

import argparse
import os
import yaml

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import torch

from modules.combined.prob_vol_net_pl import ProbVolNetPL
from modules.combined.prob_vol_net_pl_acc_only import ProbVolNetPLAccOnly
from modules.combined.combined_net_attention_pl import CombinedProbVolNetAttentionPL
from modules.combined.pose_regression_net_attention_pl import PoseRegressionNetAttentionPL
from data_utils.prob_vol_data_utils import ProbVolDataset
from attrdict import AttrDict
import torch.nn.functional as F
import numpy as np

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return AttrDict(yaml.safe_load(f))

def custom_collate_fn(batch, acc_only = False):
    if not acc_only:
        # Determine max dimensions within the batch
        max_H = max(item['prob_vol_depth'].shape[0] for item in batch)
        max_W = max(item['prob_vol_depth'].shape[1] for item in batch)
        max_D = max(item['prob_vol_depth'].shape[2] for item in batch)

        # Pad each tensor to the max dimensions and stack them
        for item in batch:
            item['prob_vol_depth'] = F.pad(
                item['prob_vol_depth'], 
                (0, max_D - item['prob_vol_depth'].shape[2],
                0, max_W - item['prob_vol_depth'].shape[1],
                0, max_H - item['prob_vol_depth'].shape[0])
            )
            item['prob_vol_semantic'] = F.pad(
                item['prob_vol_semantic'], 
                (0, max_D - item['prob_vol_semantic'].shape[2],
                0, max_W - item['prob_vol_semantic'].shape[1],
                0, max_H - item['prob_vol_semantic'].shape[0])
            )

        # Convert 'ref_pose' to tensors if they're numpy arrays and stack
        ref_pose_tensors = [torch.tensor(item['ref_pose']) if isinstance(item['ref_pose'], np.ndarray) else item['ref_pose'] for item in batch]

        # Stack tensors and return batch dictionary
        batch = {
            'prob_vol_depth': torch.stack([item['prob_vol_depth'] for item in batch]),
            'prob_vol_semantic': torch.stack([item['prob_vol_semantic'] for item in batch]),
            'ref_pose': torch.stack(ref_pose_tensors)
        }
    
    else:
        # Determine max H and W within the batch
        max_H = max(item['prob_vol_depth'].shape[0] for item in batch)
        max_W = max(item['prob_vol_depth'].shape[1] for item in batch)

        # Pad each tensor to the max H and W dimensions
        for item in batch:
            item['prob_vol_depth'] = F.pad(
                item['prob_vol_depth'], 
                (0, max_W - item['prob_vol_depth'].shape[1],  # Pad width
                 0, max_H - item['prob_vol_depth'].shape[0])  # Pad height
            )
            item['prob_vol_semantic'] = F.pad(
                item['prob_vol_semantic'], 
                (0, max_W - item['prob_vol_semantic'].shape[1],  # Pad width
                 0, max_H - item['prob_vol_semantic'].shape[0])  # Pad height
            )

        # Convert 'ref_pose' to tensors if they're numpy arrays and stack
        ref_pose_tensors = [torch.tensor(item['ref_pose']) if isinstance(item['ref_pose'], np.ndarray) else item['ref_pose'] for item in batch]

        # Stack tensors and return batch dictionary
        batch = {
            'prob_vol_depth': torch.stack([item['prob_vol_depth'] for item in batch]),
            'prob_vol_semantic': torch.stack([item['prob_vol_semantic'] for item in batch]),
            'ref_pose': torch.stack(ref_pose_tensors)
        }
    
    return batch



def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Training script for combined probability volume network.")
    parser.add_argument(
        "--config",
        type=str,
        default="Train_models/configurations/full/prob_vol_net_config.yaml",
        help="Path to the config file",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("======= USING DEVICE : ", device, " =======")

    # Paths
    dataset_dir = os.path.join(config.dataset_path, config.dataset)
    model_log_dir = os.path.join(config.ckpt_path, f'combined_prob_vols_net_type-{config.network_type}_dataset_size-{config.DATA_SET_SIZE}_epochs-{str(config.epochs)}_loss-{config.loss_type}_acc_only-{config.acc_only}')
    prob_vol_path = config.prob_vol_path

    # Create directory for model logs if it doesn't exist
    os.makedirs(model_log_dir, exist_ok=True)

    # Load dataset split
    split_file = os.path.join(dataset_dir, f"split_for_combined_{config.DATA_SET_SIZE}.yaml")
    with open(split_file, "r") as f:
        split = AttrDict(yaml.safe_load(f))
        
    # Set up dataset and DataLoader
    train_set = ProbVolDataset(
        dataset_dir,
        split.train,
        L=config.L,
        prob_vol_path=prob_vol_path,
        acc_only = config.acc_only
    )

    val_set = ProbVolDataset(
        dataset_dir,
        split.val,
        L=config.L,
        prob_vol_path=prob_vol_path,
        acc_only = config.acc_only

    )
    collate_fn = lambda batch: custom_collate_fn(batch, acc_only=config.acc_only)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Initialize combined model based on network type from config
    if config.network_type == "attention":
        model = CombinedProbVolNetAttentionPL(
            config=config,
            lr=config.lr,
            log_dir=model_log_dir
        )
    elif config.network_type == "pose_regression_attention":
        model = PoseRegressionNetAttentionPL(
            config=config,
            lr=config.lr,
            log_dir=model_log_dir,
            net_size= config.net_size            
        )
    else:
        # For other network types, use the existing CombinedProbVolNetPL
        if config.acc_only:
            model = ProbVolNetPLAccOnly(
                config=config,
                lr=config.lr,
                log_dir=model_log_dir,
                net_size=config.network_type,
                loss_type = config.loss_type
            )
        else:
            model = ProbVolNetPL(
                config=config,
                lr=config.lr,
                log_dir=model_log_dir,
                net_size=config.network_type,
                loss_type = config.loss_type
            )

    # Configure checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="loss-valid",
        dirpath=model_log_dir,
        filename="combined_net-{epoch:02d}-{loss-valid:.2f}",
        save_top_k=3,
        mode="min",
    )

    # Compute the number of trainable parameters
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_trainable_params_million = num_trainable_params / 1e6

    # Create a custom logger with meaningful names, including the number of trainable params
    experiment_name = f"model_{config.network_type}_params_{num_trainable_params_million:.3f}M_dataset_{config.DATA_SET_SIZE}_loss-{config.loss_type}_acc_only-{config.acc_only}_batch_size-{config.batch_size}"
    version = f"epochs_{config.epochs}_lr_{config.lr}"

    logger = TensorBoardLogger(
        save_dir="/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/lightning_logs/Prob_Vol_Nets", 
        name=experiment_name,      # Experiment name
        version=version            # Version of the experiment
    )

    # Initialize Trainer
    trainer = Trainer(
        max_epochs=config.epochs,
        callbacks=[checkpoint_callback],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=2 if torch.cuda.is_available() else 0,
        # devices=[0],
        strategy="ddp" if torch.cuda.device_count() > 1 else None,
        log_every_n_steps=500,
        logger=logger
        )

    # Start training
    trainer.fit(
        model,
        train_loader,
        val_loader,
        ckpt_path=None,  # Change this if you want to resume from a checkpoint        
    )

    # Save the final model checkpoint
    trainer.save_checkpoint(os.path.join(model_log_dir, "final_combined_model_checkpoint.ckpt"))

if __name__ == "__main__":
    main()
