# train_diffusion_model.py
import argparse
import os
import yaml
import torch
import numpy as np

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from modules.combined.single_source_diffusion_nets.single_source_diffusion_net_pl import SingleSourceDiffusionNetPL
from attrdict import AttrDict
import torch.nn.functional as F

from data_utils.prob_vol_data_utils import ProbVolDataset, get_narrow_prob_vol_gt, get_gaussian_prob_vol_gt

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return AttrDict(yaml.safe_load(f))

def custom_collate_fn(batch, acc_only=False, narrow_gt_map = False, config = None):
    if config.pad_all_to_max:
        max_H = config.max_h//10
        max_W = config.max_w//10
    else:
        # Determine max dimensions for 2D data
        max_H = max(item['prob_vol_depth'].shape[0] for item in batch)
        max_W = max(item['prob_vol_depth'].shape[1] for item in batch)

    # Pad each 2D tensor to the max dimensions and stack them
    for item in batch:
        item['prob_vol_depth'] = F.pad(
            item['prob_vol_depth'], 
            (0, max_W - item['prob_vol_depth'].shape[1],
                0, max_H - item['prob_vol_depth'].shape[0])
        )
        item['prob_vol_semantic'] = F.pad(
            item['prob_vol_semantic'], 
            (0, max_W - item['prob_vol_semantic'].shape[1],
                0, max_H - item['prob_vol_semantic'].shape[0])
        )
        item['prob_vol_depth_gt'] = F.pad(
            item['prob_vol_depth_gt'], 
            (0, max_W - item['prob_vol_depth_gt'].shape[1],
                0, max_H - item['prob_vol_depth_gt'].shape[0])
        )
        item['prob_vol_semantic_gt'] = F.pad(
            item['prob_vol_semantic_gt'], 
            (0, max_W - item['prob_vol_semantic_gt'].shape[1],
                0, max_H - item['prob_vol_semantic_gt'].shape[0])
        )


    # Convert 'ref_pose' to tensors if they're numpy arrays and stack
    ref_pose_tensors = [torch.tensor(item['ref_pose']) if isinstance(item['ref_pose'], np.ndarray) else item['ref_pose'] for item in batch]
    if narrow_gt_map:
        prob_vol_gt = torch.stack([
            get_gaussian_prob_vol_gt(item['prob_vol_depth_gt'] * 0.5 + item['prob_vol_semantic_gt'] * 0.5)
            for item in batch
        ])
    else:
         prob_vol_gt = torch.stack([
            item['prob_vol_depth_gt'] * 0.5 + item['prob_vol_semantic_gt'] * 0.5
            for item in batch
        ])
        
    # Stack tensors and return batch dictionary for non-acc_only
    batch = {
        'prob_vol_depth': torch.stack([item['prob_vol_depth'] for item in batch]),
        'prob_vol_semantic': torch.stack([item['prob_vol_semantic'] for item in batch]),
        'prob_vol_gt': prob_vol_gt,
        'prob_vol_semantic_gt': torch.stack([item['prob_vol_semantic_gt'] for item in batch]),
        'prob_vol_depth_gt': torch.stack([item['prob_vol_depth_gt'] for item in batch]),
        'ref_pose': torch.stack(ref_pose_tensors)
    }
    
    return batch

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Training script for conditional diffusion probability volume network.")
    parser.add_argument(
        "--config",
        type=str,
        default="Train_models/configurations/full/single_source_diffusion_net_config.yaml",
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
    model_name = f'single_source_diffusion__dataset_size-{config.DATA_SET_SIZE}_epochs-{str(config.epochs)}_model_type-{config.NET_TYPE}'
    model_log_dir = os.path.join(config.ckpt_path, model_name)
    
    os.makedirs(model_log_dir, exist_ok=True)

    # Load the configuration from the input file and write it to the output path
    with open(args.config, "r") as f_2:
        config_data = yaml.safe_load(f_2)

    output_path = os.path.join(model_log_dir, "saved_config.yaml")
    with open(output_path, "w") as f_1:
        yaml.safe_dump(config_data, f_1)
        
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
        acc_only=config.acc_only,
        max_w= config.max_w,
        max_h=config.max_h
    )

    val_set = ProbVolDataset(
        dataset_dir,
        split.val,
        L=config.L,
        prob_vol_path=prob_vol_path,
        acc_only=config.acc_only,
        max_w= config.max_w,
        max_h=config.max_h
    )

    collate_fn = lambda batch: custom_collate_fn(batch, acc_only=config.acc_only, narrow_gt_map= config.narrow_gt_map, config= config)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    model = SingleSourceDiffusionNetPL(
            config=config,
            log_dir=model_log_dir,
        )

    # Configure checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="loss-valid",
        dirpath=model_log_dir,
        filename="single_source_diffusion_-{epoch:02d}-{loss-valid:.5f}",
        save_top_k=1,
        mode="min",
    )

    # Compute the number of trainable parameters
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_trainable_params_million = num_trainable_params / 1e6

    # Create a custom logger with meaningful names, including the number of trainable params
    experiment_name = f"{model_name}_num_params_{num_trainable_params_million:.3f}M"
    version = f"epochs-{config.epochs}_lr-{config.lr}"

    logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name=experiment_name,
        version=version
    )

    # Initialize Trainer
    trainer = Trainer(
        max_epochs=config.epochs,
        callbacks=[checkpoint_callback],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        strategy= "ddp" if torch.cuda.device_count() > 1 else None,
        log_every_n_steps=500,
        logger=logger,
    )

    # Start training
    trainer.fit(
        model,
        train_loader,
        val_loader,
        ckpt_path=None,
    )

    # Save the final model checkpoint
    trainer.save_checkpoint(os.path.join(model_log_dir, "final_single_source_diffusion_checkpoint.ckpt"))

if __name__ == "__main__":
    main()
