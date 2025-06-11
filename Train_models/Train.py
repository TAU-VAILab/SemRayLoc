import argparse
import os
import yaml

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import torch
from attrdict import AttrDict
from modules.depth.depth_net_pl import depth_net_pl
from modules.semantic.semantic_net_pl import semantic_net_pl

from data_utils.data_utils import LocalizationDataset
from data_utils.data_utils_zind import LocalizationDataset as LocalizationDatasetZind

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return AttrDict(yaml.safe_load(f))

def setup_dataset_and_loader(config, dataset_dir, split):
    """Set up dataset and dataloader based on configuration."""

    #try get is_zind from config
    is_zind = config.get("is_zind", False)

    if is_zind:
        train_set = LocalizationDatasetZind(dataset_dir, split.train)
        val_set = LocalizationDatasetZind(dataset_dir, split.val)
    else:
        train_set = LocalizationDataset(dataset_dir, split.train)
        val_set = LocalizationDataset(dataset_dir, split.val)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader

def initialize_model(config, model_type):
    """Initialize the model based on model type."""
    if model_type == "depth":
        model = depth_net_pl(
            shape_loss_weight=config.shape_loss_weight,
            lr=config.lr,        
        )
    elif model_type == "semantic":
        model = semantic_net_pl(
            num_ray_classes=config.num_classes,
            num_room_types=config.num_room_types,
            lr=config.lr,    
        )
    return model

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Training script for depth or semantic prediction.")
    parser.add_argument(
        "--config",
        type=str,
        # default= r".\Train_models\configurations\S3D\depth_net_config.yaml", #S3D depth
        # default= r".\Train_models\configurations\S3D\semantic_net_config.yaml", #S3D semantic
        # default= r".\Train_models\configurations\zind\depth_net_config.yaml", #Zind depth
        default= r".\Train_models\configurations\zind\semantic_net_config.yaml", #Zind semantic
        help="Path to the config file",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("======= USING DEVICE : ", device, " =======")

    # Pathsn
    split_dir = os.path.join(config.dataset_dir, "processed")
    model_log_dir = config.ckpt_path

    # Create directory for model logs if it doesn't exist
    os.makedirs(model_log_dir, exist_ok=True)

    # Load dataset split
    split_file = os.path.join(split_dir, "split.yaml")
    with open(split_file, "r") as f:
        split = AttrDict(yaml.safe_load(f))

    # Set up dataset and DataLoader
    train_loader, val_loader = setup_dataset_and_loader(config, config.dataset_dir, split)

    # Initialize model
    model = initialize_model(config, config.model_type)

    # Configure checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="loss-valid",
        dirpath=model_log_dir,
        filename=f"{config.model_type}_net-{{epoch:02d}}-{{loss-valid:.2f}}",
        save_top_k=3,
        mode="min",
    )

    # Initialize Trainer
    trainer = Trainer(
        max_epochs=config.epochs,
        callbacks=[checkpoint_callback],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices= 0 if torch.cuda.is_available() else 1,
        log_every_n_steps=500,
    )

    # Start training
    trainer.fit(model, train_loader, val_loader)

    # Save the final model checkpoint
    trainer.save_checkpoint(os.path.join(model_log_dir, f"final_{config.model_type}_model_checkpoint.ckpt"))

if __name__ == "__main__":
    main()
