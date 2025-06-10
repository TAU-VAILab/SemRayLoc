import argparse
import os
import yaml

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import torch
from modules.depth.depth_net_pl_adaptive import depth_net_pl_adaptive
from modules.semantic.semantic_net_pl import semantic_net_pl
from attrdict import AttrDict
from data_utils.data_utils_for_laser_train import GridSeqDataset
# from data_utils.data_utils import GridSeqDataset

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return AttrDict(yaml.safe_load(f))

def setup_dataset_and_loader(config, dataset_dir, split):
    """Set up dataset and dataloader based on configuration."""
    train_set = GridSeqDataset(
        dataset_dir,
        split.train,
        L=config.depth_net.L,
        roll=config.augmentation.roll,
        pitch=config.augmentation.pitch,
        room_data_dir= config.room_data_dir,
        is_train=True,
        pano_dir= config.dataset_pano,
    )

    val_set = GridSeqDataset(
        dataset_dir,
        split.val,
        L=config.depth_net.L,
        roll=config.augmentation.roll,
        pitch=config.augmentation.pitch,
        augment= False,
        noise_std= 0,
        room_data_dir= config.room_data_dir,
        is_train=False,
        pano_dir= config.dataset_pano,
    )

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=6)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=6)

    return train_loader, val_loader

def initialize_model(config, model_type):
    """Initialize the model based on model type."""
    if model_type == "depth":
        model = depth_net_pl_adaptive(
            shape_loss_weight=config.shape_loss_weight,
            lr=config.lr,
            d_min=config.depth_net.d_min,
            d_max=config.depth_net.d_max,
            d_hyp=config.depth_net.d_hyp,
            D=config.depth_net.D,
            F_W=config.depth_net.F_W,
        )
    elif model_type == "semantic":
        model = semantic_net_pl(
                    num_ray_classes=config.num_classes,
                    num_room_types=config.num_room_types,
                    lr=config.lr,    
                    semantic_net_type= config.semantic_net_type
                )

    return model

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Training script for depth or semantic prediction.")
    parser.add_argument(
        "--config",
        type=str,
        default= "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/Train_models/configurations/laser_train/semantic_net_config.yaml", #ZIND
        # default= "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/Train_models/configurations/laser_train/depth_net_config.yaml", #ZIND
        # default= "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/Train_models/configurations/zind/depth_net_config.yaml", #ZIND
        # default= "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/Train_models/configurations/zind/semantic_net_config.yaml", #ZIND
        help="Path to the config file",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("======= USING DEVICE : ", device, " =======")

    # Pathsn
    dataset_dir = os.path.join(config.dataset_path, config.dataset)
    model_log_dir = os.path.join(config.ckpt_path, config.model_type)

    # Create directory for model logs if it doesn't exist
    os.makedirs(model_log_dir, exist_ok=True)

    # Load dataset split
    split_file = os.path.join(dataset_dir, "split.yaml")
    with open(split_file, "r") as f:
        split = AttrDict(yaml.safe_load(f))

    # Set up dataset and DataLoader
    train_loader, val_loader = setup_dataset_and_loader(config, dataset_dir, split)

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
        devices=[0] if torch.cuda.is_available() else 0,
        log_every_n_steps=500,
    )

    # Start training
    trainer.fit(model, train_loader, val_loader)

    # Save the final model checkpoint
    trainer.save_checkpoint(os.path.join(model_log_dir, f"final_{config.model_type}_model_checkpoint.ckpt"))

if __name__ == "__main__":
    main()
