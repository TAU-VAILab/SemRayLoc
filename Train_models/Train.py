import argparse
import os
import yaml

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import torch
from attrdict import AttrDict

from modules.depth.depth_net_pl import depth_net_pl
# from modules.mono.depth_net_pl_adaptive import depth_net_pl_adaptive

from modules.semantic.semantic_net_pl import semantic_net_pl
from modules.semantic.semantic_net_pl_maskformer import semantic_net_pl_maskformer
from modules.semantic.semantic_net_pl_maskformer_small import semantic_net_pl_maskformer_small
from modules.semantic.semantic_net_pl import semantic_net_pl
from modules.semantic.room_type_net_pl import room_type_net_pl
from modules.semantic.room_type_pred.room_type_pred_no_backbone_pl import room_type_pred_no_backbone_pl
from modules.semantic.room_type_pred.room_type_pred_resnet50_pl import room_type_pred_resnet50_pl
from data_utils.data_utils import LocalizationDataset
# from data_utils.data_utils_for_laser_train_s3d import GridSeqDataset # panorama

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return AttrDict(yaml.safe_load(f))

def setup_dataset_and_loader(config, dataset_dir, split):
    """Set up dataset and dataloader based on configuration."""
    train_set = LocalizationDataset(
        dataset_dir,
        split.train,
        L=config.depth_net.L,
        room_data_dir= config.room_data_dir,           
    )

    val_set = LocalizationDataset(
        dataset_dir,
        split.val,
        L=config.depth_net.L,
        augment= False,
        noise_std= 0,
        room_data_dir= config.room_data_dir,            
    )

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader

def initialize_model(config, model_type):
    """Initialize the model based on model type."""
    if model_type == "depth":
        model = depth_net_pl(
            shape_loss_weight=config.shape_loss_weight,
            lr=config.lr,
            d_min=config.depth_net.d_min,
            d_max=config.depth_net.d_max,
            d_hyp=config.depth_net.d_hyp,
            D=config.depth_net.D,
            F_W=config.depth_net.F_W,
        )
    elif model_type == "semantic":
        if config.use_room_type:
            model = semantic_net_pl(
                        num_ray_classes=config.num_classes,
                        num_room_types=config.num_room_types,
                        lr=config.lr,    
                        semantic_net_type= config.semantic_net_type
                    )
        else:
            model = semantic_net_pl(
                num_classes=config.num_classes,
                shape_loss_weight=config.shape_loss_weight,
                lr=config.lr,    
                F_W=config.depth_net.F_W,
            )
    return model

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Training script for depth or semantic prediction.")
    parser.add_argument(
        "--config",
        type=str,
        # default= "/home/yuvalg/projects/Semantic_Floor_plan_localization/Train_models/configurations/pano/depth_net_config.yaml", #S3D
        # default= "/home/yuvalg/projects/Semantic_Floor_plan_localization/Train_models/configurations/pano/semantic_net_config.yaml", #S3D
        default= "/home/yuvalg/projects/Semantic_Floor_plan_localization/Train_models/configurations/full/depth_net_config.yaml", #S3d_perspective
        # default= "/home/yuvalg/projects/Semantic_Floor_plan_localization/Train_models/configurations/full/semantic_net_config.yaml", #S3d_perspective
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
        devices=[0,1] if torch.cuda.is_available() else 0,
        log_every_n_steps=500,
    )

    # Start training
    trainer.fit(model, train_loader, val_loader)

    # Save the final model checkpoint
    trainer.save_checkpoint(os.path.join(model_log_dir, f"final_{config.model_type}_model_checkpoint.ckpt"))

if __name__ == "__main__":
    main()
