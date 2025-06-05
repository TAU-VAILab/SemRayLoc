import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from attrdict import AttrDict
from datetime import datetime

from best_k_net_pl import best_k_net_pl
from best_k_net_pl_with_fp import best_k_net_pl_with_fp
from simple_k_net_pl_with_fp import simple_k_net_pl_with_fp
from top_k_dataset import TopKDataset  


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return AttrDict(yaml.safe_load(f))

def load_split(split_file):
    """Load the dataset split YAML file."""
    with open(split_file, "r") as f:
        split = AttrDict(yaml.safe_load(f))
    return split

def process_scene_names(scene_list):
    """
    Process scene names according to the dataset type.
    If the scene name contains 'floor', it is assumed to be Zind dataset and
    processed accordingly; otherwise for S3D, remove leading zeros.
    """
    processed = []
    for scene in scene_list:
        # Default is_zind flag false
        is_zind = False
        if 'floor' in scene:
            is_zind = True
            processed.append(scene)
        else:
            # Assuming scene format "scene_00000", remove leading zeros.
            try:
                scene_number = int(scene.split('_')[1])
                scene_name = f"scene_{scene_number}"
            except Exception as e:
                print(f"Error processing scene: {scene} with error: {e}")
                scene_name = scene
            processed.append(scene_name)
    return processed

def setup_dataset_and_loader(config, split):
    """
    Set up the TopKDataset and corresponding DataLoaders using the training and validation splits.
    """
    dataset_dir = os.path.join(config.dataset_path, config.dataset)
    
    # Process scene names based on the dataset type
    # train_scenes = process_scene_names(split.train[:20])
    # val_scenes = process_scene_names(split.val[:10])
    train_scenes = process_scene_names(split.train)
    val_scenes = process_scene_names(split.val)
    
    train_set = TopKDataset(scene_names=train_scenes,
                            image_base_dir=dataset_dir,
                            top_k_dir=config.top_k_results_dir,
                            poses_filename=config.poses_filename)
    val_set = TopKDataset(scene_names=val_scenes,
                          image_base_dir=dataset_dir,
                          top_k_dir=config.top_k_results_dir,
                          poses_filename=config.poses_filename)
    
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=config.get("num_workers", 8))
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=config.get("num_workers", 8))
    
    return train_loader, val_loader

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Training script for best_k_net.")
    parser.add_argument(
        "--config",
        type=str,
        default="modules/top_k/config_train_top_k.yaml",
        help="Path to the YAML config file.",
    )
    args = parser.parse_args()

    # Load configuration.
    config = load_config(args.config)
    
    # Load split file.
    split = load_split(config.split_file)
    
    # Set device info.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("======= USING DEVICE :", device, "=======")

    # Setup dataset directory.
    dataset_dir = os.path.join(config.dataset_path, config.dataset)

    # Create a unique log directory name using config details and a timestamp.
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    # For example, you might want to incorporate batch size and epochs; adjust as needed.
    log_dir_name = f"best_k_net_bs-{config.batch_size}_ep-{config.epochs}_fp-{config.use_fp}_weights-{config.class_weights}_time-{timestamp}"
    default_root_dir = os.path.join(config.ckpt_path, log_dir_name)
    os.makedirs(default_root_dir, exist_ok=True)
    print("Logging to:", default_root_dir)

    # Set up DataLoaders using split file.
    train_loader, val_loader = setup_dataset_and_loader(config, split)

    # Initialize the model.
    if config.use_fp:
        model = best_k_net_pl_with_fp(lr=config.lr)
    elif config.is_simple:
        model = simple_k_net_pl_with_fp(lr=config.lr, class_weights=config.class_weights)
    else:
        model = best_k_net_pl(lr=config.lr)
    
    # Configure checkpoint callback.
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=default_root_dir,
        filename="best_k_net-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )
    
    # Initialize Trainer with default_root_dir set to our custom log directory.
    trainer = Trainer(
        max_epochs=config.epochs,
        callbacks=[checkpoint_callback],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        log_every_n_steps=500,
        default_root_dir=default_root_dir,
    )
    
    # Start training.
    trainer.fit(model, train_loader, val_loader)
    
    # Save the final model checkpoint.
    trainer.save_checkpoint(os.path.join(default_root_dir, "final_best_k_net_checkpoint.ckpt"))

if __name__ == "__main__":
    main()
