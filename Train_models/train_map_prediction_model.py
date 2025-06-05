import argparse
import os
import yaml
import torch
import numpy as np
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from attrdict import AttrDict
import torch.nn.functional as F

from modules.combined.map_prediction_nets.map_predictor_pl import MapPredictorPL
from data_utils.bets_map_vector_data_utils import BestMapVectorDataset

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return AttrDict(yaml.safe_load(f))

def custom_collate_fn(batch, config):
    if config.pad_all_to_max:
                max_H = config.max_h // 10
                max_W = config.max_w // 10
                max_D = 36
    else:            
        # Determine max dimensions within the batch
        max_H = max(item['prob_vol_depth'].shape[0] for item in batch)
        max_W = max(item['prob_vol_depth'].shape[1] for item in batch)
        max_D = 36

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
        'best_pred_map_vector': torch.stack([item['best_pred_map_vector'] for item in batch]),        
        'best_gt_map_vector': torch.stack([item['best_gt_map_vector'] for item in batch]),        
        'ref_pose': torch.stack(ref_pose_tensors)
    }

    return batch

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Training script for combined probability volume network.")
    parser.add_argument(
        "--config",
        type=str,
        default="Train_models/configurations/full/map_prediction_net_config.yaml",
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
    model_log_dir = os.path.join(config.ckpt_path, f'map_predictor_net_epochs-{str(config.epochs)}_net_size-{config.NET_SIZE}_train_dataset_size-{config.DATA_SET_SIZE}-v2')
    prob_vol_path = config.prob_vol_path
    best_map_path = config.best_map_path

    # Create directory for model logs if it doesn't exist
    os.makedirs(model_log_dir, exist_ok=True)

    # Save the configuration used for training as a standard dictionary
    with open(os.path.join(model_log_dir, "saved_config.yaml"), "w") as f:
        yaml.safe_dump(dict(config), f)

    # Load dataset split
    split_file = os.path.join(dataset_dir, f"split_for_combined_{config.DATA_SET_SIZE}.yaml")
    with open(split_file, "r") as f:
        split = AttrDict(yaml.safe_load(f))
        

    # Set up dataset and DataLoader
    train_set = BestMapVectorDataset(
        dataset_dir,
        split.train,
        L=0,
        prob_vol_path=prob_vol_path,
        best_map_path=best_map_path,
        max_w=config.max_w,
        max_h=config.max_h
    )
    
    if config.batch_size > len(train_set):
        print(f"Warning: Batch size {config.batch_size} is larger than the dataset size {len(train_set)}")

    val_set = BestMapVectorDataset(
        dataset_dir,
        split.val,
        L=0,
        prob_vol_path=prob_vol_path,
        best_map_path=best_map_path,
        max_w=config.max_w,
        max_h=config.max_h
    )

    collate_fn = lambda batch: custom_collate_fn(batch, config)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Initialize model
    model = MapPredictorPL(
        config=config,
        lr=config.lr,
        log_dir=model_log_dir,
        net_size=config.NET_SIZE,
    )

    # Configure checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="loss_valid",
        dirpath=model_log_dir,
        filename="map_predictor_net-{epoch:02d}-{loss_valid:.5f}",
        save_top_k=2,
        mode="min",
    )

    # Compute the number of trainable parameters
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_trainable_params_million = num_trainable_params / 1e6

    # Create a custom logger with meaningful names, including the number of trainable params
    experiment_name = f"map_predictor_net_{num_trainable_params_million:.3f}M_train_dataset-{config.DATA_SET_SIZE}_batch_size-{config.batch_size}_net_size-{config.NET_SIZE}-v2"
    version = f"epochs_{config.epochs}_lr_{config.lr}"

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
        strategy="ddp" if torch.cuda.device_count() > 1 else None,
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
    trainer.save_checkpoint(os.path.join(model_log_dir, "final_map_predictor_model_checkpoint.ckpt"))

if __name__ == "__main__":
    main()
