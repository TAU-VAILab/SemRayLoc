# evaluate_diffusion_model.py

import argparse
import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
import cv2

from attrdict import AttrDict
import torch.nn.functional as F
from tqdm import tqdm

# Import the diffusion model
from modules.combined.whight_prediction_nets.weight_predictor_pl import WeightPredictorPL
from data_utils.prob_vol_data_utils import ProbVolDataset
# Import the localization utilities
from utils.localization_utils import finalize_localization, finalize_localization_acc_only
from utils.visualization_utils import plot_weight_pred_comparison

# Import the result utilities
from utils.result_utils import (
    save_acc_and_orn_records,
    calculate_recalls,
    save_recalls,
    create_combined_results_table
)    

def custom_collate_fn(batch, config):
    if config.pad_to_max:
        max_H = config.max_h //10 
        max_W = config.max_w //10
    else:        
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
    scene_names = [item['scene_name'] for item in batch]
    ref_idx = [item['ref_idx'] for item in batch]
    
    prob_vol_gt = torch.stack([
        item['prob_vol_depth_gt'] * 0.5 + item['prob_vol_semantic_gt'] * 0.5
        for item in batch
    ])

    # Stack tensors and return batch dictionary
    batch = {
        'prob_vol_depth': torch.stack([item['prob_vol_depth'] for item in batch]),
        'prob_vol_semantic': torch.stack([item['prob_vol_semantic'] for item in batch]),
        'prob_vol_depth_gt': torch.stack([item['prob_vol_depth_gt'] for item in batch]),
        'prob_vol_semantic_gt': torch.stack([item['prob_vol_semantic_gt'] for item in batch]),
        'prob_vol_gt': prob_vol_gt,
        'ref_pose': torch.stack(ref_pose_tensors),
        'scene_name': scene_names,
        'ref_idx': ref_idx
    }

    return batch

def evaluate_weight_prediction_model(
    diffusion_net,
    data_loader,
    device,
    results_dir,
    dataset_dir,
    config,
):
    # Initialize accuracy records for cond, pred, and gt
    acc_records_cond = []
    acc_records_pred = []
    acc_records_gt = []
    acc_records_gt_pred = []  # For GT-predicted map
    acc_orn_records = []  # Placeholder if needed

    diffusion_net.eval()
    diffusion_net.to(device)

    # Disable gradient calculations for inference
    with torch.no_grad():
        for batch in tqdm(data_loader):
            # Move data to device
            prob_vol_depth = batch['prob_vol_depth'].to(device)  # [B, H, W]
            prob_vol_semantic = batch['prob_vol_semantic'].to(device)  # [B, H, W]
            prob_vol_depth_gt = batch['prob_vol_depth_gt'].to(device)  # [B, H, W]
            prob_vol_semantic_gt = batch['prob_vol_semantic_gt'].to(device)  # [B, H, W]
            ref_pose = batch['ref_pose'].to(device)
            B = prob_vol_depth.shape[0]

            w_depth, w_semantic = diffusion_net(prob_vol_depth, prob_vol_semantic)  # Outputs are weights

            # Stack weights and apply Softmax over channel dimension
            w = torch.stack([w_depth, w_semantic], dim=1)  # [B, 2, H, W]
            w = F.softmax(w, dim=1)
            w_depth = w[:, 0, :, :]  # [B, H, W]
            w_semantic = w[:, 1, :, :]  # [B, H, W]

            # Conditional input map (cond): Equal weights
            cond_map = 0.5 * prob_vol_depth + 0.5 * prob_vol_semantic  # [B, H, W]

            # Prediction map (pred): Network-generated weights
            pred_map = w_depth * prob_vol_depth + w_semantic * prob_vol_semantic  # [B, H, W]

            # Ground truth map (gt): Equal weights from GT volumes
            gt_map = 0.5 * prob_vol_depth_gt + 0.5 * prob_vol_semantic_gt  # [B, H, W]

            # Ground truth-predicted map (gt-pred): Network weights on GT volumes
            gt_pred_map = w_depth * prob_vol_depth_gt + w_semantic * prob_vol_semantic_gt  # [B, H, W]

            for i in range(B):
                cond_map_i = cond_map[i].cpu()
                pred_map_i = pred_map[i].cpu()
                w_depth_i = w_depth[i].cpu()
                w_semantic_i = w_semantic[i].cpu()
                pred_map_i = pred_map[i].cpu()
                gt_map_i = gt_map[i].cpu()
                gt_pred_map_i = gt_pred_map[i].cpu()
                ref_pose_i = ref_pose[i].cpu()
                prob_vol_depth_i = prob_vol_depth[i].cpu()
                prob_vol_semantic_i = prob_vol_semantic[i].cpu()
                

                 

                # Calculate accuracy for conditional input (cond)
                _, _, pose_cond = finalize_localization_acc_only(cond_map_i)
                acc_cond = torch.norm(torch.tensor(pose_cond[:2], dtype=torch.float32) / 10 - ref_pose_i[:2], p=2).item()
                acc_records_cond.append(acc_cond)

                # Calculate accuracy for predicted output (pred)
                _, _, pose_pred = finalize_localization_acc_only(pred_map_i)
                acc_pred = torch.norm(torch.tensor(pose_pred[:2], dtype=torch.float32) / 10 - ref_pose_i[:2], p=2).item()
                acc_records_pred.append(acc_pred)

                # Calculate accuracy for ground truth (gt)
                _, _, pose_gt = finalize_localization_acc_only(gt_map_i)
                acc_gt = torch.norm(torch.tensor(pose_gt[:2], dtype=torch.float32) / 10 - ref_pose_i[:2], p=2).item()
                acc_records_gt.append(acc_gt)

                # Calculate accuracy for GT-predicted map (gt-pred)
                _, _, pose_gt_with_weight_pred = finalize_localization_acc_only(gt_pred_map_i)
                acc_gt_pred = torch.norm(torch.tensor(pose_gt_with_weight_pred[:2], dtype=torch.float32) / 10 - ref_pose_i[:2], p=2).item()
                acc_records_gt_pred.append(acc_gt_pred)

                # Placeholder for orientation accuracy
                acc_orn_records.append(0)

                # Visualization
                occ = None  # Or load occupancy map if needed
                if config.add_plots:
                    plot_weight_pred_comparison(
                        cond_map=cond_map_i.numpy(),
                        pred_map=pred_map_i.numpy(),
                        gt_map=gt_map_i.numpy(),
                        gt_pred_map=gt_pred_map_i.numpy(),
                        w_depth= w_depth_i.numpy(),
                        w_semantic =w_semantic_i.numpy(),
                        prob_vol_depth = prob_vol_depth_i.numpy(),
                        prob_vol_semantic = prob_vol_semantic_i.numpy(),
                        resolution=0.1,
                        save_path=os.path.join(results_dir, "visualizations"),
                        file_name=f"scene-{batch['scene_name'][i]}_ref_index-{batch['ref_idx'][i]}",
                        occ=occ,
                        pose_pred_cond=torch.tensor(pose_cond) / 10,
                        pose_pred=torch.tensor(pose_pred) / 10,
                        pose_pred_gt=torch.tensor(pose_gt) / 10,
                        pose_gt_with_weight_pred=torch.tensor(pose_gt_with_weight_pred) / 10,
                        ref_pose_map=ref_pose_i,
                        acc_cond=acc_cond,
                        acc_pred=acc_pred,
                        acc_gt=acc_gt,
                        model_type="Weight Prediction"
                    )

    # Save accuracy records and calculate recalls for cond, pred, gt, and gt-pred
    os.makedirs(results_dir, exist_ok=True)

        # For original_maps__weight-0.5
    acc_record_cond = np.array(acc_records_cond)
    save_acc_and_orn_records(acc_record_cond, None, os.path.join(results_dir, 'original_maps__weight-0.5'))
    recalls_cond = calculate_recalls(acc_record_cond, None)
    save_recalls(recalls_cond, os.path.join(results_dir, 'original_maps__weight-0.5'), 'original_maps__weight-0.5')

    # For original_maps__weight-Pred
    acc_record_pred = np.array(acc_records_pred)
    save_acc_and_orn_records(acc_record_pred, None, os.path.join(results_dir, 'original_maps__weight-Pred'))
    recalls_pred = calculate_recalls(acc_record_pred, None)
    save_recalls(recalls_pred, os.path.join(results_dir, 'original_maps__weight-Pred'), 'original_maps__weight-Pred')

    # For GT__weight-0.5
    acc_record_gt = np.array(acc_records_gt)
    save_acc_and_orn_records(acc_record_gt, None, os.path.join(results_dir, 'GT__weight-0.5'))
    recalls_gt = calculate_recalls(acc_record_gt, None)
    save_recalls(recalls_gt, os.path.join(results_dir, 'GT__weight-0.5'), 'GT__weight-0.5')

    # For GT__weight-Pred
    acc_record_gt_pred = np.array(acc_records_gt_pred)
    save_acc_and_orn_records(acc_record_gt_pred, None, os.path.join(results_dir, 'GT__weight-Pred'))
    recalls_gt_pred = calculate_recalls(acc_record_gt_pred, None)
    save_recalls(recalls_gt_pred, os.path.join(results_dir, 'GT__weight-Pred'), 'GT__weight-Pred')

    # Combine all recalls into a dictionary for LaTeX table
    combined_recalls = {
        'original_maps__weight-0.5': recalls_cond,
        'original_maps__weight-Pred': recalls_pred,
        'GT__weight-0.5': recalls_gt,
        'GT__weight-Pred': recalls_gt_pred
    }

    # Generate the combined LaTeX table
    create_combined_results_table(combined_recalls, results_dir)

def main():
    parser = argparse.ArgumentParser(description="weight_prediction_net evaluation.")
    parser.add_argument(
        "--config_file",
        type=str,
        default="evaluation/configuration/S3D/config_eval_weight_prediction.yaml",
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    # Load configuration from file
    with open(args.config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    config = AttrDict(config_dict)

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("======= USING DEVICE : ", device, " =======")

    # Paths
    dataset_dir = config.dataset_dir
    prob_vol_path = config.prob_vol_path
    results_dir = config.results_dir
    split_file = config.split_file
    acc_only = config.acc_only

    # Instantiate dataset
    L = config.L

    # Load the test set
    with open(split_file, "r") as f:
        split = AttrDict(yaml.safe_load(f))

    # Adjust the number of scenes as needed
    scene_names = split.test  # Use all test scenes
    # scene_names = split.test[:10]  # Uncomment to use a subset

    test_set = ProbVolDataset(
        dataset_dir=dataset_dir,
        scene_names=scene_names,
        L=L,
        prob_vol_path=prob_vol_path,
        acc_only=acc_only,
        max_w=config.max_w,
        max_h=config.max_h
    )

    collate_fn = lambda batch: custom_collate_fn(batch, config = config)

    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,  # Adjust batch size as needed
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
    )

    # Load the diffusion model
    diffusion_model_checkpoint = config.diffusion_model_checkpoint

    diffusion_net = WeightPredictorPL.load_from_checkpoint(
        checkpoint_path=diffusion_model_checkpoint,
        config=config,
    ).to(device) 

    results_dir = os.path.join(
        results_dir,
        f"weight_prediction_net__model_size-{config.NET_SIZE}_train_dataset_size-{config.DATA_SET_SIZE}_batch_size-{config.batch_size}_n_scenes-{len(scene_names)}"
    )

    output_path = os.path.join(results_dir, "saved_config.yaml")
    os.makedirs(results_dir, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.safe_dump(config_dict, f)
        
    
    evaluate_weight_prediction_model(
        diffusion_net=diffusion_net,
        data_loader=test_loader,
        device=device,
        results_dir=results_dir,
        dataset_dir=dataset_dir,
        config=config,
    )    

if __name__ == "__main__":
    main()
