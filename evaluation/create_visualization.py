import argparse
import os
import numpy as np
import torch
import tqdm
import yaml
from attrdict import AttrDict
import cv2
import torch.nn as nn
from PIL import Image

# Import your models
from modules.mono.depth_net_pl import depth_net_pl
from modules.semantic.semantic_net_pl import semantic_net_pl

# Import your dataset
from data_utils.data_utils import GridSeqDataset

# Import helper functions from utils
from utils.localization_utils import (
    get_ray_from_depth,
    get_ray_from_semantics,
    localize,
    finalize_localization,
)
from utils.result_utils import (
    save_acc_and_orn_records,
    calculate_recalls,
    save_recalls,
    create_combined_results_table,
)
from utils.data_loader_helper import load_scene_data
from utils.visualization_utils import plot_prob_dist


def evaluate_combined_model(
    depth_net,
    semantic_net,
    desdfs,
    semantics,
    test_set,
    gt_poses,
    maps,
    device,
    results_type_dir,
    config=None,
):
    weight_combinations = config.weight_combinations
    combined_recalls = {}
    
    acc_orn_records_for_all_weights = {}
    acc_records_for_all_weights = {}
    for use_ground_truth in [True, False]:
        for depth_weight, semantic_weight in weight_combinations:
            key = f"{depth_weight}_{semantic_weight}_{use_ground_truth}"
            acc_records_for_all_weights[key] = []
            acc_orn_records_for_all_weights[key] = []
            

    for data_idx in tqdm.tqdm(range(len(test_set))):
        data = test_set[data_idx]
        scene_idx = np.sum(data_idx >= np.array(test_set.scene_start_idx)) - 1
        scene = test_set.scene_names[scene_idx]
        if 'floor' in scene:
            pass
        else:
            scene_number = int(scene.split("_")[1])
            scene = f"scene_{scene_number}"
        idx_within_scene = data_idx - test_set.scene_start_idx[scene_idx]

        desdf = desdfs[scene]
        semantic = semantics[scene]
        ref_pose_map = gt_poses[scene][idx_within_scene * (config.L + 1) + config.L, :]

        for use_ground_truth in [True, False]:
            # Use ground truth data for depth if specified
            if not use_ground_truth:
                ref_img_torch = torch.tensor(data["ref_img"], device=device).unsqueeze(0)
                pred_depths, _, _ = depth_net.encoder(ref_img_torch, None)
                pred_depths = pred_depths.squeeze(0).detach().cpu().numpy()
                pred_rays_depth = get_ray_from_depth(pred_depths)
            else:
                pred_depths = data["ref_depth"]
                pred_rays_depth = get_ray_from_depth(pred_depths)

            # Use ground truth data for semantics if specified
            if not use_ground_truth:
                ref_img_torch = torch.tensor(data["ref_img"], device=device).unsqueeze(0)
                _, _, prob = semantic_net.encoder(ref_img_torch, None)
                prob_squeezed = prob.squeeze(dim=0)
                sampled_indices = torch.multinomial(
                    prob_squeezed, num_samples=1, replacement=True
                )
                sampled_indices = sampled_indices.squeeze(dim=1)
                sampled_indices_np = sampled_indices.cpu().numpy()
                pred_rays_semantic = get_ray_from_semantics(sampled_indices_np)
            else:
                sampled_indices_np = data["ref_semantics"]
                pred_rays_semantic = get_ray_from_semantics(sampled_indices_np)

            # Localization
            prob_vol_pred_depth, _, _, _ = localize(
                torch.tensor(desdf["desdf"]),
                torch.tensor(pred_rays_depth, device="cpu"),
                return_np=False,
            )
            
            prob_vol_pred_semantic, _, _, _ = localize(
                torch.tensor(semantic["desdf"]),
                torch.tensor(pred_rays_semantic, device="cpu"),
                return_np=False,
            )

            # Combine probabilities
            for depth_weight, semantic_weight in weight_combinations:
                weight_key = f"{depth_weight}_{semantic_weight}_{use_ground_truth}"
                min_shape = [min(d, s) for d, s in zip(prob_vol_pred_depth.shape, prob_vol_pred_semantic.shape)]
                prob_vol_pred_depth_sliced = prob_vol_pred_depth[tuple(slice(0, min_shape[i]) for i in range(len(min_shape)))]
                prob_vol_pred_semantic_sliced = prob_vol_pred_semantic[tuple(slice(0, min_shape[i]) for i in range(len(min_shape)))]

                combined_prob_vol_pred = depth_weight * prob_vol_pred_depth_sliced + semantic_weight * prob_vol_pred_semantic_sliced

                prob_vol_pred, prob_dist_pred, orientations_pred, pose_pred = finalize_localization(combined_prob_vol_pred)
                
                pose_pred[:2] = pose_pred[:2] / 10  # Scale poses
                acc = np.linalg.norm(pose_pred[:2] - ref_pose_map[:2], 2.0)
                acc_orn = (pose_pred[2] - ref_pose_map[2]) % (2 * np.pi)
                acc_orn = min(acc_orn, 2 * np.pi - acc_orn) / np.pi * 180

                acc_records_for_all_weights[weight_key].append(acc)
                acc_orn_records_for_all_weights[weight_key].append(acc_orn)
                
                # Define the file naming format
                file_identifier = f"{idx_within_scene}_{depth_weight}_{semantic_weight}_{use_ground_truth}"
                
                save_path = os.path.join(results_type_dir, 'plots', scene, str(idx_within_scene), "gt" if use_ground_truth else "net")
                # Define paths for the accuracy files
                acc_file_path = os.path.join(save_path, f"{file_identifier}_acc.txt")
                acc_orn_file_path = os.path.join(save_path, f"{file_identifier}_acc_orn.txt")
                
                # Ensure the directory exists
                os.makedirs(os.path.dirname(acc_file_path), exist_ok=True)
                os.makedirs(os.path.dirname(acc_orn_file_path), exist_ok=True)
                
                # Write the accuracy values to their respective files
                with open(acc_file_path, "w") as acc_file:
                    acc_file.write(f"{file_identifier}: {acc}\n")
                
                with open(acc_orn_file_path, "w") as acc_orn_file:
                    acc_orn_file.write(f"{file_identifier}: {acc_orn}\n")
    
                
                plot_prob_dist(
                prob_dist=prob_dist_pred, 
                resolution=0.1, 
                save_path= save_path, 
                file_name=f"{idx_within_scene}_{depth_weight}_{semantic_weight}_{use_ground_truth}", 
                occ=maps[scene], 
                pose_pred=pose_pred, 
                ref_pose_map=ref_pose_map,
                plot_type="combined",
                acc = acc,
                acc_orn = acc_orn                
                )

    # Process and save results
    for use_ground_truth in [True, False]:
        for depth_weight, semantic_weight in weight_combinations:
            weight_key = f"{depth_weight}_{semantic_weight}_{use_ground_truth}"
            acc_record = np.array(acc_records_for_all_weights[weight_key])
            acc_orn_record = np.array(acc_orn_records_for_all_weights[weight_key])

            weight_dir = os.path.join(
                results_type_dir, f"depth{depth_weight}_semantic{semantic_weight}_use_gt_{use_ground_truth}"
            )
            os.makedirs(weight_dir, exist_ok=True)
            save_acc_and_orn_records(acc_record, acc_orn_record, weight_dir)

            recalls = calculate_recalls(acc_record, acc_orn_record)
            combined_recalls[weight_key] = recalls
            save_recalls(recalls, weight_dir, weight_key)

    create_combined_results_table(combined_recalls, results_type_dir)


def evaluate_observation(prediction_type, config, test_set, device):
    # Extract configuration parameters
    dataset_dir = config.dataset_dir
    desdf_path = config.desdf_path
    results_dir = config.results_dir
    log_dir_depth = config.log_dir_depth
    log_dir_semantic = config.log_dir_semantic
    
    # Instantiate dataset and models
    D = config.D
    d_min = config.d_min
    d_max = config.d_max
    d_hyp = config.d_hyp

    depth_net = depth_net_pl.load_from_checkpoint(
        checkpoint_path=log_dir_depth,
        d_min=d_min,
        d_max=d_max,
        d_hyp=d_hyp,
        D=D,
    ).to(device)
    
    semantic_net = semantic_net_pl.load_from_checkpoint(
        checkpoint_path=log_dir_semantic,
        num_classes=config.num_classes,
    ).to(device)

    # Create results directory for this prediction type if it doesn't exist
    results_type_dir = os.path.join(results_dir, prediction_type)
    os.makedirs(results_type_dir, exist_ok=True)

    # Load desdf, semantics, maps, and ground truth poses
    desdfs, semantics, maps, gt_poses = load_scene_data(
        test_set, dataset_dir, desdf_path
    )

    evaluate_combined_model(
        depth_net,
        semantic_net,
        desdfs,
        semantics,
        test_set,
        gt_poses,
        maps,
        device,
        results_type_dir,
        config=config,
    )
    

def main():
    parser = argparse.ArgumentParser(description="Observation evaluation.")
    parser.add_argument(
        "--config_file",
        type=str,
        default="evaluation/configuration/config_visualizations.yaml",
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    # Load configuration from file
    with open(args.config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    config = AttrDict(config_dict)

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Extract configuration parameters
    dataset_dir = config.dataset_dir
    
    # Extract prediction_type and use_ground_truth from the configuration
    prediction_type = config.prediction_type

    scene_numbers = config.scene_numbers
    scene_names = [f'scene_{str(i).zfill(5)}' for i in scene_numbers]

    test_set = GridSeqDataset(
        dataset_dir,
        scene_names,
        L=0,
        start_scene=None,
        end_scene=None,
    )
    
    evaluate_observation(
    prediction_type, config, test_set, device
    )


if __name__ == "__main__":
    main()
