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
from modules.semantic.semantic_net_pl_maskformer import semantic_net_pl_maskformer
from modules.semantic.semantic_net_pl_maskformer_small import semantic_net_pl_maskformer_small
from modules.semantic.semantic_net_pl_maskformer_small_room_type import semantic_net_pl_maskformer_small_room_type

# Import your dataset
from data_utils.data_utils import GridSeqDataset
import torch.nn.functional as F

# Import helper functions from utils
from utils.localization_utils import (
    get_ray_from_depth,
    get_ray_from_semantics,
    get_ray_from_semantics_v2,
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
from utils.visualization_utils import plot_prob_dist, plot_dict_relationship

from data_utils.prob_vol_data_utils import ProbVolDataset
from torch.utils.data import DataLoader

def pad_to_max(prob_vol,max_H, max_W):
    prob_vol= F.pad(prob_vol,(0, max_W - prob_vol.shape[1],0, max_H - prob_vol.shape[0]))
    return prob_vol

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
    valid_scene_names,
    config,  
):
    weight_combinations = config.weight_combinations
    combined_recalls = {}

    # Initialize records for weight combinations
    acc_records_for_all_weights = {
        f"{depth_weight}_{semantic_weight}": []
        for depth_weight, semantic_weight in weight_combinations
    }
    acc_orn_records_for_all_weights = {
        f"{depth_weight}_{semantic_weight}": []
        for depth_weight, semantic_weight in weight_combinations
    }
    all_weight_comb_dict = {f"{depth_weight}_{semantic_weight}": {} for depth_weight, semantic_weight in weight_combinations}
    
    for data_idx in tqdm.tqdm(range(len(test_set))):
        data = test_set[data_idx]
        scene_idx = np.sum(data_idx >= np.array(test_set.scene_start_idx)) - 1
        scene = test_set.scene_names[scene_idx]
        if 'floor' in scene:
            pass
        else:
            scene_number = int(scene.split("_")[1])
            scene = f"scene_{scene_number}"
        
        if not scene in valid_scene_names:
            continue
            
        idx_within_scene = data_idx - test_set.scene_start_idx[scene_idx]

        desdf = desdfs[scene]
        semantic = semantics[scene]
        ref_pose_map = gt_poses[scene][idx_within_scene * (config.L + 1) + config.L, :]
        if not config.use_saved_prob_vol:
            # Use ground truth data for depth if specified
            if not config.use_ground_truth_depth:
                ref_img_torch = torch.tensor(data["ref_img"], device=device).unsqueeze(0)
                ref_mask_torch = torch.tensor(data["ref_mask"], device=device).unsqueeze(0)
                with torch.no_grad():
                    pred_depths, _, _ = depth_net.encoder(ref_img_torch, ref_mask_torch)
                pred_depths = pred_depths.squeeze(0).detach().cpu().numpy()
                pred_rays_depth = get_ray_from_depth(pred_depths, V=config.V, F_W= config.F_W)
            else:
                pred_depths = data["ref_depth"]
                pred_rays_depth = get_ray_from_depth(pred_depths, V=config.V, F_W= config.F_W)

            # Use ground truth data for semantics if specified
            if not config.use_ground_truth_semantic:
                ref_img_torch = torch.tensor(data["ref_img"], device=device).unsqueeze(0)
                ref_mask_torch = torch.tensor(data["ref_mask"], device=device).unsqueeze(0)
                with torch.no_grad():
                    if config.use_maskformer:
                        ray_prob, _, _ = semantic_net(ref_img_torch, ref_mask_torch)
                        prob = F.softmax(ray_prob, dim=-1)
                    else:
                        _, _, prob = semantic_net.encoder(ref_img_torch, ref_mask_torch)
                prob_squeezed = prob.squeeze(dim=0)
                sampled_indices = torch.multinomial(
                    prob_squeezed, num_samples=1, replacement=True
                )
                sampled_indices = sampled_indices.squeeze(dim=1)
                sampled_indices_np = sampled_indices.cpu().numpy()
                # pred_rays_semantic = get_ray_from_semantics(sampled_indices_np, V=config.V, F_W= config.F_W)        
                pred_rays_semantic = get_ray_from_semantics_v2(sampled_indices_np)
                
            else:
                sampled_indices_np = data["ref_semantics"]
                pred_rays_semantic = get_ray_from_semantics_v2(sampled_indices_np)

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
                localize_type = "semantic"
            )
        else:
            if config.use_ground_truth_depth:
                prob_vol_pred_depth = data['prob_vol_depth_gt'].to(device)
                prob_vol_pred_semantic = data['prob_vol_semantic_gt'].to(device)
            else:
                prob_vol_pred_depth = data['prob_vol_depth'].to(device)
                prob_vol_pred_semantic = data['prob_vol_semantic'].to(device)

            
        # Combine probabilities using weights
        for depth_weight, semantic_weight in weight_combinations:
            weight_key = f"{depth_weight}_{semantic_weight}"
            min_shape = [min(d, s) for d, s in zip(prob_vol_pred_depth.shape, prob_vol_pred_semantic.shape)]
            prob_vol_pred_depth_sliced = prob_vol_pred_depth[tuple(slice(0, min_shape[i]) for i in range(len(min_shape)))]
            prob_vol_pred_semantic_sliced = prob_vol_pred_semantic[tuple(slice(0, min_shape[i]) for i in range(len(min_shape)))]
      
            combined_prob_vol_pred = depth_weight * prob_vol_pred_depth_sliced + semantic_weight * prob_vol_pred_semantic_sliced    
                
            prob_vol_pred, prob_dist_pred, orientations_pred, pose_pred = finalize_localization(combined_prob_vol_pred,data["room_polygons"] )
            
            # Convert pose_pred to torch.Tensor before calling .cpu()
            pose_pred = torch.tensor(pose_pred, device=device, dtype=torch.float32)
            pose_pred[:2] = pose_pred[:2] / 10  # Scale poses
            acc = torch.norm(pose_pred[:2] - torch.tensor(ref_pose_map[:2], device=device), p=2).item()
            acc_orn = ((pose_pred[2] - torch.tensor(ref_pose_map[2], device=device)) % (2 * np.pi)).item()
            acc_orn = min(acc_orn, 2 * np.pi - acc_orn) / np.pi * 180
                
            key = prob_dist_pred.max().item()  # Ensure scalar value
            if key not in all_weight_comb_dict[weight_key]:  # Check if the key exists
                all_weight_comb_dict[weight_key][key] = []  # Initialize with an empty list
            all_weight_comb_dict[weight_key][key].append(acc)  # Append the value
            
            acc_records_for_all_weights[weight_key].append(acc)
            acc_orn_records_for_all_weights[weight_key].append(acc_orn)
        
    # Directory to save the plots
    save_path = '/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/results/temp_figs'

    for weight_key, dict_max_val_to_acc in all_weight_comb_dict.items():
        plot_dict_relationship(
            dict_max_val_to_acc,
            save_path=save_path,
            weight_key=weight_key,
            title='Relationship Between Max Probability and Distance'
        )

    # Process and save results for weight combinations
    for depth_weight, semantic_weight in weight_combinations:
        weight_key = f"{depth_weight}_{semantic_weight}"
        acc_record = np.array(acc_records_for_all_weights[weight_key])
        acc_orn_record = np.array(acc_orn_records_for_all_weights[weight_key])

        weight_dir = os.path.join(
            results_type_dir, f"depth{depth_weight}_semantic{semantic_weight}"
        )
        os.makedirs(weight_dir, exist_ok=True)
        save_acc_and_orn_records(acc_record, acc_orn_record, weight_dir)

        recalls = calculate_recalls(acc_record, acc_orn_record)
        combined_recalls[weight_key] = recalls
        save_recalls(recalls, weight_dir, weight_key)

    create_combined_results_table(combined_recalls, results_type_dir)



def evaluate_observation(prediction_type, config, device):
    # Extract configuration parameters
    dataset_dir = config.dataset_dir
    desdf_path = config.desdf_path
    log_dir_depth = config.log_dir_depth
    log_dir_semantic = config.log_dir_semantic
    results_dir = config.results_dir
    split_file = config.split_file

    # Instantiate dataset and models
    L = config.L
    D = config.D
    d_min = config.d_min
    d_max = config.d_max
    d_hyp = config.d_hyp

    # split_file = os.path.join(dataset_dir, "split.yaml")
    with open(split_file, "r") as f:
        split = AttrDict(yaml.safe_load(f))
    
    if config.use_saved_prob_vol:
        scene_names = split.test[:config.num_of_scenes]  # Use all test scenes
        # scene_names = split.val[:config.num_of_scenes]  # Use all test scenes
        # scene_names = split.train[:config.num_of_scenes] 
        test_set = ProbVolDataset(
            dataset_dir=dataset_dir,
            scene_names=scene_names,
            L=L,
            prob_vol_path=config.prob_vol_path,
            acc_only=False,
        )
    else:
        test_set = GridSeqDataset(
            dataset_dir,
            split.test[:config.num_of_scenes],
            # split.train[:config.num_of_scenes],
            L=L,
        )

    depth_net = None
    semantic_net = None

    if config.use_ground_truth_depth is False or config.use_ground_truth_semantic is False:
        if prediction_type in ["depth", "combined"]:
            depth_net = depth_net_pl.load_from_checkpoint(
                checkpoint_path=log_dir_depth,
                d_min=d_min,
                d_max=d_max,
                d_hyp=d_hyp,
                D=D,
            ).to(device)
            depth_net.eval()

        if prediction_type in ["semantic", "combined"]:
            if config.use_maskformer:
                semantic_net = semantic_net_pl_maskformer_small_room_type.load_from_checkpoint(
                    checkpoint_path=config.log_dir_semantic_and_room_aware,
                    num_classes=config.num_classes,
                    semantic_net_type = config.semantic_net_type
                ).to(device)
                semantic_net.eval()              
            else:
                semantic_net = semantic_net_pl.load_from_checkpoint(
                    checkpoint_path=log_dir_semantic,
                    num_classes=config.num_classes,
                ).to(device)
                semantic_net.eval()

    # Create results directory for this prediction type if it doesn't exist
    results_type_dir = os.path.join(results_dir, prediction_type)
    if config.use_ground_truth_depth or config.use_ground_truth_semantic:
        results_type_dir = os.path.join(results_type_dir, "gt")
    os.makedirs(results_type_dir, exist_ok=True)

    # Load desdf, semantics, maps, and ground truth poses
    desdfs, semantics, maps, gt_poses, valid_scene_names, _ = load_scene_data(
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
        valid_scene_names,
        config=config,  
    )


def main():
    parser = argparse.ArgumentParser(description="Observation evaluation.")
    parser.add_argument(
        "--config_file",
        type=str,
        default="evaluation/configuration/S3D/config_eval.yaml",
        # default="evaluation/configuration/zind/config_eval.yaml",
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

    # Extract prediction_type and use_ground_truth from the configuration
    prediction_type = config.prediction_type

    # Evaluate based on prediction_type
    if prediction_type == "all":
        for pred_type in ["depth", "semantic", "combined"]:
            evaluate_observation(
                pred_type, config, device
            )
    else:
        evaluate_observation(
            prediction_type, config, device
        )


if __name__ == "__main__":
    main()
