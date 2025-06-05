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
    finalize_localization_acc_only,
)
from utils.result_utils import (
    save_acc_and_orn_records,
    calculate_recalls,
    save_recalls,
    create_combined_results_table,
)
from utils.data_loader_helper import load_scene_data
from utils.visualization_utils import plot_prob_dist

from modules.combined.gt_prob_vol_loss_nets.prob_vol_net_pl_acc_only import ProbVolNetPLAccOnly

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
    
    acc_records_combined_net = []
    acc_orn_records_combined_net = []
    combinations = [
        {'net_size': 'small', 'dataset_size': 'tiny'},
        {'net_size': 'small', 'dataset_size': 'small'},
        {'net_size': 'small', 'dataset_size': 'medium'},
        {'net_size': 'small', 'dataset_size': 'full'},
        # {'net_size': 'medium', 'dataset_size': 'tiny'},
        # {'net_size': 'medium', 'dataset_size': 'small'},
        # {'net_size': 'medium', 'dataset_size': 'medium'},
        # {'net_size': 'medium', 'dataset_size': 'full'},
        # {'net_size': 'large', 'dataset_size': 'small'},
    ]
    combined_nets = {}
    for comb in combinations:
        net_size = comb['net_size']
        dataset_size = comb['dataset_size']
        # Load small_combined_net directly from checkpoint
        combined_nets[f"{net_size}_{dataset_size}"] = ProbVolNetPLAccOnly.load_from_checkpoint(
            checkpoint_path=f"/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/logs/combined/prb_vol_nets/combined_prob_vols_net_type-{net_size}_dataset_size-{dataset_size}_epochs-100/final_combined_model_checkpoint.ckpt",
            net_size=net_size,  
            strict=False 
        ).to(device)


    # Initialize records for small and medium nets
    acc_records_prob_vol_net = []
    acc_orn_records_small_net = []

    # Initialize records for weight combinations
    acc_records_for_all_weights = {
        f"{depth_weight}_{semantic_weight}": []
        for depth_weight, semantic_weight in weight_combinations
    }
    acc_orn_records_for_all_weights = {
        f"{depth_weight}_{semantic_weight}": []
        for depth_weight, semantic_weight in weight_combinations
    }

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

        # Use ground truth data for depth if specified
        if not config.use_ground_truth_depth:
            ref_img_torch = torch.tensor(data["ref_img"], device=device).unsqueeze(0)
            pred_depths, _, _ = depth_net.encoder(ref_img_torch, None) #TODO add masks
            pred_depths = pred_depths.squeeze(0).detach().cpu().numpy()
            pred_rays_depth = get_ray_from_depth(pred_depths, V=config.V, F_W= config.F_W)
        else:
            pred_depths = data["ref_depth"]
            pred_rays_depth = get_ray_from_depth(pred_depths, V=config.V, F_W= config.F_W)

        # Use ground truth data for semantics if specified
        if not config.use_ground_truth_semantic:
            ref_img_torch = torch.tensor(data["ref_img"], device=device).unsqueeze(0)
            _, _, prob = semantic_net.encoder(ref_img_torch, None)
            prob_squeezed = prob.squeeze(dim=0)
            sampled_indices = torch.multinomial(
                prob_squeezed, num_samples=1, replacement=True
            )
            sampled_indices = sampled_indices.squeeze(dim=1)
            sampled_indices_np = sampled_indices.cpu().numpy()
            pred_rays_semantic = get_ray_from_semantics(sampled_indices_np, V=config.V, F_W= config.F_W)
        else:
            sampled_indices_np = data["ref_semantics"]
            pred_rays_semantic = get_ray_from_semantics(sampled_indices_np, V=config.V, F_W= config.F_W)

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

        # Combine probabilities using weights
        for depth_weight, semantic_weight in weight_combinations:
            weight_key = f"{depth_weight}_{semantic_weight}"
            min_shape = [min(d, s) for d, s in zip(prob_vol_pred_depth.shape, prob_vol_pred_semantic.shape)]
            prob_vol_pred_depth_sliced = prob_vol_pred_depth[tuple(slice(0, min_shape[i]) for i in range(len(min_shape)))]
            prob_vol_pred_semantic_sliced = prob_vol_pred_semantic[tuple(slice(0, min_shape[i]) for i in range(len(min_shape)))]
    
            combined_prob_vol_pred = depth_weight * prob_vol_pred_depth_sliced + semantic_weight * prob_vol_pred_semantic_sliced

            # Assume finalize_localization returns pose_pred as list or numpy array
            prob_vol_pred, prob_dist_pred, orientations_pred, pose_pred = finalize_localization(combined_prob_vol_pred)
            
            # Convert pose_pred to torch.Tensor before calling .cpu()
            pose_pred = torch.tensor(pose_pred, device=device, dtype=torch.float32)
            pose_pred[:2] = pose_pred[:2] / 10  # Scale poses
            acc = torch.norm(pose_pred[:2] - torch.tensor(ref_pose_map[:2], device=device), p=2).item()
            acc_orn = ((pose_pred[2] - torch.tensor(ref_pose_map[2], device=device)) % (2 * np.pi)).item()
            acc_orn = min(acc_orn, 2 * np.pi - acc_orn) / np.pi * 180

            acc_records_for_all_weights[weight_key].append(acc)
            acc_orn_records_for_all_weights[weight_key].append(acc_orn)        

        # Now process with small and medium combined nets
        min_shape = [min(d, s) for d, s in zip(prob_vol_pred_depth.shape, prob_vol_pred_semantic.shape)]
        prob_vol_pred_depth_sliced = prob_vol_pred_depth[tuple(slice(0, min_shape[i]) for i in range(len(min_shape)))]
        prob_vol_pred_semantic_sliced = prob_vol_pred_semantic[tuple(slice(0, min_shape[i]) for i in range(len(min_shape)))]

        #do max pool
        depth_prob_map, _ = torch.max(prob_vol_pred_depth_sliced, dim=2)  # [H, W]
        semantic_prob_map, _ = torch.max(prob_vol_pred_semantic_sliced, dim=2)  # [H, W]

        for comb in combinations:
                    net_size = comb['net_size']
                    dataset_size = comb['dataset_size']
                    
                    with torch.no_grad():
                        combined_prob_vol_pred = combined_nets[f"{net_size}_{dataset_size}"](
                            depth_prob_map,
                            semantic_prob_map
                        )
                        # Remove batch dimension
                        combined_prob_vol_pred = combined_prob_vol_pred.detach()

                        # Perform localization
                        _, prob_dist_pred, _, pose_pred_combined = finalize_localization(combined_prob_vol_pred.squeeze(0).cpu())
                        

                    # Compute accuracy and orientation error
                    pose_pred_combined = torch.tensor(pose_pred_combined, device=device, dtype=torch.float32)
                    pose_pred_combined[:2] = pose_pred_combined[:2] / 10  # Scale poses
                    acc_combined = torch.norm(pose_pred_combined[:2] - torch.tensor(ref_pose_map[:2], device=device), p=2).item()
                    acc_orn_combined = ((pose_pred_combined[2] - torch.tensor(ref_pose_map[2], device=device)) % (2 * np.pi)).item()
                    acc_orn_combined = min(acc_orn_combined, 2 * np.pi - acc_orn_combined) / np.pi * 180

                    acc_records_combined_net.append(acc_combined)
                    acc_orn_records_combined_net.append(acc_orn_combined)
                    
                    acc_record_small = np.array(acc_records_combined_net)
                    acc_orn_record_small = np.array(acc_orn_records_combined_net)

                    small_net_dir = os.path.join(results_type_dir, f"{net_size}_{dataset_size}")
                    os.makedirs(small_net_dir, exist_ok=True)
                    save_acc_and_orn_records(acc_record_small, acc_orn_record_small, small_net_dir)

                    recalls_small = calculate_recalls(acc_record_small, acc_orn_record_small)
                    combined_recalls[f"{net_size}_{dataset_size}"] = recalls_small
                    save_recalls(recalls_small, small_net_dir, f"{net_size}_{dataset_size}")
        
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

    test_set = GridSeqDataset(
        dataset_dir,
        split.test,
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
        if prediction_type in ["semantic", "combined"]:
            semantic_net = semantic_net_pl.load_from_checkpoint(
                checkpoint_path=log_dir_semantic,
                num_classes=config.num_classes,
            ).to(device)

    # Create results directory for this prediction type if it doesn't exist
    results_type_dir = os.path.join(results_dir, prediction_type)
    if config.use_ground_truth_depth or config.use_ground_truth_semantic:
        results_type_dir = os.path.join(results_type_dir, "gt")
    os.makedirs(results_type_dir, exist_ok=True)

    # Load desdf, semantics, maps, and ground truth poses
    desdfs, semantics, maps, gt_poses, valid_scene_names = load_scene_data(
        test_set, dataset_dir, desdf_path
    )

    # Evaluate based on prediction_type
    if prediction_type == "combined":
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
    else:
        acc_record = []
        acc_orn_record = []

        for data_idx in tqdm.tqdm(range(len(test_set))):
            data = test_set[data_idx]
            scene_idx = np.sum(data_idx >= np.array(test_set.scene_start_idx)) - 1
            scene = test_set.scene_names[scene_idx]
            if 'floor' in scene: #zind
                pass
            else:
                scene_number = int(scene.split("_")[1])
                scene = f"scene_{scene_number}"
            idx_within_scene = data_idx - test_set.scene_start_idx[scene_idx]

            desdf = desdfs[scene]
            semantic = semantics[scene]
            ref_pose_map = gt_poses[scene][
                idx_within_scene * (config.L + 1) + config.L, :
            ]

            if not config.use_ground_truth_depth or not config.use_ground_truth_semantic:
                ref_img_torch = torch.tensor(data["ref_img"], device=device).unsqueeze(0)
                if prediction_type == "depth" and not config.use_ground_truth_depth:
                    pred_depths, _, _ = depth_net.encoder(ref_img_torch, None)
                    pred_depths = pred_depths.squeeze(0).detach().cpu().numpy()
                elif prediction_type == "semantic" and not config.use_ground_truth_semantic:
                    _, _, prob = semantic_net.encoder(ref_img_torch, None)
                    prob_squeezed = prob.squeeze(dim=0)
                    sampled_indices = torch.multinomial(
                        prob_squeezed, num_samples=1, replacement=True
                    )
                    sampled_indices = sampled_indices.squeeze(dim=1)
                    sampled_indices_np = sampled_indices.cpu().numpy()
            else:
                if prediction_type == "depth":
                    pred_depths = data["ref_depth"]
                elif prediction_type == "semantic":
                    sampled_indices_np = data["ref_semantics"]

            if prediction_type == "depth":
                pred_rays = get_ray_from_depth(pred_depths, V=config.V, F_W= config.F_W)
                pred_rays = torch.tensor(pred_rays, device="cpu")
                prob_vol_pred, prob_dist_pred, orientations_pred, pose_pred = localize(
                    torch.tensor(desdf["desdf"]), pred_rays
                )
            elif prediction_type == "semantic":
                pred_rays = get_ray_from_semantics(sampled_indices_np, V=config.V, F_W= config.F_W)
                pred_rays = torch.tensor(pred_rays, device="cpu")
                prob_vol_pred, prob_dist_pred, orientations_pred, pose_pred = localize(
                    torch.tensor(semantic["desdf"]), pred_rays
                )

            pose_pred[:2] = pose_pred[:2] / 10  # Scale poses
            acc = np.linalg.norm(pose_pred[:2] - ref_pose_map[:2], 2.0)
            acc_record.append(acc)
            acc_orn = (pose_pred[2] - ref_pose_map[2]) % (2 * np.pi)
            acc_orn = min(acc_orn, 2 * np.pi - acc_orn) / np.pi * 180
            acc_orn_record.append(acc_orn)

        acc_record = np.array(acc_record)
        acc_orn_record = np.array(acc_orn_record)

        # Save accuracy records and calculate recalls
        save_acc_and_orn_records(acc_record, acc_orn_record, results_type_dir)
        recalls = calculate_recalls(acc_record, acc_orn_record)
        save_recalls(recalls, results_type_dir, prediction_type)


def main():
    parser = argparse.ArgumentParser(description="Observation evaluation.")
    parser.add_argument(
        "--config_file",
        type=str,
        default="evaluation/configuration/S3D/config_eval_prob_vol.yaml",
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
