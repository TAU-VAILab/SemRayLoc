import argparse
import os
import numpy as np
import torch
import tqdm
import yaml
from attrdict import AttrDict
import gzip

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

from utils.data_loader_helper import load_scene_data
from utils.visualization_utils import plot_prob_dist

from modules.combined.expected_pose_pred_loss_nets.combined_net_pl import CombinedProbVolNetPL
import torch.nn.functional as F  # For MSE and other similarity measures

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

    match_stats = {
        "depth_matches": 0,
        "semantic_matches": 0,
        "total_depth_comparisons": 0,
        "total_semantic_comparisons": 0,
        "depth_mse": 0.0,
        "semantic_mse": 0.0,
        "failed_scenes": []
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
            pred_depths, _, _ = depth_net.encoder(ref_img_torch, None) # TODO add masks
            pred_depths = pred_depths.squeeze(0).detach().cpu().numpy()
            pred_rays_depth = get_ray_from_depth(pred_depths, V=config.V, F_W=config.F_W)
        else:
            pred_depths = data["ref_depth"]
            pred_rays_depth = get_ray_from_depth(pred_depths, V=config.V, F_W=config.F_W)

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
            pred_rays_semantic = get_ray_from_semantics(sampled_indices_np, V=config.V, F_W=config.F_W)
        else:
            sampled_indices_np = data["ref_semantics"]
            pred_rays_semantic = get_ray_from_semantics(sampled_indices_np, V=config.V, F_W=config.F_W)

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

        base_path = "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/test_data_set_full/prob_vols"
        depth_prob_vol_path = os.path.join(base_path, f"{scene}/camera_{idx_within_scene}_pred_depth_prob_vol.pt.gz")
        semantic_prob_vol_path = os.path.join(base_path, f"{scene}/camera_{idx_within_scene}_pred_semantic_prob_vol.pt.gz")
        
        # Load saved probability volumes
        def load_gzipped_tensor(file_path):
            with gzip.open(file_path, 'rb') as f:
                return torch.load(f)
        
        saved_depth_prob_vol = load_gzipped_tensor(depth_prob_vol_path)
        saved_semantic_prob_vol = load_gzipped_tensor(semantic_prob_vol_path)

        # Now compare the tensors
        depth_identical = torch.allclose(prob_vol_pred_depth, saved_depth_prob_vol, atol=1e-5)
        semantic_identical = torch.allclose(prob_vol_pred_semantic, saved_semantic_prob_vol, atol=1e-5)

        # Increment match statistics
        match_stats["total_depth_comparisons"] += 1
        match_stats["total_semantic_comparisons"] += 1

        if depth_identical:
            match_stats["depth_matches"] += 1
        else:
            match_stats['failed_scenes'].append(scene)
            print(f"failed depth in: {scene}")

        if semantic_identical:
            match_stats["semantic_matches"] += 1

        # Calculate MSE for depth and semantic probability volumes
        depth_mse = F.mse_loss(prob_vol_pred_depth, saved_depth_prob_vol).item()
        semantic_mse = F.mse_loss(prob_vol_pred_semantic, saved_semantic_prob_vol).item()

        match_stats["depth_mse"] += depth_mse
        match_stats["semantic_mse"] += semantic_mse

    # Finalize match statistics
    match_stats["depth_accuracy"] = (
        match_stats["depth_matches"] / match_stats["total_depth_comparisons"]
    )
    match_stats["semantic_accuracy"] = (
        match_stats["semantic_matches"] / match_stats["total_semantic_comparisons"]
    )
    match_stats["mean_depth_mse"] = (
        match_stats["depth_mse"] / match_stats["total_depth_comparisons"]
    )
    match_stats["mean_semantic_mse"] = (
        match_stats["semantic_mse"] / match_stats["total_semantic_comparisons"]
    )

    # Print summary statistics
    print("\n=== Match Statistics ===")
    print(f"Depth Matches: {match_stats['depth_matches']}")
    print(f"Semantic Matches: {match_stats['semantic_matches']}")
    print(f"Depth Accuracy: {match_stats['depth_accuracy']:.2%}")
    print(f"Semantic Accuracy: {match_stats['semantic_accuracy']:.2%}")
    print(f"Mean Depth MSE: {match_stats['mean_depth_mse']:.6f}")
    print(f"Mean Semantic MSE: {match_stats['mean_semantic_mse']:.6f}")
    print(f"print failed scenes: {match_stats['failed_scenes']}")




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
    # scene_names = split.train[2700:2702]
    scene_names = split.val[240:242]
    test_set = GridSeqDataset(
        dataset_dir,
        scene_names,
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
    desdfs, semantics, maps, gt_poses, valid_scene_names = load_scene_data(
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
