import os
import argparse
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from attrdict import AttrDict
import math
import tqdm

# == Local modules (adapt as needed) ==
from modules.depth.depth_net_pl import depth_net_pl
from modules.semantic.semantic_net_pl import semantic_net_pl

# from data_utils.data_utils_for_laser_train import GridSeqDataset
from data_utils.data_utils import LocalizationDataset
import numpy as np

# Helper utilities
from utils.data_loader_helper import load_scene_data
from utils.localization_utils import (
    get_ray_from_depth,
    get_ray_from_semantics,    
    localize,
    finalize_localization,
)

from utils.result_utils import (
    calculate_recalls,
)
# If you have your own ray_cast function, import it:
from utils.raycast_utils import ray_cast

# --- Refactored Modules ---
from evaluation.candidate_extractor import extract_top_k_locations
from evaluation.candidate_refiner import refine_and_select_best_candidate, angular_difference_deg
from evaluation.room_predictor import predict_room_and_get_polygons
from evaluation.result_handler import calculate_recalls, save_and_print_results
from evaluation.localization_helpers import combine_prob_volumes

def indices_to_radians(orientation_idx: int, num_orientations: int = 36) -> float:
    """
    Convert an orientation index (0..num_orientations-1) to radians [0, 2π).
    """
    return orientation_idx / num_orientations * 2.0 * math.pi

def extract_top_k_locations(
    prob_dist: np.ndarray,
    orientation_map: np.ndarray,
    K: int = 10,
    min_dist_m: float = 0.5,
    resolution_m_per_pixel: float = 0.1,
    num_orientations: int = 36,
):
    """
    From a 2D probability map (H, W), pick the top-K (x, y, orientation, prob_value)
    ensuring no two picks are within 'min_dist_m' in real-world space.
    """
    H, W = prob_dist.shape
    prob_dist_torch = torch.from_numpy(prob_dist)
    flat_prob = prob_dist_torch.view(-1)
    orientation_map = torch.from_numpy(orientation_map)
    flat_orient = orientation_map.view(-1)  # shape [H*W]

    sorted_indices = torch.argsort(flat_prob, descending=True)

    picks = []
    min_dist_pixels = min_dist_m / resolution_m_per_pixel
    excluded_mask = torch.zeros(H, W, dtype=torch.bool)

    for idx in sorted_indices:
        if len(picks) >= K:
            break
        y = idx // W
        x = idx % W
        if excluded_mask[y, x]:
            continue

        pick_orientation_idx = flat_orient[idx].item()
        pick_prob_value = flat_prob[idx].item()
        pick_orientation_rad = indices_to_radians(pick_orientation_idx, num_orientations)

        picks.append({
            'x': float(x.item()),
            'y': float(y.item()),
            'orientation_radians': pick_orientation_rad,
            'prob_value': pick_prob_value
        })

        # Exclude neighbors within min_dist_pixels
        y_min = max(0, int(y.item() - min_dist_pixels))
        y_max = min(H - 1, int(y.item() + min_dist_pixels))
        x_min = max(0, int(x.item() - min_dist_pixels))
        x_max = min(W - 1, int(x.item() + min_dist_pixels))
        for yy in range(y_min, y_max + 1):
            for xx in range(x_min, x_max + 1):
                dist = math.sqrt((yy - y.item())**2 + (xx - x.item())**2)
                if dist <= min_dist_pixels:
                    excluded_mask[yy, xx] = True

    return picks

def combine_prob_volumes(prob_vol_depth: torch.Tensor,
                         prob_vol_semantic: torch.Tensor,
                         depth_weight: float,
                         semantic_weight: float) -> torch.Tensor:
    """
    Combine two probability volumes [H, W, O] with given weights.
    We'll slice them to the min shared shape.
    Returns: [H', W', O']
    """
    H = min(prob_vol_depth.shape[0], prob_vol_semantic.shape[0])
    W = min(prob_vol_depth.shape[1], prob_vol_semantic.shape[1])
    O = min(prob_vol_depth.shape[2], prob_vol_semantic.shape[2])

    depth_sliced = prob_vol_depth[:H, :W, :O]
    semantic_sliced = prob_vol_semantic[:H, :W, :O]
    return depth_weight * depth_sliced + semantic_weight * semantic_sliced

def angular_difference_deg(ang1_rad, ang2_rad):
    """
    Minimal angular difference in degrees between two angles in radians.
    """
    diff_deg = abs(math.degrees(ang1_rad) - math.degrees(ang2_rad)) % 360
    return min(diff_deg, 360 - diff_deg)


def get_predicted_rays(model, img_torch, mask_torch, config, use_gt, gt_data):
    """Helper to get rays from a model or ground truth data."""
    if not use_gt:
        with torch.no_grad():
            if "depth" in gt_data:  # Distinguish by expected output
                pred_depths, _, _ = model.encoder(img_torch, mask_torch)
                pred_depths_np = pred_depths.squeeze(0).cpu().numpy()
                return get_ray_from_depth(pred_depths_np, V=config.V, F_W=config.F_W), pred_depths_np, None
            else:  # Assumes semantic net
                ray_logits, room_logits, _ = model(img_torch, mask_torch)
                ray_prob = F.softmax(ray_logits, dim=-1).squeeze(dim=0)
                sampled_indices = torch.multinomial(ray_prob, num_samples=1, replacement=True).squeeze(dim=1)
                pred_semantics_np = sampled_indices.cpu().numpy()
                return get_ray_from_semantics(pred_semantics_np), pred_semantics_np, room_logits
    else:
        if "depth" in gt_data:
            pred_depths_np = gt_data['depth']
            return get_ray_from_depth(pred_depths_np, V=config.V, F_W=config.F_W), pred_depths_np, None
        else:
            pred_semantics_np = gt_data['semantics']
            return get_ray_from_semantics(pred_semantics_np), pred_semantics_np, None

def evaluate_localization_pipeline(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # --- Load Dataset, Models, and Scene Data ---
    split_file = os.path.join(config.dataset_dir, "processed", "split.yaml")
    with open(split_file, "r") as f:
        split = AttrDict(yaml.safe_load(f))
    test_set = LocalizationDataset(os.path.join(config.dataset_dir, "processed"), split.test[:config.num_of_scenes])
    
    depth_net = None if config.use_ground_truth_depth else depth_net_pl.load_from_checkpoint(config.depth_weights).to(device).eval()
    semantic_net = None if config.use_ground_truth_semantic else semantic_net_pl.load_from_checkpoint(config.semantic_weights, num_classes=config.num_classes, num_room_types=config.num_room_types).to(device).eval()

    depth_df, semantic_df, maps, gt_poses, valid_scene_names, walls = load_scene_data(test_set, os.path.join(config.dataset_dir, "processed"), os.path.join(config.dataset_dir, "df"))

    # --- Main Loop ---
    for depth_weight, semantic_weight in tqdm.tqdm(config.weight_combinations, desc="Weight Combinations"):
        weight_key = f"{depth_weight}_{semantic_weight}"
        print(f"\nEvaluating: {weight_key}")

        errors = {k: {'trans': [], 'rot': []} for k in ['baseline', 'baseline_n', 'refine', 'refine_n']}

        for data_idx in tqdm.tqdm(range(len(test_set)), desc="Samples", leave=False):
            data = test_set[data_idx]
            scene_idx = np.sum(data_idx >= np.array(test_set.scene_start_idx)) - 1
            scene_name = test_set.scene_names[scene_idx]
            scene_name = f"scene_{int(scene_name.split('_')[1])}" if 'floor' not in scene_name else scene_name
            if scene_name not in valid_scene_names: continue

            gt_x, gt_y, gt_o = gt_poses[scene_name][data_idx - test_set.scene_start_idx[scene_idx], :3]
            img_torch = torch.tensor(data["ref_img"], device=device).unsqueeze(0)
            mask_torch = torch.tensor(data["ref_mask"], device=device).unsqueeze(0)
            
            pred_rays_depth, pred_depths_np, _ = get_predicted_rays(depth_net, img_torch, mask_torch, config, config.use_ground_truth_depth, {"depth": data["ref_depth"]})
            pred_rays_semantic, pred_semantics_np, room_logits = get_predicted_rays(semantic_net, img_torch, mask_torch, config, config.use_ground_truth_semantic, {"semantics": data["ref_semantics"]})
            
            prob_vol_depth, _, _, _ = localize(torch.tensor(depth_df[scene_name]), torch.tensor(pred_rays_depth), return_np=False)
            prob_vol_semantic, _, _, _ = localize(torch.tensor(semantic_df[scene_name]), torch.tensor(pred_rays_semantic), return_np=False)
            
            combined_prob_vol = combine_prob_volumes(prob_vol_depth, prob_vol_semantic, depth_weight, semantic_weight)

            room_polygons = predict_room_and_get_polygons(room_logits, data["room_polygons"], config.room_selection_threshold, config.is_zind) if config.use_room_aware and room_logits is not None else []
            
            # --- Room-Aware Localization ---
            _, prob_dist, orient_map, pose = finalize_localization(combined_prob_vol, data["room_polygons"], room_polygons)
            px, py, po = np.array(pose) / [10., 10., 1.]
            errors['baseline']['trans'].append(np.sqrt((px - gt_x)**2 + (py - gt_y)**2))
            errors['baseline']['rot'].append(angular_difference_deg(po, gt_o))
            
            candidates = extract_top_k_locations(prob_dist, orient_map, K=config.top_k, min_dist_m=config.min_dist_m, resolution_m_per_pixel=config.resolution_m_per_pixel)
            loc, orient, _ = refine_and_select_best_candidate(candidates, walls[scene_name], maps[scene_name], pred_depths_np, pred_semantics_np, config.alpha_similarity)
            if loc:
                errors['refine']['trans'].append(np.sqrt((loc[0] - gt_x)**2 + (loc[1] - gt_y)**2))
                errors['refine']['rot'].append(angular_difference_deg(orient, gt_o))

            # --- Non-Room-Aware Localization ---
            _, prob_dist_n, orient_map_n, pose_n = finalize_localization(combined_prob_vol, data["room_polygons"])
            px_n, py_n, po_n = np.array(pose_n) / [10., 10., 1.]
            errors['baseline_n']['trans'].append(np.sqrt((px_n - gt_x)**2 + (py_n - gt_y)**2))
            errors['baseline_n']['rot'].append(angular_difference_deg(po_n, gt_o))

            candidates_n = extract_top_k_locations(prob_dist_n, orient_map_n, K=config.top_k, min_dist_m=config.min_dist_m, resolution_m_per_pixel=config.resolution_m_per_pixel)
            loc_n, orient_n, _ = refine_and_select_best_candidate(candidates_n, walls[scene_name], maps[scene_name], pred_depths_np, pred_semantics_np, config.alpha_similarity)
            if loc_n:
                errors['refine_n']['trans'].append(np.sqrt((loc_n[0] - gt_x)**2 + (loc_n[1] - gt_y)**2))
                errors['refine_n']['rot'].append(angular_difference_deg(orient_n, gt_o))

        # --- Calculate and Save Recalls ---
        results_dir = os.path.join(config.results_dir, "gt" if config.use_ground_truth_depth or config.use_ground_truth_semantic else "")
        recalls = {k: calculate_recalls(np.array(v['trans']), np.array(v['rot'])) for k, v in errors.items()}
        save_and_print_results(results_dir, weight_key, recalls['baseline'], recalls['baseline_n'], recalls['refine'], recalls['refine_n'])

def main():
    parser = argparse.ArgumentParser(description="Run the localization evaluation pipeline.")    
    parser.add_argument("--config_file", type=str, default="evaluation/configuration/S3D/config_eval.yaml", help="Path to the configuration file.")
    args = parser.parse_args()
    with open(args.config_file, "r") as f:
        config = AttrDict(yaml.safe_load(f))
    evaluate_localization_pipeline(config)

if __name__ == "__main__":
    main()
