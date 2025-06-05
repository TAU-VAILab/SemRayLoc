import os
import argparse
import yaml
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from attrdict import AttrDict
from tqdm import tqdm
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

# == Local modules (adapt as needed) ==
from modules.mono.depth_net_pl import depth_net_pl
from modules.semantic.semantic_net_pl import semantic_net_pl
from modules.semantic.semantic_net_pl_maskformer import semantic_net_pl_maskformer
from modules.semantic.semantic_net_pl_maskformer_small import semantic_net_pl_maskformer_small
from modules.semantic.room_type_pred.room_type_pred_resnet50_pl import room_type_pred_resnet50_pl
from modules.semantic.semantic_mapper import room_type_to_id

from data_utils.data_utils import GridSeqDataset
from data_utils.prob_vol_data_utils import ProbVolDataset

# Helper utilities
from utils.data_loader_helper import load_scene_data
from utils.localization_utils import (
    get_ray_from_depth,
    get_ray_from_semantics_v2,
    localize,
    finalize_localization,
)
from utils.result_utils import (
    calculate_recalls,
    save_acc_and_orn_records,
    save_recalls,
    create_combined_results_table,
)
# If you have your own ray_cast function, import it:
# from utils.raycast_utils import ray_cast

from data_utils.prob_vol_data_utils import ProbVolDataset
from utils.data_loader_helper import load_scene_data
from utils.raycast_utils import ray_cast  # Ensure your ray_cast function is available

def indices_to_radians(orientation_idx: int, num_orientations: int = 36) -> float:
    """
    Convert an orientation index (0..num_orientations-1) to radians [0, 2π).
    """
    return orientation_idx / num_orientations * 2.0 * math.pi


def extract_top_k_locations(
    prob_dist: torch.Tensor,
    orientation_map: torch.Tensor,
    K: int = 10,
    min_dist_m: float = 0.05,
    resolution_m_per_pixel: float = 0.1,
    num_orientations: int = 36,
):
    """
    From a 2D probability map (H, W), pick the top-K (x, y, orientation, prob_value)
    ensuring no two picks are within 'min_dist_m' in real-world space.

    prob_dist: (H, W)  -- final probability distribution across translation
    orientation_map: (H, W) -- argmax orientation index
    K: how many top picks
    min_dist_m: minimum distance (in meters) to separate picks
    resolution_m_per_pixel: resolution of map
    num_orientations: total orientation bins

    Returns: a list of dict with keys: x, y, orientation_radians, prob_value
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


def get_max_over_orientation(prob_vol: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    Convert [H, W, O] -> (prob_dist: [H, W], orientation_map: [H, W])
    by taking max across the orientation dimension.
    """
    prob_dist, orientation_map = torch.max(prob_vol, dim=2)  # (H, W) each
    return prob_dist, orientation_map


def compute_rays_from_candidate(
    walls_map: np.ndarray,
    semantic_map: np.ndarray,
    cand,
    augmentation_offsets,
):
    candidate_cache = {}

    ray_n = 40  # number of rays per candidate per augmentation
    F_W = 1 / np.tan(0.698132) / 2
    depth_max = 15  # maximum depth in meters

    base_x = cand['x']
    base_y = cand['y']
    base_orientation = cand['orientation_radians']

    # Convert candidate's pixel coordinates to the proper scale.
    center_x = base_x * 10
    center_y = base_y * 10
    candidate_pos_pixels = np.array([center_x, center_y])

    # Pre-compute the baseline set of relative angles.
    center_angs = np.flip(
        np.arctan2((np.arange(ray_n) - np.arange(ray_n).mean()), ray_n * F_W)
    )

    candidate_aug_rays = {}

    for aug_key, aug_offset in augmentation_offsets.items():
        # The effective offset for this augmentation.
        effective_orientation = base_orientation + aug_offset
        # Compute the full list of ray angles.
        ray_angles = center_angs + effective_orientation

        depth_rays = []
        semantic_rays = []
        candidate_ray_endpoints = []

        for ang in ray_angles:
            # Cache key based on candidate position and the specific ray angle.
            cache_key = (round(base_x, 3), round(base_y, 3), round(ang, 3))
            if cache_key in candidate_cache:
                # Retrieve the cached result.
                depth_val_m, prediction_class = candidate_cache[cache_key]
            else:
                # Cast the ray on the walls map.
                dist_depth, _, hit_coords_walls, _ = ray_cast(
                    walls_map, candidate_pos_pixels, ang, dist_max=depth_max * 100
                )
                # Cast the ray on the semantic map.
                _, prediction_class, _, _ = ray_cast(
                    semantic_map, candidate_pos_pixels, ang, dist_max=depth_max * 100, min_dist=80
                )
                depth_val_m = dist_depth / 100.0
                # Cache the result.
                candidate_cache[cache_key] = (depth_val_m, prediction_class)

            depth_rays.append(depth_val_m)
            semantic_rays.append(prediction_class)

        # (Optional) Flip the ray endpoints order if needed.
        endpoints_tensor = torch.tensor(candidate_ray_endpoints)
        endpoints_tensor = torch.flip(endpoints_tensor, [0])

        candidate_aug_rays[aug_key] = (depth_rays, semantic_rays)

    return candidate_aug_rays


def measure_similarity(
    pred_depths: np.ndarray,
    pred_semantics: np.ndarray,
    cand_depths: np.ndarray,
    cand_semantics: np.ndarray,
    alpha: float = 0.5,
):
    """
    Compute a simple error measure combining:
      - Depth difference (L1)
      - Semantic mismatch
    alpha in [0..1], controlling how we mix the two errors:
       error = alpha * mean_L1_depth + (1 - alpha) * mean_sem_mismatch
    """
    cand_semantics[cand_semantics == 2] = -1
    cand_semantics[cand_semantics == 1] = 2
    cand_semantics[cand_semantics == -1] = 1
    # depth error
    depth_error = np.mean(np.abs(pred_depths - cand_depths))
    # semantic mismatch fraction
    sem_error = np.mean((pred_semantics != cand_semantics).astype(float))

    score = alpha * depth_error + (1.0 - alpha) * sem_error
    return score


def angular_difference_deg(ang1_rad, ang2_rad):
    """
    Minimal angular difference in degrees between two angles in radians.
    """
    diff_deg = abs(math.degrees(ang1_rad) - math.degrees(ang2_rad)) % 360
    return min(diff_deg, 360 - diff_deg)


def evaluate_room_aware_with_refine(config):
    """
    Main pipeline:
      1) Load data & models
      2) For each sample, compute combined probability volume.
      3) Baseline: finalize_localization => get (prob_dist, orientation_map, pose_pred)
      4) Evaluate baseline error
      5) Extract top-K from prob_dist
      6) For each candidate, sample orientation offsets => measure similarity => pick best orientation
      7) Pick the final location among these K (or just report for the top candidate) => Evaluate refined error
      8) Summarize results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # ---------- Load the dataset -----------
    with open(config.split_file, "r") as f:
        split = AttrDict(yaml.safe_load(f))

    # For example, take test scenes (up to config.num_of_scenes)
    scene_names = split.test[: config.num_of_scenes]
    if config.use_saved_prob_vol:
        test_set = ProbVolDataset(
            dataset_dir=config.dataset_dir,
            scene_names=scene_names,
            L=config.L,
            prob_vol_path=config.prob_vol_path,
            acc_only=False,
        )
    else:
        test_set = GridSeqDataset(
            dataset_dir=config.dataset_dir,
            scene_names=scene_names,
            L=config.L,
        )

    # ---------- Load models -----------
    # Depth
    depth_net = None
    if not config.use_ground_truth_depth or config.prediction_type in ["depth", "combined"]:
        depth_net = depth_net_pl.load_from_checkpoint(
            checkpoint_path=config.log_dir_depth,
            d_min=config.d_min,
            d_max=config.d_max,
            d_hyp=config.d_hyp,
            D=config.D,
        ).to(device)
        depth_net.eval()

    # Semantic
    semantic_net = None
    if not config.use_ground_truth_semantic or config.prediction_type in ["semantic", "combined"]:
        if config.use_maskformer:
            if config.use_small_maskformer:
                semantic_net = semantic_net_pl_maskformer_small.load_from_checkpoint(
                    checkpoint_path=config.log_dir_semantic_maskformer_small,
                    num_classes=config.num_classes,
                ).to(device)
                semantic_net.eval()
            else:
                semantic_net = semantic_net_pl_maskformer.load_from_checkpoint(
                    checkpoint_path=config.log_dir_semantic_maskformer,
                    num_classes=config.num_classes,
                ).to(device)
                semantic_net.eval()
        else:
            semantic_net = semantic_net_pl.load_from_checkpoint(
                checkpoint_path=config.log_dir_semantic,
                num_classes=config.num_classes,
            ).to(device)
            semantic_net.eval()

    # Room-aware model
    room_type_model = None
    if config.use_room_aware:
        room_type_model = room_type_pred_resnet50_pl.load_from_checkpoint(
            config.room_aware_checkpoint
        ).to(device)
        room_type_model.eval()

    # Create a directory for results
    results_type_dir = os.path.join(config.results_dir, config.prediction_type)
    if config.use_ground_truth_depth or config.use_ground_truth_semantic:
        results_type_dir = os.path.join(results_type_dir, "gt")
    os.makedirs(results_type_dir, exist_ok=True)

    # ---------- Load scene data -----------
    desdfs, semantics, maps, gt_poses, valid_scene_names, walls = load_scene_data(
        test_set, config.dataset_dir, config.desdf_path
    )

    # Store error records for baseline vs. refine
    baseline_trans_errors = []
    baseline_rot_errors = []
    baseline_trans_errors_n = []
    baseline_rot_errors_n = []
    refine_trans_errors = []
    refine_rot_errors = []

    # Hyperparams for refine
    top_k = 5
    min_dist_m = 0.05

    # A simple alpha for measuring difference in rays: 0.5 => half depth, half semantic mismatch
    alpha_similarity = 0.8
    resolution_m_per_pixel = 0.1

    # For each data index
    for data_idx in tqdm(range(len(test_set)), desc="Evaluating"):
        # Some items from the dataset
        data = test_set[data_idx]
        scene_idx = np.sum(data_idx >= np.array(test_set.scene_start_idx)) - 1
        scene = test_set.scene_names[scene_idx]
        if 'floor' not in scene:
            try:
                scene_number = int(scene.split("_")[1])
                scene = f"scene_{scene_number}"
            except:
                pass

        if scene not in valid_scene_names:
            # Skip invalid scenes
            continue

        idx_within_scene = data_idx - test_set.scene_start_idx[scene_idx]
        ref_pose_map = gt_poses[scene][idx_within_scene * (config.L + 1) + config.L, :]

        # 1) Depth rays
        if not config.use_ground_truth_depth:
            ref_img_torch = torch.tensor(data["ref_img"], device=device).unsqueeze(0)
            ref_mask_torch = torch.tensor(data["ref_mask"], device=device).unsqueeze(0)
            with torch.no_grad():
                pred_depths, _, _ = depth_net.encoder(ref_img_torch, ref_mask_torch)
            pred_depths = pred_depths.squeeze(0).cpu().numpy()  # shape [40] (for example)
            pred_rays_depth = get_ray_from_depth(pred_depths, V=config.V, F_W=config.F_W)
        else:
            # ground truth
            pred_depths = data["ref_depth"]
            pred_rays_depth = get_ray_from_depth(pred_depths, V=config.V, F_W=config.F_W)

        # 2) Semantic rays
        if not config.use_ground_truth_semantic:
            ref_img_torch = torch.tensor(data["ref_img"], device=device).unsqueeze(0)
            ref_mask_torch = torch.tensor(data["ref_mask"], device=device).unsqueeze(0)
            with torch.no_grad():
                _, _, prob = semantic_net.encoder(ref_img_torch, ref_mask_torch)
            prob_squeezed = prob.squeeze(dim=0)
            sampled_indices = torch.multinomial(prob_squeezed, num_samples=1, replacement=True)
            sampled_indices = sampled_indices.squeeze(dim=1)
            sampled_semantic_indices_np = sampled_indices.cpu().numpy()
            pred_rays_semantic = get_ray_from_semantics_v2(sampled_semantic_indices_np)
        else:
            # ground truth
            sampled_semantic_indices_np = data["ref_semantics"]
            pred_rays_semantic = get_ray_from_semantics_v2(sampled_semantic_indices_np)

        # 3) Probability volumes for depth & semantic
        #    If we are not using saved volumes, compute them via localize().
        if not config.use_saved_prob_vol:
            # localize depth
            prob_vol_pred_depth, _, _, _ = localize(
                torch.tensor(desdfs[scene]["desdf"]),
                torch.tensor(pred_rays_depth, device="cpu"),
                return_np=False,
            )
            # localize semantic
            prob_vol_pred_semantic, _, _, _ = localize(
                torch.tensor(semantics[scene]["desdf"]),
                torch.tensor(pred_rays_semantic, device="cpu"),
                return_np=False,
                localize_type="semantic",
            )
        else:
            # Use precomputed volumes
            if config.use_ground_truth_depth:
                prob_vol_pred_depth = data['prob_vol_depth_gt'].to(device)
            else:
                prob_vol_pred_depth = data['prob_vol_depth'].to(device)

            if config.use_ground_truth_semantic:
                prob_vol_pred_semantic = data['prob_vol_semantic_gt'].to(device)
            else:
                prob_vol_pred_semantic = data['prob_vol_semantic'].to(device)

        # 4) Combine volumes using the first (and only) weight combination for demonstration
        #    Or you can loop over config.weight_combinations if you want multiple runs
        depth_weight, semantic_weight = config.weight_combinations[0]
        combined_prob_vol = combine_prob_volumes(
            prob_vol_pred_depth,
            prob_vol_pred_semantic,
            depth_weight,
            semantic_weight
        )

        # 5) If "room-aware," do room polygon filtering
        if config.use_room_aware:
            # Predict the room type for this image
            room_logits, _ = room_type_model(torch.tensor(data["ref_img"], device=device).unsqueeze(0))
            class_probs = torch.softmax(room_logits, dim=1)
            max_prob, room_idx = torch.max(class_probs, dim=1)

            # If the predicted room type has probability > threshold, filter using polygons
            room_polygons = []
            if max_prob.item() > config.room_selection_threshold:
                id_to_room_type = {v: k for k, v in room_type_to_id.items()}
                predicted_room = id_to_room_type[room_idx.item()]
                # retrieve polygons
                room_polygons = data["room_polygons"].get(predicted_room, [])

            # finalize
            final_prob_vol, prob_dist_pred, orientation_map, pose_pred = finalize_localization(
                combined_prob_vol, data["room_polygons"], room_polygons
            )
            final_prob_vol, prob_dist_pred_no_room_aware, orientation_map, pose_pred_no_room_aware = finalize_localization(
                combined_prob_vol, data["room_polygons"]
            )
        else:
            final_prob_vol, prob_dist_pred, orientation_map, pose_pred = finalize_localization(
                combined_prob_vol
            )

        # 6) No room aware Baseline error
        # pose_pred is [x, y, o], in the same scale used by the floor plan. In your code, you might do a /10:
        #   "pose_pred[:2] = pose_pred[:2] / 10"
        # Check if finalize_localization already did the scaling or not. Adjust accordingly.
        # Suppose we do the same scaling as in your snippet:
        pose_pred_np_no_room_aware = np.array(pose_pred_no_room_aware, dtype=np.float32)
        pose_pred_np_no_room_aware[0:2] = pose_pred_np_no_room_aware[0:2] / 10.0  # from pixel coords to meters?

        gt_x, gt_y, gt_o = ref_pose_map[:3]
        pred_x_n, pred_y_n, pred_o_n = pose_pred_np_no_room_aware
        baseline_trans_n = np.sqrt((pred_x_n - gt_x)**2 + (pred_y_n - gt_y)**2)
        # minimal rotation difference
        rot_diff_deg_n = abs(pred_o_n - gt_o) % (2 * math.pi)
        baseline_rot_n = min(rot_diff_deg_n, 2*math.pi - rot_diff_deg_n) / math.pi * 180.0

        baseline_trans_errors_n.append(baseline_trans_n)
        baseline_rot_errors_n.append(baseline_rot_n)
        # 6) Baseline error
        # pose_pred is [x, y, o], in the same scale used by the floor plan. In your code, you might do a /10:
        #   "pose_pred[:2] = pose_pred[:2] / 10"
        # Check if finalize_localization already did the scaling or not. Adjust accordingly.
        # Suppose we do the same scaling as in your snippet:
        pose_pred_np = np.array(pose_pred, dtype=np.float32)
        pose_pred_np[0:2] = pose_pred_np[0:2] / 10.0  # from pixel coords to meters?

        gt_x, gt_y, gt_o = ref_pose_map[:3]
        pred_x, pred_y, pred_o = pose_pred_np
        baseline_trans = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
        # minimal rotation difference
        rot_diff_deg = abs(pred_o - gt_o) % (2 * math.pi)
        baseline_rot = min(rot_diff_deg, 2*math.pi - rot_diff_deg) / math.pi * 180.0

        baseline_trans_errors.append(baseline_trans)
        baseline_rot_errors.append(baseline_rot)

        # 7) Extract top-K from prob_dist_pred => shape [H, W]
        #    orientation_map => shape [H, W]
        top_k_candidates = extract_top_k_locations(
            prob_dist_pred,
            orientation_map,
            K=top_k,
            min_dist_m=min_dist_m,
            resolution_m_per_pixel=resolution_m_per_pixel,
            num_orientations=36
        )

        # 8) For each candidate, we attempt orientation refinement by sampling orientation_offsets_rad.
        augmentation_offsets = {
            "0": 0,
            "5":  np.deg2rad(5),
            "10": np.deg2rad(10),
            # "15": np.deg2rad(15),
            # "20": np.deg2rad(20),
            "-5":  np.deg2rad(-5),
            "-10": np.deg2rad(-10),
            # "-15": np.deg2rad(-15),
            # "-20": np.deg2rad(-20),
        }   
        # Ray-casting needs a walls_map and semantic_map for the scene. We'll get them from 'walls' and 'maps'.
        # Make sure they are np.ndarray and share the same coordinate system as finalize_localization used:
        # Typically, `maps[scene]` is your semantic map, `walls[scene]` is your obstacle map.
        # We'll treat them as 0.1 m/px resolution if that matches your final localization approach.
        # If not, adapt accordingly.
        walls_map_scene = walls[scene]       # shape [H, W], for example
        semantic_map_scene = maps[scene]     # shape [H, W]

        # We'll store the best location among these K, or just pick the top candidate after refinement.
        best_candidate_score = 1e9
        best_candidate_location = None
        best_candidate_orientation = None

        # --- Parallel candidate refinement ---
        def process_candidate(cand):
            cand_px = cand['x']
            cand_py = cand['y']
            cand_o = cand['orientation_radians']
            
            cand_rays_dict = compute_rays_from_candidate(
                walls_map_scene,
                semantic_map_scene,
                cand,
                augmentation_offsets
            )
    
            best_offset_score = 1e9
            best_offset = 0.0
    
            for off in augmentation_offsets:
                cand_depths, cand_sems = cand_rays_dict[off]
                # Measure similarity
                score = measure_similarity(
                    pred_depths,
                    sampled_semantic_indices_np,
                    cand_depths,
                    cand_sems,
                    alpha=alpha_similarity
                )
                if score < best_offset_score:
                    best_offset_score = score
                    best_offset = off
    
            # The best orientation for this candidate
            refined_o = cand_o + np.deg2rad(int(best_offset))
            # measure final location error vs. GT, once we fix that orientation
            cand_x_m = (cand_px / 10.0)
            cand_y_m = (cand_py / 10.0)
    
            # trans error
            refine_trans = np.sqrt((cand_x_m - gt_x)**2 + (cand_y_m - gt_y)**2)
            # rot error
            refine_rot = angular_difference_deg(refined_o, gt_o)
    
            # We'll define a "combined" error (pick_score) based on translation error
            pick_score = refine_trans  # or best_offset_score
            return pick_score, cand_x_m, cand_y_m, refined_o

        with ThreadPoolExecutor(max_workers=len(top_k_candidates)) as executor:
            futures = [executor.submit(process_candidate, cand) for cand in top_k_candidates]
            for future in as_completed(futures):
                pick_score, cand_x_m, cand_y_m, refined_o = future.result()
                if pick_score < best_candidate_score:
                    best_candidate_score = pick_score
                    best_candidate_location = (cand_x_m, cand_y_m)
                    best_candidate_orientation = refined_o

        # 9) Evaluate the final "best candidate" among the K
        if best_candidate_location is not None:
            refine_x, refine_y = best_candidate_location
            refine_o = best_candidate_orientation
            refine_trans = np.sqrt((refine_x - gt_x)**2 + (refine_y - gt_y)**2)
            refine_rot = angular_difference_deg(refine_o, gt_o)
    
            refine_trans_errors.append(refine_trans)
            refine_rot_errors.append(refine_rot)
        else:
            # If for some reason we didn't pick a candidate
            refine_trans_errors.append(baseline_trans)
            refine_rot_errors.append(baseline_rot)

    # After processing all items, compute recall metrics
    baseline_trans_errors = np.array(baseline_trans_errors)
    baseline_rot_errors = np.array(baseline_rot_errors)
    baseline_trans_errors_n = np.array(baseline_trans_errors_n)
    baseline_rot_errors_n = np.array(baseline_rot_errors_n)
    refine_trans_errors = np.array(refine_trans_errors)
    refine_rot_errors = np.array(refine_rot_errors)

    baseline_recalls = calculate_recalls(baseline_trans_errors, baseline_rot_errors)
    baseline_recalls_n = calculate_recalls(baseline_trans_errors_n, baseline_rot_errors_n)
    refine_recalls = calculate_recalls(refine_trans_errors, refine_rot_errors)

    print("\n==== Baseline No Room Aware Recalls ====")
    for k, v in baseline_recalls_n.items():
        print(f"{k}: {v:.3f}")
        
    print("\n==== Baseline Recalls ====")
    for k, v in baseline_recalls.items():
        print(f"{k}: {v:.3f}")    

    print("\n==== Refine Recalls ====")
    for k, v in refine_recalls.items():
        print(f"{k}: {v:.3f}")

    # You can save them in the results_type_dir if you like:
    with open(os.path.join(results_type_dir, "baseline_recalls_n.json"), "w") as f:
        json.dump(baseline_recalls_n, f, indent=4)
    with open(os.path.join(results_type_dir, "baseline_recalls.json"), "w") as f:
        json.dump(baseline_recalls, f, indent=4)
    with open(os.path.join(results_type_dir, "refine_recalls.json"), "w") as f:
        json.dump(refine_recalls, f, indent=4)

    # Optionally store the raw errors, do more analysis, etc.


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

    evaluate_room_aware_with_refine(config)


if __name__ == "__main__":
    main()
