import os
import argparse
import yaml
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from attrdict import AttrDict
import math
import tqdm
import time

# Parallel execution
from concurrent.futures import ThreadPoolExecutor, as_completed

# == Local modules (adapt as needed) ==
from modules.mono.depth_net_pl import depth_net_pl
from modules.semantic.semantic_net_pl_maskformer_small_room_type import semantic_net_pl_maskformer_small_room_type
from modules.semantic.semantic_mapper import room_type_to_id, zind_room_type_to_id

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
    create_combined_results_table,
)
# If you have your own ray_cast function, import it:
from utils.raycast_utils import ray_cast

from utils.visualization_utils import plot_prob_dist
# =============================================================================
# Utility functions
# =============================================================================

import numpy as np
import math
from skimage.feature import peak_local_max

def indices_to_radians(orientation_idx: int, num_orientations: int = 36) -> float:
    """
    Convert an orientation index (0..num_orientations-1) to radians [0, 2π).
    """
    return orientation_idx / num_orientations * 2.0 * math.pi

def extract_top_k_locations(
    prob_dist: np.ndarray,
    orientation_map: np.ndarray,
    K: int = 10,
    min_dist_m: float = 0.05,
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
        effective_orientation = base_orientation + aug_offset
        ray_angles = center_angs + effective_orientation

        depth_rays = []
        semantic_rays = []

        for ang in ray_angles:
            cache_key = (round(base_x, 2), round(base_y, 2), round(math.degrees(ang)))
            if cache_key in candidate_cache:
                depth_val_m, prediction_class = candidate_cache[cache_key]
            else:
                depth_val_m, prediction_class, _, _ = ray_cast(
                    semantic_map, candidate_pos_pixels, ang, dist_max=depth_max * 100, min_dist=5, cast_type=2
                )
                depth_val_m = depth_val_m / 100.0
                candidate_cache[cache_key] = (depth_val_m, prediction_class)

            depth_rays.append(depth_val_m)
            semantic_rays.append(prediction_class)

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
    alpha in [0..1], controlling how we mix the two errors.
    """
    depth_error = np.mean(np.abs(pred_depths - cand_depths))
    sem_error = np.mean((pred_semantics != cand_semantics).astype(float))
    score = alpha * depth_error + (1.0 - alpha) * sem_error
    return score

def angular_difference_deg(ang1_rad, ang2_rad):
    """
    Minimal angular difference in degrees between two angles in radians.
    """
    diff_deg = abs(math.degrees(ang1_rad) - math.degrees(ang2_rad)) % 360
    return min(diff_deg, 360 - diff_deg)

# =============================================================================
# Main evaluation pipeline supporting a single (room-aware) weight combination
# =============================================================================
def evaluate_room_aware_with_refine(config):
    """
    Main pipeline:
      1) Load data & models.
      2) For each sample, compute probability volumes.
      3) Combine volumes and run room-aware localization,
         compute baseline and refined errors.
      4) Compute recalls and save results in a combined table.
         Also, compute timing metrics for each section over all samples.
         The timing information along with run parameters is saved to a file.

    In candidate refinement, we parallelize over all (candidate, offset) combos
    to find the best offset for each candidate and then the best candidate overall.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # ---------- Load the dataset -----------
    with open(config.split_file, "r") as f:
        split = AttrDict(yaml.safe_load(f))

    scene_names = split.test[:config.num_of_scenes]
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
            room_data_dir=config.room_data_dir,
        )

    # ---------- Load models -----------
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

    semantic_net = None
    if not config.use_ground_truth_semantic or config.prediction_type in ["semantic", "combined"]:
        semantic_net = semantic_net_pl_maskformer_small_room_type.load_from_checkpoint(
            checkpoint_path=config.log_dir_semantic_and_room_aware,
            num_classes=config.num_classes,
            semantic_net_type=config.semantic_net_type,
            num_room_types=config.num_room_types,
        ).to(device)
        semantic_net.eval()

    # Create a directory for results
    results_type_dir = os.path.join(config.results_dir, config.prediction_type)
    if config.use_ground_truth_depth or config.use_ground_truth_semantic:
        results_type_dir = os.path.join(results_type_dir, "gt")
    os.makedirs(results_type_dir, exist_ok=True)

    # ---------- Load scene data -----------
    desdfs, semantics, maps, gt_poses, valid_scene_names, walls = load_scene_data(
        test_set, config.dataset_dir, config.desdf_path
    )

    # Dictionary to hold recall metrics per weight combination
    baseline_combined = {}
    refine_combined = {}

    # Loop over each weight combination in the configuration
    for depth_weight, semantic_weight in tqdm.tqdm(config.weight_combinations, desc="Weight combinations"):
        weight_key = f"{depth_weight}_{semantic_weight}"
        print(f"\nEvaluating weight combination: {weight_key}")

        baseline_trans_errors = []
        baseline_rot_errors = []
        refine_trans_errors = []
        refine_rot_errors = []

        # Lists to collect timing measurements for each section over all samples
        time_stats = {
            'depth': [],
            'semantic': [],
            'prob_vol': [],
            'combine': [],
            'finalize_room': [],
            'candidate_extract': [],
            'candidate_process_room': [],
            'sample_total': []
        }
        
        # Hyperparameters for candidate refinement
        top_k = config.top_k if hasattr(config, "top_k") else 2
        min_dist_m = config.min_dist_m if hasattr(config, "min_dist_m") else 0.05
        alpha_similarity = config.alpha_similarity if hasattr(config, "alpha_similarity") else 0.8
        resolution_m_per_pixel = config.resolution_m_per_pixel if hasattr(config, "resolution_m_per_pixel") else 0.1

        # The offsets you want to evaluate per candidate
        augmentation_offsets = {
            "0": 0,
            "5": np.deg2rad(5),
            "10": np.deg2rad(10),
            "-5": np.deg2rad(-5),
            "-10": np.deg2rad(-10),
        }

        # Process each sample in the test set
        for data_idx in tqdm.tqdm(range(len(test_set)), desc="Samples", leave=False):
            sample_start = time.perf_counter()
            data = test_set[data_idx]
            scene_idx = np.sum(data_idx >= np.array(test_set.scene_start_idx)) - 1
            scene = test_set.scene_names[scene_idx]
            if 'floor' in scene:
                pass
            else:
                scene_number = int(scene.split("_")[1])
                scene = f"scene_{scene_number}"

            if scene not in valid_scene_names:
                continue

            idx_within_scene = data_idx - test_set.scene_start_idx[scene_idx]
            ref_pose_map = gt_poses[scene][idx_within_scene * (config.L + 1) + config.L, :]

            # --- Get depth rays ---
            start_depth = time.perf_counter()
            if not config.use_ground_truth_depth:
                ref_img_torch = torch.tensor(data["ref_img"], device=device).unsqueeze(0)
                ref_mask_torch = torch.tensor(data["ref_mask"], device=device).unsqueeze(0)
                with torch.no_grad():
                    pred_depths, _, _ = depth_net.encoder(ref_img_torch, ref_mask_torch)
                pred_depths = pred_depths.squeeze(0).cpu().numpy()
                pred_rays_depth = get_ray_from_depth(pred_depths, V=config.V, F_W=config.F_W)
            else:
                pred_depths = data["ref_depth"]
                pred_rays_depth = get_ray_from_depth(pred_depths, V=config.V, F_W=config.F_W)
            end_depth = time.perf_counter()
            time_stats['depth'].append(end_depth - start_depth)

            # --- Get semantic rays ---
            start_semantic = time.perf_counter()
            if not config.use_ground_truth_semantic:
                ref_img_torch = torch.tensor(data["ref_img"], device=device).unsqueeze(0)
                ref_mask_torch = torch.tensor(data["ref_mask"], device=device).unsqueeze(0)
                with torch.no_grad():
                    ray_logits, room_logits, _ = semantic_net(ref_img_torch, ref_mask_torch)
                ray_prob = F.softmax(ray_logits, dim=-1)
                prob_squeezed = ray_prob.squeeze(dim=0)
                sampled_indices = torch.multinomial(prob_squeezed, num_samples=1, replacement=True)
                sampled_indices = sampled_indices.squeeze(dim=1)
                sampled_semantic_indices_np = sampled_indices.cpu().numpy()
                pred_rays_semantic = get_ray_from_semantics_v2(sampled_semantic_indices_np)
            else:
                sampled_semantic_indices_np = data["ref_semantics"]
                pred_rays_semantic = get_ray_from_semantics_v2(sampled_semantic_indices_np)
            end_semantic = time.perf_counter()
            time_stats['semantic'].append(end_semantic - start_semantic)

            # --- Compute probability volumes ---
            start_prob_vol = time.perf_counter()
            if not config.use_saved_prob_vol:
                prob_vol_pred_depth, _, _, _ = localize(
                    torch.tensor(desdfs[scene]["desdf"]),
                    torch.tensor(pred_rays_depth, device="cpu"),
                    return_np=False,
                )
                prob_vol_pred_semantic, _, _, _ = localize(
                    torch.tensor(semantics[scene]["desdf"]),
                    torch.tensor(pred_rays_semantic, device="cpu"),
                    return_np=False,
                )
            else:
                if config.use_ground_truth_depth:
                    prob_vol_pred_depth = data['prob_vol_depth_gt'].to(device)
                else:
                    prob_vol_pred_depth = data['prob_vol_depth'].to(device)
                if config.use_ground_truth_semantic:
                    prob_vol_pred_semantic = data['prob_vol_semantic_gt'].to(device)
                else:
                    prob_vol_pred_semantic = data['prob_vol_semantic'].to(device)
            end_prob_vol = time.perf_counter()
            time_stats['prob_vol'].append(end_prob_vol - start_prob_vol)

            # --- Combine volumes with current weight combination ---
            start_combine = time.perf_counter()
            combined_prob_vol = combine_prob_volumes(
                prob_vol_pred_depth,
                prob_vol_pred_semantic,
                depth_weight,
                semantic_weight
            )
            end_combine = time.perf_counter()
            time_stats['combine'].append(end_combine - start_combine)

            # --- Finalize localization (room aware) ---
            start_finalize_room = time.perf_counter()
            class_probs = torch.softmax(room_logits, dim=1)
            max_prob, room_idx = torch.max(class_probs, dim=1)
            room_polygons = []
            predicted_room = "None"
            if max_prob.item() > config.room_selection_threshold:
                id_to_room_type = {v: k for k, v in room_type_to_id.items()}
                predicted_room = id_to_room_type[room_idx.item()]
                room_polygons = data["room_polygons"].get(predicted_room, [])
            final_prob_vol, prob_dist_pred, orientation_map, pose_pred = finalize_localization(
                combined_prob_vol, data["room_polygons"], room_polygons
            )
            end_finalize_room = time.perf_counter()
            time_stats['finalize_room'].append(end_finalize_room - start_finalize_room)

            # --- Baseline error (room aware) ---
            pose_pred_np = np.array(pose_pred, dtype=np.float32)
            pose_pred_np[0:2] = pose_pred_np[0:2] / 10.0
            gt_x, gt_y, gt_o = ref_pose_map[:3]
            pred_x, pred_y, pred_o = pose_pred_np
            baseline_trans = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
            rot_diff_deg = abs(pred_o - gt_o) % (2 * math.pi)
            baseline_rot = min(rot_diff_deg, 2*math.pi - rot_diff_deg) / math.pi * 180.0
            
            baseline_trans_errors.append(baseline_trans)
            baseline_rot_errors.append(baseline_rot)

            # --- Candidate refinement ---
            # 1) We get top_k candidates
            start_candidate_extract = time.perf_counter()
            top_k_candidates = extract_top_k_locations(
                prob_dist_pred,
                orientation_map,
                K=top_k,
                min_dist_m=min_dist_m,
                resolution_m_per_pixel=resolution_m_per_pixel,
                num_orientations=36
            )
            end_candidate_extract = time.perf_counter()
            time_stats['candidate_extract'].append(end_candidate_extract - start_candidate_extract)

            # 2) We want to find best offset for each candidate, then the best candidate overall.
            #    We'll parallelize across all (candidate, offset) combos in one pool.

            def combo_worker(cand_idx, offset_key):
                """
                Worker that:
                  - obtains rays for the given candidate
                  - measure similarity for the candidate with that offset
                  - returns pick_score, offset_key
                """
                cand = top_k_candidates[cand_idx]
                cand_px = cand['x']
                cand_py = cand['y']
                cand_o = cand['orientation_radians']

                # We do compute_rays_from_candidate once *per candidate* in normal code,
                # but if top_k is small, it's not so expensive. For large top_k, you can
                # precompute and cache. We'll do it directly here for simplicity:
                cand_rays_dict = compute_rays_from_candidate(
                    walls[scene],
                    maps[scene],
                    cand,
                    augmentation_offsets
                )

                cand_depths, cand_sems = cand_rays_dict[offset_key]
                score = measure_similarity(
                    pred_depths,
                    sampled_semantic_indices_np,
                    cand_depths,
                    cand_sems,
                    alpha=alpha_similarity
                )
                return score, offset_key

            start_candidate_process_room = time.perf_counter()

            # We'll store best offset and best score for each candidate
            best_offsets = [("none", 1e9)] * len(top_k_candidates)  # (offset_key, score)

            # Create tasks for all combos: for c in top_k, for off in offsets
            tasks = []
            for cand_idx in range(len(top_k_candidates)):
                for offset_key in augmentation_offsets:
                    tasks.append((cand_idx, offset_key))

            # We'll do multi-threading, but limit threads if top_k * offsets is large
            max_workers = 1
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {}
                # Submit each (cand_idx, offset_key) as a separate job
                for cand_idx, offset_key in tasks:
                    future = executor.submit(combo_worker, cand_idx, offset_key)
                    future_map[future] = cand_idx

                # Gather results
                for future in as_completed(future_map):
                    cand_idx = future_map[future]
                    pick_score, offset_key = future.result()
                    if pick_score < best_offsets[cand_idx][1]:
                        best_offsets[cand_idx] = (offset_key, pick_score)

            # Now we have best offset, best score for each candidate
            # Next step: pick best candidate among them
            best_candidate_idx = -1
            best_candidate_score = 1e9
            for i, (off_k, sc) in enumerate(best_offsets):
                if sc < best_candidate_score:
                    best_candidate_idx = i
                    best_candidate_score = sc

            # If best_candidate_idx is not -1, finalize
            best_candidate_location = None
            best_candidate_orientation = None
            if best_candidate_idx != -1:
                # Recompute final orientation for that candidate with that offset
                cand = top_k_candidates[best_candidate_idx]
                cand_px = cand['x']
                cand_py = cand['y']
                cand_o = cand['orientation_radians']
                offset_key = best_offsets[best_candidate_idx][0]
                offset_val = augmentation_offsets[offset_key]
                refined_o = cand_o + np.deg2rad(int(math.degrees(offset_val)))
                # Or we could just do (cand_o + offset_val)
                # but the original code did int(best_offset)...

                best_candidate_location = (cand_px / 10.0, cand_py / 10.0)
                best_candidate_orientation = refined_o

            end_candidate_process_room = time.perf_counter()
            time_stats['candidate_process_room'].append(end_candidate_process_room - start_candidate_process_room)

            if best_candidate_location is not None:
                refine_x, refine_y = best_candidate_location
                refine_o = best_candidate_orientation
                refine_trans = np.sqrt((refine_x - gt_x)**2 + (refine_y - gt_y)**2)
                refine_rot = angular_difference_deg(refine_o, gt_o)
                refine_trans_errors.append(refine_trans)
                refine_rot_errors.append(refine_rot)

            sample_end = time.perf_counter()
            time_stats['sample_total'].append(sample_end - sample_start)

        # After processing all samples, compute recalls
        baseline_trans_errors = np.array(baseline_trans_errors)
        baseline_rot_errors = np.array(baseline_rot_errors)
        refine_trans_errors = np.array(refine_trans_errors)
        refine_rot_errors = np.array(refine_rot_errors)

        baseline_recalls = calculate_recalls(baseline_trans_errors, baseline_rot_errors)
        refine_recalls = calculate_recalls(refine_trans_errors, refine_rot_errors)

        # Save individual JSON files per weight combination (room-aware only)
        with open(os.path.join(results_type_dir, f"baseline_recalls_{weight_key}.json"), "w") as f:
            json.dump(baseline_recalls, f, indent=4)
        with open(os.path.join(results_type_dir, f"refine_recalls_{weight_key}.json"), "w") as f:
            json.dump(refine_recalls, f, indent=4)

        # Store the recalls for summary
        baseline_combined[weight_key] = baseline_recalls
        refine_combined[weight_key] = refine_recalls

        print(f"Weight {weight_key} recalls:")
        print("  baseline:", baseline_recalls)
        print("  refine:", refine_recalls)

        # Create directories for summary tables (room-aware only)
        baseline_dir = os.path.join(results_type_dir, "summary_baseline")
        refine_dir = os.path.join(results_type_dir, "summary_refine")
        os.makedirs(baseline_dir, exist_ok=True)
        os.makedirs(refine_dir, exist_ok=True)

        # Create and save the summary tables
        create_combined_results_table(baseline_combined, baseline_dir)
        create_combined_results_table(refine_combined, refine_dir)
        print("All combined summary tables created.")

        # Print timing analysis (mean and standard deviation for each section)
        timing_summary = {}
        for section, times in time_stats.items():
            if times:
                mean_time = np.mean(times)
                std_time = np.std(times)
                timing_summary[section] = {"mean": mean_time, "std": std_time}
                print(f"Timing for {section}: mean = {mean_time:.6f}s, std = {std_time:.6f}s")
            else:
                timing_summary[section] = {"mean": None, "std": None}
                print(f"Timing for {section}: No measurements.")

        # --- Aggregated timing metrics ---
        prediction_times = np.array(time_stats['depth']) + np.array(time_stats['semantic'])
        localization_times = np.array(time_stats['prob_vol']) + np.array(time_stats['combine']) + np.array(time_stats['finalize_room'])
        refinement_times = np.array(time_stats['candidate_extract']) + np.array(time_stats['candidate_process_room'])

        aggregated_metrics = {
            "prediction": {
                "mean": float(np.mean(prediction_times)),
                "std": float(np.std(prediction_times))
            },
            "localization": {
                "mean": float(np.mean(localization_times)),
                "std": float(np.std(localization_times))
            },
            "refinement": {
                "mean": float(np.mean(refinement_times)),
                "std": float(np.std(refinement_times))
            },
            "sample_total": timing_summary['sample_total']
        }

        print("\nAggregated Timing Metrics:")
        print(f"Prediction: mean = {aggregated_metrics['prediction']['mean']:.6f}s, std = {aggregated_metrics['prediction']['std']:.6f}s")
        print(f"Localization: mean = {aggregated_metrics['localization']['mean']:.6f}s, std = {aggregated_metrics['localization']['std']:.6f}s")
        print(f"Refinement: mean = {aggregated_metrics['refinement']['mean']:.6f}s, std = {aggregated_metrics['refinement']['std']:.6f}s")
        print(f"Sample Total: mean = {aggregated_metrics['sample_total']['mean']:.6f}s, std = {aggregated_metrics['sample_total']['std']:.6f}s")

        # Save timing stats along with run parameters to a file
        run_params = {
            "top_k": top_k,
            "min_dist_m": min_dist_m,
            "alpha_similarity": alpha_similarity,
            "resolution_m_per_pixel": resolution_m_per_pixel,
            "num_samples": len(test_set)
        }
        timing_info = {
            "weight_combination": weight_key,
            "parameters": run_params,
            "timing_summary": timing_summary,
            "aggregated_timing": aggregated_metrics
        }
        timing_file = os.path.join(results_type_dir, f"timing_stats_{weight_key}_top_k_{top_k}.json")
        with open(timing_file, "w") as f:
            json.dump(timing_info, f, indent=4)
        print(f"Timing statistics saved to: {timing_file}")

def main():
    parser = argparse.ArgumentParser(description="Observation evaluation with room-aware timing analysis.")
    parser.add_argument(
        "--config_file",
        type=str,
        default="evaluation/configuration/S3D/config_eval_time.yaml",
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    config = AttrDict(config_dict)

    evaluate_room_aware_with_refine(config)

if __name__ == "__main__":
    main()
