import os
import argparse
import yaml
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from attrdict import AttrDict
import matplotlib.pyplot as plt
import tqdm

# If you use Pillow or other libraries in your pipeline, keep them:
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# == Local modules (adapt as needed) ==
# Example placeholders – replace with your actual imports
# from modules.mono.depth_net_pl import depth_net_pl
# from modules.semantic.semantic_net_pl_maskformer_small_room_type import semantic_net_pl_maskformer_small_room_type
# from modules.semantic.semantic_mapper import room_type_to_id, zind_room_type_to_id
# from data_utils.data_utils import GridSeqDataset
# from data_utils.prob_vol_data_utils import ProbVolDataset
# from utils.data_loader_helper import load_scene_data
# from utils.localization_utils import localize, finalize_localization, get_ray_from_depth, get_ray_from_semantics_v2
# from utils.result_utils import calculate_recalls, create_combined_results_table
# from utils.raycast_utils import ray_cast
# from utils.visualization_utils import plot_prob_dist

# -------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------
def indices_to_radians(orientation_idx: int, num_orientations: int = 36) -> float:
    """Convert an orientation index to radians."""
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
    From a 2D probability map (H, W), pick the top-K ensuring no two picks are 
    within `min_dist_m` in real-world space.
    Returns a list of dicts with (x, y, orientation_radians, prob_value).
    """
    H, W = prob_dist.shape
    prob_dist_torch = torch.from_numpy(prob_dist)
    flat_prob = prob_dist_torch.view(-1)
    orientation_map_torch = torch.from_numpy(orientation_map)
    flat_orient = orientation_map_torch.view(-1)

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
import contextlib  # For redirect_stdout

def combine_prob_volumes(prob_vol_depth: torch.Tensor,
                         prob_vol_semantic: torch.Tensor,
                         depth_weight: float,
                         semantic_weight: float) -> torch.Tensor:
    """
    Combine two probability volumes [H, W, O] with given weights.
    We'll slice them to the min shared shape.
    Returns: [H', W', O'].
    """
    H = min(prob_vol_depth.shape[0], prob_vol_semantic.shape[0])
    W = min(prob_vol_depth.shape[1], prob_vol_semantic.shape[1])
    O = min(prob_vol_depth.shape[2], prob_vol_semantic.shape[2])

    depth_sliced = prob_vol_depth[:H, :W, :O]
    semantic_sliced = prob_vol_semantic[:H, :W, :O]
    return depth_weight * depth_sliced + semantic_weight * semantic_sliced

def angular_difference_deg(ang1_rad, ang2_rad):
    """Minimal angular difference in degrees between two angles in radians."""
    diff_deg = abs(math.degrees(ang1_rad) - math.degrees(ang2_rad)) % 360
    return min(diff_deg, 360 - diff_deg)

# -------------------------------------------------------------------------
# The main localization refinement function (adapt your local modules)
# -------------------------------------------------------------------------
def evaluate_room_aware_with_refine_single(config, penalty_matrix, experiment_tag):
    """
    Evaluate across the dataset for a SINGLE combination of:
      - penalty_matrix
      - config.min_dist_m
      - config.alpha_similarity
      - config.top_k
    (Plus the normal depth/semantic weighting from config.weight_combinations.)

    It will create a subdirectory with the experiment_tag for saving results and 
    will print the skip-cand0 statistics.
    """
    # Create subdir for the experiment
    results_type_dir = os.path.join(config.results_dir, f"exp_{experiment_tag}")
    os.makedirs(results_type_dir, exist_ok=True)
    log_path = os.path.join(results_type_dir, "experiment_logs.txt")
    with open(log_path, "w") as logfile, contextlib.redirect_stdout(logfile):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n=== Running single evaluation for: {experiment_tag} ===")
        print(f"   penalty_matrix:\n{penalty_matrix}")
        print(f"   min_dist_m: {config.min_dist_m}")
        print(f"   alpha_similarity: {config.alpha_similarity}")
        print(f"   top_k: {config.top_k}")


        # ---------------------------------------------------------------------
        # Load dataset
        #   Replace with your actual dataset instantiation (GridSeqDataset or ProbVolDataset)
        # ---------------------------------------------------------------------
        from data_utils.data_utils import GridSeqDataset
        from data_utils.prob_vol_data_utils import ProbVolDataset
        with open(config.split_file, "r") as f:
            split = AttrDict(yaml.safe_load(f))
        scene_names = split.val[:config.num_of_scenes]
        # scene_names = split.test[:config.num_of_scenes]
        if config.use_saved_prob_vol:
            test_set = ProbVolDataset(
                dataset_dir=config.dataset_dir,
                scene_names=scene_names,  # or parse from your split
                L=config.L,
                prob_vol_path=config.prob_vol_path,
                acc_only=False,
            )
        else:
            test_set = GridSeqDataset(
                dataset_dir=config.dataset_dir,
                scene_names=scene_names,  # or parse from your split
                L=config.L,
                room_data_dir=config.room_data_dir,  
            )

        # ---------------------------------------------------------------------
        # Load your local modules (depth_net, semantic_net, data, etc.)
        #   - Below is an example placeholder; adapt to your actual code
        # ---------------------------------------------------------------------
        from modules.mono.depth_net_pl import depth_net_pl
        from modules.semantic.semantic_net_pl_maskformer_small_room_type import semantic_net_pl_maskformer_small_room_type
        from modules.semantic.semantic_mapper import room_type_to_id
        from utils.data_loader_helper import load_scene_data
        from utils.localization_utils import localize, finalize_localization, get_ray_from_depth, get_ray_from_semantics_v2
        from utils.result_utils import calculate_recalls, create_combined_results_table
        from utils.raycast_utils import ray_cast

        # ---------- Load the dataset’s scene data for building maps, GT, etc. ----------
        desdfs, semantics, maps, gt_poses, valid_scene_names, walls = load_scene_data(
            test_set, config.dataset_dir, config.desdf_path
        )

        # ---------- Load your depth model if needed ----------
        depth_net = None
        if (not config.use_ground_truth_depth) or (config.prediction_type in ["depth", "combined"]):
            depth_net = depth_net_pl.load_from_checkpoint(
                checkpoint_path=config.log_dir_depth,
                d_min=config.d_min,
                d_max=config.d_max,
                d_hyp=config.d_hyp,
                D=config.D,
            ).to(device)
            depth_net.eval()

        # ---------- Load your semantic model if needed ----------
        semantic_net = None
        if (not config.use_ground_truth_semantic) or (config.prediction_type in ["semantic", "combined"]):
            semantic_net = semantic_net_pl_maskformer_small_room_type.load_from_checkpoint(
                checkpoint_path=config.log_dir_semantic_and_room_aware,
                num_classes=config.num_classes,
                semantic_net_type=config.semantic_net_type,
                num_room_types=config.num_room_types,
            ).to(device)
            semantic_net.eval()

        # ---------------------------------------------------------------------
        # Define our measure_similarity function for this experiment
        #  using the given penalty_matrix
        # ---------------------------------------------------------------------
        def measure_similarity(
            pred_depths: np.ndarray,
            pred_semantics: np.ndarray,
            cand_depths: np.ndarray,
            cand_semantics: np.ndarray,
            alpha: float = 0.5,
        ):
            """
            Depth difference (L1) + Semantic mismatch with penalty_matrix.

            We treat label 3 (wall) as 0, label 1 (door) as 1, label 2 (window) as 2.

            penalty_matrix is a 3x3 cost matrix, rows = GT, columns = predicted:
            e.g. penalty_matrix[gt_label, cand_label].
            """
            # 1) Depth error
            depth_error = np.mean(np.abs(pred_depths - cand_depths))

            # 2) Remap labels to 0=wall, 1=door, 2=window
            cand_semantics_int = np.array(cand_semantics, dtype=np.int32)
            pred_semantics_int = np.array(pred_semantics, dtype=np.int32)
            cand_semantics_int[cand_semantics_int == 3] = 0
            pred_semantics_int[pred_semantics_int == 3] = 0

            # 3) Penalty lookups
            sem_penalties = penalty_matrix[pred_semantics_int, cand_semantics_int]
            sem_error = np.mean(sem_penalties)

            # 4) Weighted sum
            score = alpha * depth_error + (1.0 - alpha) * sem_error
            return score

        # We’ll track the final aggregated results for each weight combination
        from collections import defaultdict
        baseline_combined = {}
        baseline_n_combined = {}
        refine_combined = {}
        refine_n_combined = {}

        # Also track skipping-cand0 statistics (room-aware vs. non-room-aware)
        skip_stats_room = {
            "skip_count": 0,
            "score_diffs": [],
            "trans_diffs": [],
        }
        skip_stats_no_room = {
            "skip_count": 0,
            "score_diffs": [],
            "trans_diffs": [],
        }

        # ---------------------------------------------------------------------
        # Start looping over weight combinations
        # ---------------------------------------------------------------------
        for depth_weight, semantic_weight in config.weight_combinations:
            weight_key = f"{depth_weight}_{semantic_weight}"
            print(f"\nEvaluating weight combination: {weight_key}")

            baseline_trans_errors = []
            baseline_rot_errors = []
            refine_trans_errors = []
            refine_rot_errors = []

            baseline_trans_errors_n = []
            baseline_rot_errors_n = []
            refine_trans_errors_n = []
            refine_rot_errors_n = []

            # Go over each sample in test_set
            for data_idx in tqdm.tqdm(range(len(test_set)), desc="Samples", leave=False):
                data = test_set[data_idx]

                # figure out scene name
                scene_idx = np.sum(data_idx >= np.array(test_set.scene_start_idx)) - 1
                scene = test_set.scene_names[scene_idx]
                if scene not in valid_scene_names:
                    continue

                idx_within_scene = data_idx - test_set.scene_start_idx[scene_idx]
                ref_pose_map = gt_poses[scene][idx_within_scene * (config.L + 1) + config.L, :]

                # 1) Predict depth rays
                if not config.use_ground_truth_depth:
                    ref_img_torch = torch.tensor(data["ref_img"], device=device).unsqueeze(0)
                    ref_mask_torch = torch.tensor(data["ref_mask"], device=device).unsqueeze(0)
                    with torch.no_grad():
                        pred_depths_tensor, _, _ = depth_net.encoder(ref_img_torch, ref_mask_torch)
                    pred_depths = pred_depths_tensor.squeeze(0).cpu().numpy()
                else:
                    pred_depths = data["ref_depth"]

                pred_rays_depth = get_ray_from_depth(pred_depths, V=config.V, F_W=config.F_W)

                # 2) Predict semantic rays
                if not config.use_ground_truth_semantic:
                    ref_img_torch = torch.tensor(data["ref_img"], device=device).unsqueeze(0)
                    ref_mask_torch = torch.tensor(data["ref_mask"], device=device).unsqueeze(0)
                    with torch.no_grad():
                        ray_logits, room_logits, _ = semantic_net(ref_img_torch, ref_mask_torch)
                    ray_prob = F.softmax(ray_logits, dim=-1).squeeze(0)  # shape: [V, #classes]
                    sampled_indices = torch.multinomial(ray_prob, num_samples=1, replacement=True).squeeze(dim=1)
                    sampled_semantic_indices_np = sampled_indices.cpu().numpy()
                else:
                    # ground-truth semantic in data
                    sampled_semantic_indices_np = data["ref_semantics"]

                pred_rays_semantic = get_ray_from_semantics_v2(sampled_semantic_indices_np)

                # 3) Probability volumes (depth, semantic)
                if not config.use_saved_prob_vol:
                    prob_vol_pred_depth, prob_vol_dist_pred_depth, _, _ = localize(
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
                    # Provided by ProbVolDataset
                    prob_vol_pred_depth = data["prob_vol_depth_gt" if config.use_ground_truth_depth else "prob_vol_depth"]
                    prob_vol_pred_semantic = data["prob_vol_semantic_gt" if config.use_ground_truth_semantic else "prob_vol_semantic"]

                # 4) Combine volumes
                combined_prob_vol = combine_prob_volumes(
                    prob_vol_pred_depth,
                    prob_vol_pred_semantic,
                    depth_weight,
                    semantic_weight
                )

                # 5) If you do "finalize_localization" with or without room polygons
                from utils.localization_utils import finalize_localization
                if config.use_room_aware:
                    class_probs = torch.softmax(room_logits, dim=1)
                    max_prob_val, room_idx = torch.max(class_probs, dim=1)

                    # Possibly retrieve the polygons for predicted room type
                    predicted_room = "None"
                    polygons_to_use = []
                    if max_prob_val.item() > config.room_selection_threshold:
                        id_to_room_type = {v: k for k, v in room_type_to_id.items()}
                        predicted_room = id_to_room_type[room_idx.item()]
                        polygons_to_use = data["room_polygons"].get(predicted_room, [])

                    final_prob_vol, prob_dist_pred, orientation_map, pose_pred = finalize_localization(
                        combined_prob_vol, [], polygons_to_use
                    )
                    final_prob_vol_n, prob_dist_pred_n, orientation_map_n, pose_pred_n = finalize_localization(
                        combined_prob_vol, data["room_polygons"]
                    )
                else:
                    final_prob_vol, prob_dist_pred, orientation_map, pose_pred = finalize_localization(
                        combined_prob_vol, data["room_polygons"]
                    )
                    final_prob_vol_n, prob_dist_pred_n, orientation_map_n, pose_pred_n = final_prob_vol, prob_dist_pred, orientation_map, pose_pred

                # --- Baseline (room aware) ---
                gt_x, gt_y, gt_o = ref_pose_map[:3]
                pose_pred_np = np.array(pose_pred, dtype=np.float32)
                pose_pred_np[0:2] = pose_pred_np[0:2] / 10.0
                pred_x, pred_y, pred_o = pose_pred_np
                baseline_trans = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
                rot_diff_deg = abs(pred_o - gt_o) % (2 * math.pi)
                baseline_rot = min(rot_diff_deg, 2*math.pi - rot_diff_deg) / math.pi * 180.0
                baseline_trans_errors.append(baseline_trans)
                baseline_rot_errors.append(baseline_rot)

                # --- Baseline (non-room-aware) ---
                pose_pred_np_n = np.array(pose_pred_n, dtype=np.float32)
                pose_pred_np_n[0:2] = pose_pred_np_n[0:2] / 10.0
                pred_x_n, pred_y_n, pred_o_n = pose_pred_np_n
                baseline_trans_n = np.sqrt((pred_x_n - gt_x)**2 + (pred_y_n - gt_y)**2)
                rot_diff_deg_n = abs(pred_o_n - gt_o) % (2 * math.pi)
                baseline_rot_n = min(rot_diff_deg_n, 2*math.pi - rot_diff_deg_n) / math.pi * 180.0
                baseline_trans_errors_n.append(baseline_trans_n)
                baseline_rot_errors_n.append(baseline_rot_n)

                # --- Candidate refinement step (room aware) ---
                top_k_candidates = extract_top_k_locations(
                    prob_dist_pred,
                    orientation_map,
                    K=config.top_k,
                    min_dist_m=config.min_dist_m,
                    resolution_m_per_pixel=config.resolution_m_per_pixel,
                    num_orientations=36
                )

                # We'll define a small function that: 
                #  1) for each candidate, tries multiple orientation offsets
                #  2) picks the offset with the best (lowest) measure_similarity
                #  3) returns the best offset score
                def compute_candidate_score_offset(cand):
                    # Let's do a few offsets in degrees:
                    offsets_deg = [0, 5 -5]
                    # You can expand or reduce the set as you wish
                    cand_best_score = 1e9
                    cand_best_offset = 0

                    # We need to do a raycast for each offset
                    # Precompute candidate position in pixel space:
                    base_x_px = cand["x"] * 10.0
                    base_y_px = cand["y"] * 10.0
                    base_orientation = cand["orientation_radians"]

                    # For generating multiple rays:
                    ray_n = 40
                    F_W_local = 1 / np.tan(0.698132) / 2
                    depth_max = 15

                    # Precompute baseline angles for the rays
                    center_angs = np.flip(
                        np.arctan2((np.arange(ray_n) - np.arange(ray_n).mean()), ray_n * F_W_local)
                    )

                    # We do not want to raycast each offset+angle for each cand repeatedly 
                    # if possible. But for clarity, we do so:
                    from utils.raycast_utils import ray_cast

                    for off_deg in offsets_deg:
                        off_rad = math.radians(off_deg)
                        effective_orientation = base_orientation + off_rad
                        ray_angles = center_angs + effective_orientation

                        cand_depths = []
                        cand_sems = []
                        for ang in ray_angles:
                            depth_val_cm, prediction_class, _, _ = ray_cast(
                                maps[scene], 
                                np.array([base_x_px, base_y_px]),
                                ang,
                                dist_max=depth_max*100,
                                min_dist=5,
                                cast_type=2
                            )
                            depth_val_m = depth_val_cm / 100.0
                            cand_depths.append(depth_val_m)
                            cand_sems.append(prediction_class)

                        # measure similarity
                        score = measure_similarity(
                            pred_depths,
                            sampled_semantic_indices_np,
                            np.array(cand_depths),
                            np.array(cand_sems),
                            alpha=config.alpha_similarity
                        )
                        if score < cand_best_score:
                            cand_best_score = score
                            cand_best_offset = off_rad

                    return cand_best_score, cand_best_offset

                # Evaluate all top_k candidates
                best_candidate_score = 1e9
                best_candidate_offset = 0.0
                best_candidate_index = 0
                best_candidate_xy = (0.0, 0.0)

                # Also compute the same for cand0 (the highest-prob candidate) so we can compare
                cand0 = top_k_candidates[0] if len(top_k_candidates) > 0 else None
                cand0_score = None
                cand0_trans_error = None

                for i, cand in enumerate(top_k_candidates):
                    cscore, coffset = compute_candidate_score_offset(cand)
                    if i == 0:
                        cand0_score = cscore  # for skip-cand0 stats

                    # Convert from pixel to meter
                    cand_x_m = cand["x"] / 10.0
                    cand_y_m = cand["y"] / 10.0
                    candidate_score = cscore
                    if candidate_score < best_candidate_score:
                        best_candidate_score = candidate_score
                        best_candidate_offset = coffset
                        best_candidate_index = i
                        best_candidate_xy = (cand_x_m, cand_y_m)

                # If we have a best candidate, refine
                if len(top_k_candidates) > 0:
                    refine_x, refine_y = best_candidate_xy
                    refine_o = top_k_candidates[best_candidate_index]["orientation_radians"] + best_candidate_offset
                    refine_trans = np.sqrt((refine_x - gt_x)**2 + (refine_y - gt_y)**2)
                    refine_rot = angular_difference_deg(refine_o, gt_o)
                    refine_trans_errors.append(refine_trans)
                    refine_rot_errors.append(refine_rot)

                    # If best candidate not cand0, update skip stats
                    if best_candidate_index != 0 and cand0_score is not None:
                        skip_stats_room["skip_count"] += 1
                        score_diff = best_candidate_score - cand0_score
                        skip_stats_room["score_diffs"].append(score_diff)

                        # Also compare trans error vs. cand0’s trans error
                        # We must compute cand0’s best offset for trans
                        if cand0 is not None:
                            # Re-run compute_candidate_score_offset for cand0 
                            # to get cand0 offset
                            c0_score, c0_off = compute_candidate_score_offset(cand0)
                            cand0_o = cand0["orientation_radians"] + c0_off
                            cand0_x_m = cand0["x"] / 10.0
                            cand0_y_m = cand0["y"] / 10.0
                            cand0_trans = np.sqrt((cand0_x_m - gt_x)**2 + (cand0_y_m - gt_y)**2)
                            skip_stats_room["trans_diffs"].append(refine_trans - cand0_trans)

                # --- Candidate refinement step (non-room aware) ---
                # top_k_candidates_n = extract_top_k_locations(
                #     prob_dist_pred_n,
                #     orientation_map_n,
                #     K=config.top_k,
                #     min_dist_m=config.min_dist_m,
                #     resolution_m_per_pixel=config.resolution_m_per_pixel,
                #     num_orientations=36
                # )

                # best_candidate_score_n = 1e9
                # best_candidate_offset_n = 0.0
                # best_candidate_index_n = 0
                # best_candidate_xy_n = (0.0, 0.0)

                # cand0_score_n = None
                # cand0_trans_error_n = None

                # for i, cand in enumerate(top_k_candidates_n):
                #     cscore, coffset = compute_candidate_score_offset(cand)
                #     if i == 0:
                #         cand0_score_n = cscore

                #     cand_x_m = cand["x"] / 10.0
                #     cand_y_m = cand["y"] / 10.0
                #     if cscore < best_candidate_score_n:
                #         best_candidate_score_n = cscore
                #         best_candidate_offset_n = coffset
                #         best_candidate_index_n = i
                #         best_candidate_xy_n = (cand_x_m, cand_y_m)

                # if len(top_k_candidates_n) > 0:
                #     refine_x_n, refine_y_n = best_candidate_xy_n
                #     refine_o_n = top_k_candidates_n[best_candidate_index_n]["orientation_radians"] + best_candidate_offset_n
                #     refine_trans_n = np.sqrt((refine_x_n - gt_x)**2 + (refine_y_n - gt_y)**2)
                #     refine_rot_n = angular_difference_deg(refine_o_n, gt_o)
                #     refine_trans_errors_n.append(refine_trans_n)
                #     refine_rot_errors_n.append(refine_rot_n)

                #     # skip-cand0 stats (no-room-aware)
                #     if best_candidate_index_n != 0 and cand0_score_n is not None:
                #         skip_stats_no_room["skip_count"] += 1
                #         score_diff_n = best_candidate_score_n - cand0_score_n
                #         skip_stats_no_room["score_diffs"].append(score_diff_n)

                #         # Compare trans error 
                #         cand0_n = top_k_candidates_n[0]
                #         c0_score_n, c0_off_n = compute_candidate_score_offset(cand0_n)
                #         cand0_o_n = cand0_n["orientation_radians"] + c0_off_n
                #         cand0_x_n_m = cand0_n["x"] / 10.0
                #         cand0_y_n_m = cand0_n["y"] / 10.0
                #         cand0_trans_n = np.sqrt((cand0_x_n_m - gt_x)**2 + (cand0_y_n_m - gt_y)**2)
                #         skip_stats_no_room["trans_diffs"].append(refine_trans_n - cand0_trans_n)

            # ------------------------------------------------------
            # Summaries (recalls) for the entire dataset
            # ------------------------------------------------------
            from utils.result_utils import calculate_recalls
            baseline_trans_errors = np.array(baseline_trans_errors)
            baseline_rot_errors = np.array(baseline_rot_errors)
            refine_trans_errors = np.array(refine_trans_errors)
            refine_rot_errors = np.array(refine_rot_errors)

            baseline_trans_errors_n = np.array(baseline_trans_errors_n)
            baseline_rot_errors_n = np.array(baseline_rot_errors_n)
            refine_trans_errors_n = np.array(refine_trans_errors_n)
            refine_rot_errors_n = np.array(refine_rot_errors_n)

            baseline_recalls = calculate_recalls(baseline_trans_errors, baseline_rot_errors)
            refine_recalls = calculate_recalls(refine_trans_errors, refine_rot_errors)
            baseline_recalls_n = calculate_recalls(baseline_trans_errors_n, baseline_rot_errors_n)
            refine_recalls_n = calculate_recalls(refine_trans_errors_n, refine_rot_errors_n)

            # Save JSONs
            with open(os.path.join(results_type_dir, f"baseline_recalls_{weight_key}.json"), "w") as f:
                json.dump(baseline_recalls, f, indent=4)
            with open(os.path.join(results_type_dir, f"baseline_recalls_n_{weight_key}.json"), "w") as f:
                json.dump(baseline_recalls_n, f, indent=4)
            with open(os.path.join(results_type_dir, f"refine_recalls_{weight_key}.json"), "w") as f:
                json.dump(refine_recalls, f, indent=4)
            with open(os.path.join(results_type_dir, f"refine_recalls_n_{weight_key}.json"), "w") as f:
                json.dump(refine_recalls_n, f, indent=4)

            print(f"[{experiment_tag} -> {weight_key}] baseline:", baseline_recalls)
            print(f"[{experiment_tag} -> {weight_key}] baseline_n:", baseline_recalls_n)
            print(f"[{experiment_tag} -> {weight_key}] refine:", refine_recalls)
            print(f"[{experiment_tag} -> {weight_key}] refine_n:", refine_recalls_n)

        # ------------------------------------------------------------
        # At the end, print skip-cand0 stats for this entire experiment
        # ------------------------------------------------------------
        def print_skip_stats(stats_dict, label):
            skip_count = stats_dict["skip_count"]
            if skip_count > 0:
                mean_score_diff = np.mean(stats_dict["score_diffs"])
                mean_trans_diff = np.mean(stats_dict["trans_diffs"])
                improved_count = np.sum([1 for d in stats_dict["trans_diffs"] if d < 0])
                worse_count = skip_count - improved_count
                print(f"\n  {label} skip-cand0 stats:")
                print(f"    skip_count = {skip_count}")
                print(f"    mean_score_diff = {mean_score_diff:.4f}")
                print(f"    mean_trans_diff = {mean_trans_diff:.4f}")
                print(f"    improved_count = {improved_count}, worse_count = {worse_count}")
            else:
                print(f"\n  {label} skip-cand0 stats: (No skip occurred)")

        print_skip_stats(skip_stats_room, "Room-aware")
        print_skip_stats(skip_stats_no_room, "Non-room-aware")

        # Optionally, you can create combined summary tables for each experiment.
        # For example, if you want them across weight combos, you can do:
        # from utils.result_utils import create_combined_results_table
        # create_combined_results_table(...)

        print(f"=== Finished single evaluation for: {experiment_tag} ===\n")


# -------------------------------------------------------------------------
# Main script that loops over multiple experiments
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="evaluation/configuration/S3D/config_experiments.yaml",
                        help="Path to the configuration file.")
    args = parser.parse_args()

    # 1) Load config
    with open(args.config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    config = AttrDict(config_dict)

    # By default, define your "prediction_type" if needed:
    #   e.g., config.prediction_type = "combined"
    config.prediction_type = "combined"

    # 2) Define the penalty matrices you want to test
    penalty_matrices = {
        "pm1": np.array([
            [0, 1, 1],  # GT wall(0)
            [2, 0, 0],  # GT door(1)
            [2, 0, 0],  # GT window(2)
        ]),
        "pm2": np.array([
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
        ]),
        "pm3": np.array([
            [0, 1, 1],
            [2, 0, 1],
            [5, 2, 0],
        ]),
        "pm4": np.array([
            [0, 1, 1],
            [5, 0, 2],
            [2, 1, 0],
        ]),
    }
    
    pm_map = {
        1.0: penalty_matrices["pm1"],
        2.0: penalty_matrices["pm2"],
        3.0: penalty_matrices["pm3"],
        4.0: penalty_matrices["pm4"],
    }
    # # 3) Define the experiment ranges for min_dist_m, alpha_similarity, and top_k
    # dist_values = [0.1, 0.5, 1.0]
    # alpha_values = [0.1, 0.3, 0.5]
    # top_k_values = [2, 3]

    # # 4) For each combination, override config fields and call the evaluation
    # for pm_key, pmatrix in penalty_matrices.items():
    #     for dist_m in dist_values:
    #         for alpha in alpha_values:
    #             for tk in top_k_values:
    #                 experiment_tag = f"{pm_key}_dist{dist_m}_alpha{alpha}_topK{tk}"
    #                 config.min_dist_m = dist_m
    #                 config.alpha_similarity = alpha
    #                 config.top_k = tk

    #                 # Evaluate
    #                 evaluate_room_aware_with_refine_single(config, pmatrix, experiment_tag)
    
    experiments = [
        (1.0, 0.1, 0.1, 2),
        (1.0, 0.1, 0.1, 3),
        (1.0, 0.1, 0.1, 5),
        (1.0, 0.1, 0.3, 2),
        (1.0, 0.1, 0.3, 3),
        (1.0, 0.1, 0.3, 5),        
        (1.0, 0.1, 0.5, 2),
        (1.0, 0.1, 0.5, 3),
        (1.0, 0.1, 0.5, 5),
        
        (1.0, 0.5, 0.1, 2),    
        (1.0, 0.5, 0.1, 3),
        (1.0, 0.5, 0.1, 5),
        (1.0, 0.5, 0.3, 2),
        (1.0, 0.5, 0.3, 3),
        (1.0, 0.5, 0.3, 5),        
        (1.0, 0.5, 0.5, 2),
        (1.0, 0.5, 0.5, 3),
        (1.0, 0.5, 0.5, 5),
        
        (1.0, 1.0, 0.1, 2),
        (1.0, 1.0, 0.1, 3),
        (1.0, 1.0, 0.1, 5),
        (1.0, 1.0, 0.3, 2),
        (1.0, 1.0, 0.3, 3),
        (1.0, 1.0, 0.3, 5),        
        (1.0, 1.0, 0.5, 2),
        (1.0, 1.0, 0.5, 3),
        (1.0, 1.0, 0.5, 5),
    ]

    # 5) Loop over only these experiments
    for (pm_value, dist, alpha, tk) in experiments:
        # Build an experiment tag
        experiment_tag = f"pm{pm_value}_dist{dist}_alpha{alpha}_topK{tk}"

        # Override config with the chosen parameters
        config.min_dist_m = dist
        config.alpha_similarity = alpha
        config.top_k = tk

        # The chosen penalty matrix from our map
        chosen_pm = pm_map[pm_value]

        # Evaluate
        evaluate_room_aware_with_refine_single(config, chosen_pm, experiment_tag)

    print("All experiments completed.")


if __name__ == "__main__":
    main()
