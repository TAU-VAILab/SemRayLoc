import os
import argparse
import math
import yaml
import json
import numpy as np
import torch
import torch.nn.functional as F
from attrdict import AttrDict
from tqdm import tqdm

# == Local modules (adapt paths as needed) ==
from modules.mono.depth_net_pl import depth_net_pl
from modules.semantic.semantic_net_pl_maskformer_small_room_type import (
    semantic_net_pl_maskformer_small_room_type
)
from modules.semantic.semantic_mapper import room_type_to_id

# from data_utils.data_utils_for_laser_train import GridSeqDataset
from data_utils.data_utils import GridSeqDataset
from data_utils.prob_vol_data_utils import ProbVolDataset

from utils.data_loader_helper import load_scene_data
from utils.localization_utils import (
    get_ray_from_depth,
    get_ray_from_semantics_v2,
    localize,
    finalize_localization,
)
from utils.raycast_utils import ray_cast  # If you have your own function
# We won't use angle augmentations or refine here, per your request.


# ------------------------------------------------------------------------
# Utility: extract top K from a 2D distribution
# ------------------------------------------------------------------------
def indices_to_radians(orientation_idx: int, num_orientations: int = 36) -> float:
    """Convert orientation index [0..num_orientations-1] to [0, 2π)."""
    return orientation_idx / num_orientations * 2.0 * math.pi


def extract_top_k_locations(
    prob_dist: np.ndarray,
    orientation_map: np.ndarray,
    K: int,
    min_dist_m: float,
    resolution_m_per_pixel: float = 0.1,
    num_orientations: int = 36,
):
    """
    From a 2D probability map (H, W), pick up to the top-K (x,y,orientation,prob_value),
    enforcing that any newly picked candidate is at least 'min_dist_m' from previous picks.
    """
    H, W = prob_dist.shape
    prob_torch = torch.from_numpy(prob_dist).float()
    orient_torch = torch.from_numpy(orientation_map)

    flat_probs = prob_torch.view(-1)  # shape [H*W]
    flat_orient = orient_torch.view(-1)  # shape [H*W]

    sorted_indices = torch.argsort(flat_probs, descending=True)
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

        pick_prob = float(flat_probs[idx].item())
        pick_orient_idx = int(flat_orient[idx].item())
        pick_orient_rad = indices_to_radians(pick_orient_idx, num_orientations)

        picks.append({
            "x": float(x.item()),
            "y": float(y.item()),
            "orientation_radians": pick_orient_rad,
            "prob_value": pick_prob
        })

        # Suppress neighbors within min_dist_pixels
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
    Combine 2 volumes [H,W,O] with given weights, slicing to the min shared shape.
    Output is also [H', W', O'].
    """
    H = min(prob_vol_depth.shape[0], prob_vol_semantic.shape[0])
    W = min(prob_vol_depth.shape[1], prob_vol_semantic.shape[1])
    O = min(prob_vol_depth.shape[2], prob_vol_semantic.shape[2])
    vol_depth_sliced = prob_vol_depth[:H, :W, :O]
    vol_sem_sliced = prob_vol_semantic[:H, :W, :O]
    return depth_weight * vol_depth_sliced + semantic_weight * vol_sem_sliced


# ------------------------------------------------------------------------
# Utility: checking thresholds
# ------------------------------------------------------------------------
def is_within_threshold(
    cand_x_m: float,
    cand_y_m: float,
    cand_o_rad: float,
    gt_x_m: float,
    gt_y_m: float,
    gt_o_rad: float,
    trans_thresh: float,
    rot_thresh_deg: float,
) -> bool:
    """
    For the first three metrics (0.1, 0.5, 1.0), rot_thresh_deg = 0
      => we skip rotation check, only check distance
    For the last metric (1.0m + 30 deg), rot_thresh_deg=30 => check both distance & orientation
    """
    # Check translation first
    dist = math.sqrt((cand_x_m - gt_x_m)**2 + (cand_y_m - gt_y_m)**2)
    if dist > trans_thresh:
        return False
    # If rotation threshold is 0 => skip orientation check
    if rot_thresh_deg <= 0:
        return True

    # Check rotation
    gt_deg = math.degrees(gt_o_rad)
    cand_deg = math.degrees(cand_o_rad)
    diff_deg = abs(cand_deg - gt_deg) % 360
    rot_err = min(diff_deg, 360 - diff_deg)
    return (rot_err <= rot_thresh_deg)


# ------------------------------------------------------------------------
# Main: Evaluate top-K for each delta (0.05, 0.5, 1.0)
#       Then measure recall for top-K=1,2,3,5 under 4 metrics:
#         - 0.1m
#         - 0.5m
#         - 1.0m
#         - 1.0m + 30 deg
# ------------------------------------------------------------------------
def evaluate_topk_deltas(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 1) Load split
    with open(config.split_file, "r") as f:
        split_data = yaml.safe_load(f)
    split_data = AttrDict(split_data)
    scene_names = split_data.test[: config.num_of_scenes]

    # 2) Create dataset
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

    # 3) Load models (if needed)
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

    semantic_net = None
    if (not config.use_ground_truth_semantic) or (config.prediction_type in ["semantic", "combined"]):
        semantic_net = semantic_net_pl_maskformer_small_room_type.load_from_checkpoint(
            checkpoint_path=config.log_dir_semantic_and_room_aware,
            num_classes=config.num_classes,
            semantic_net_type=config.semantic_net_type,
            num_room_types=config.num_room_types,
        ).to(device)
        semantic_net.eval()

    # 4) Results directory
    results_dir = os.path.join(config.results_dir, config.prediction_type)
    if config.use_ground_truth_depth or config.use_ground_truth_semantic:
        results_dir = os.path.join(results_dir, "gt")
    os.makedirs(results_dir, exist_ok=True)

    # 5) Load scene data
    desdfs, semantics, maps, gt_poses, valid_scene_names, walls = load_scene_data(
        test_set, config.dataset_dir, config.desdf_path
    )

    # 6) We'll do 3 deltas, up to top-K=5
    # DELTA_LIST = [0.05, 0.5, 1.0]
    DELTA_LIST = [0.05,0.5, 1]
    K_LEVELS = [1, 2, 3, 5]

    # Our 4 metrics:
    #   (trans_thresh, rot_thresh_deg)
    #   The "no rotation" ones have rot_thresh_deg=0
    METRICS = [
        (0.1, 0),     # 0.1m
        (0.5, 0),     # 0.5m
        (1.0, 0),     # 1.0m
        (1.0, 30),    # 1.0m + 30 deg
    ]

    # We store final results in big dictionaries for each weight combination
    #   final_room_aware[weight_key] = {
    #       delta: {
    #         "0.1m_noRot": {1: recall, 2: recall, 3: recall, 5: recall},
    #         "0.5m_noRot": {...},
    #         "1.0m_noRot": {...},
    #         "1.0m_30deg": {...}
    #       },
    #       next_delta: {...}
    #   }
    # same for final_no_room
    final_room_aware = {}
    final_no_room = {}

    # 7) Loop over each weight combo
    for depth_weight, semantic_weight in tqdm(config.weight_combinations, desc="Weight combinations"):
        wkey = f"{depth_weight}_{semantic_weight}"
        print(f"\nEvaluating weight combination: {wkey}")

        # For counting successes, we do a structure:
        #   success_counters_room[delta][(trans,rot)][k] = int
        #   success_counters_no_room[delta][(trans,rot)][k] = int
        success_counters_room = {}
        success_counters_no_room = {}
        for dlt in DELTA_LIST:
            success_counters_room[dlt] = {}
            success_counters_no_room[dlt] = {}
            for (t_thr, r_thr) in METRICS:
                success_counters_room[dlt][(t_thr, r_thr)] = {k: 0 for k in K_LEVELS}
                success_counters_no_room[dlt][(t_thr, r_thr)] = {k: 0 for k in K_LEVELS}

        total_valid_samples = 0

        # 8) Go through each sample
        for data_idx in tqdm(range(len(test_set)), desc="Samples", leave=False):
            data = test_set[data_idx]
            scene_idx = np.sum(data_idx >= np.array(test_set.scene_start_idx)) - 1
            scene_name = test_set.scene_names[scene_idx]

            # Quick fix to ensure it's a valid scene
            if 'floor' not in scene_name:
                try:
                    # e.g. "scene_14"
                    scene_number = int(scene_name.split("_")[1])
                    scene_name = f"scene_{scene_number}"
                except:
                    pass

            if scene_name not in valid_scene_names:
                continue
            total_valid_samples += 1

            idx_within_scene = data_idx - test_set.scene_start_idx[scene_idx]
            # e.g. if L=1, the reference pose is at index = idx_within_scene*(L+1) + L
            ref_pose_map = gt_poses[scene_name][idx_within_scene * (config.L + 1) + config.L, :]
            gt_x_m, gt_y_m, gt_o_rad = ref_pose_map[0], ref_pose_map[1], ref_pose_map[2]

            # --------------------------------------------------------------
            # (a) Prepare depth rays
            # --------------------------------------------------------------
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

            # --------------------------------------------------------------
            # (b) Prepare semantic rays
            # --------------------------------------------------------------
            if not config.use_ground_truth_semantic:
                ref_img_torch = torch.tensor(data["ref_img"], device=device).unsqueeze(0)
                ref_mask_torch = torch.tensor(data["ref_mask"], device=device).unsqueeze(0)
                with torch.no_grad():
                    ray_logits, room_logits, _ = semantic_net(ref_img_torch, ref_mask_torch)
                ray_prob = F.softmax(ray_logits, dim=-1)
                sampled_indices = torch.multinomial(ray_prob.squeeze(0), num_samples=1, replacement=True)
                sampled_indices = sampled_indices.squeeze(dim=1)
                sampled_semantic_indices_np = sampled_indices.cpu().numpy()
                pred_rays_semantic = get_ray_from_semantics_v2(sampled_semantic_indices_np)
            else:
                sampled_semantic_indices_np = data["ref_semantics"]
                pred_rays_semantic = get_ray_from_semantics_v2(sampled_semantic_indices_np)

            # --------------------------------------------------------------
            # (c) Localize => probability volumes
            # --------------------------------------------------------------
            if not config.use_saved_prob_vol:
                # Depth volume
                prob_vol_pred_depth, _, _, _ = localize(
                    torch.tensor(desdfs[scene_name]["desdf"]),
                    torch.tensor(pred_rays_depth),
                    return_np=False,
                )
                # Semantic volume
                prob_vol_pred_semantic, _, _, _ = localize(
                    torch.tensor(semantics[scene_name]["desdf"]),
                    torch.tensor(pred_rays_semantic),
                    return_np=False,
                )
            else:
                # Precomputed volumes
                if config.use_ground_truth_depth:
                    prob_vol_pred_depth = data['prob_vol_depth_gt'].to(device)
                else:
                    prob_vol_pred_depth = data['prob_vol_depth'].to(device)
                if config.use_ground_truth_semantic:
                    prob_vol_pred_semantic = data['prob_vol_semantic_gt'].to(device)
                else:
                    prob_vol_pred_semantic = data['prob_vol_semantic'].to(device)

            # --------------------------------------------------------------
            # (d) Combine volumes
            # --------------------------------------------------------------
            combined_prob_vol = combine_prob_volumes(
                prob_vol_pred_depth, prob_vol_pred_semantic,
                depth_weight, semantic_weight
            )

            # --------------------------------------------------------------
            # (e) Room‐aware / Non‐room‐aware final 2D maps
            # --------------------------------------------------------------
            # Room‐aware
            if config.use_room_aware and ('room_logits' in locals() or 'room_logits' in globals()):
                # we only do polygons if we have them
                class_probs = torch.softmax(room_logits, dim=1)
                max_prob, room_idx = torch.max(class_probs, dim=1)
                predicted_room = None
                user_polygons = []
                if max_prob.item() > config.room_selection_threshold:
                    id2room = {v: k for k, v in room_type_to_id.items()}
                    predicted_room = id2room.get(room_idx.item(), None)
                    if predicted_room is not None:
                        user_polygons = data["room_polygons"].get(predicted_room, [])
                _, prob_dist_pred, orientation_map, _ = finalize_localization(
                    combined_prob_vol, data["room_polygons"], user_polygons
                )
            else:
                _, prob_dist_pred, orientation_map, _ = finalize_localization(
                    combined_prob_vol, data["room_polygons"]
                )

            # Non‐room‐aware
            _, prob_dist_pred_n, orientation_map_n, _ = finalize_localization(
                combined_prob_vol, data["room_polygons"]
            )

            if isinstance(prob_dist_pred, torch.Tensor):
                prob_dist_room = prob_dist_pred.cpu().numpy()
                orient_room = orientation_map.cpu().numpy()
            else:
                prob_dist_room = prob_dist_pred
                orient_room = orientation_map

            if isinstance(prob_dist_pred_n, torch.Tensor):
                prob_dist_no_room = prob_dist_pred_n.cpu().numpy()
                orient_no_room = orientation_map_n.cpu().numpy()
            else:
                prob_dist_no_room = prob_dist_pred_n
                orient_no_room = orientation_map_n

            # --------------------------------------------------------------
            # (f) For each delta in [0.05, 0.5, 1.0], extract up to top-5
            # --------------------------------------------------------------
            for dlt in DELTA_LIST:
                picks_room = extract_top_k_locations(
                    prob_dist_room, orient_room,
                    K=5,
                    min_dist_m=dlt,
                    resolution_m_per_pixel=0.1,
                    num_orientations=36
                )
                picks_no = extract_top_k_locations(
                    prob_dist_no_room, orient_no_room,
                    K=5,
                    min_dist_m=dlt,
                    resolution_m_per_pixel=0.1,
                    num_orientations=36
                )

                # ----------------------------------------------------------
                # (g) For each metric, for top-K in [1,2,3,5], check success
                # ----------------------------------------------------------
                for (trans_thr, rot_thr) in METRICS:
                    # For room:
                    for K_val in K_LEVELS:
                        # if K_val > len(picks), no success
                        if K_val <= len(picks_room):
                            # is there at least 1 that meets threshold?
                            success = False
                            subset = picks_room[:K_val]
                            for cand in subset:
                                cx_m = cand["x"] / 10.0
                                cy_m = cand["y"] / 10.0
                                co_rad = cand["orientation_radians"]
                                if is_within_threshold(
                                    cx_m, cy_m, co_rad,
                                    gt_x_m, gt_y_m, gt_o_rad,
                                    trans_thr, rot_thr
                                ):
                                    success = True
                                    break
                            if success:
                                success_counters_room[dlt][(trans_thr, rot_thr)][K_val] += 1

                    # For non‐room:
                    for K_val in K_LEVELS:
                        if K_val <= len(picks_no):
                            success = False
                            subset = picks_no[:K_val]
                            for cand in subset:
                                cx_m = cand["x"] / 10.0
                                cy_m = cand["y"] / 10.0
                                co_rad = cand["orientation_radians"]
                                if is_within_threshold(
                                    cx_m, cy_m, co_rad,
                                    gt_x_m, gt_y_m, gt_o_rad,
                                    trans_thr, rot_thr
                                ):
                                    success = True
                                    break
                            if success:
                                success_counters_no_room[dlt][(trans_thr, rot_thr)][K_val] += 1

        # --------------------------------------------------------------
        # (h) Convert success counts -> recall
        # --------------------------------------------------------------
        # final_room_out: dict of { delta -> { "0.1m_noRot": {top1: x, ...}, ... } }
        room_results = {}
        no_room_results = {}

        for dlt in DELTA_LIST:
            room_results[str(dlt)] = {}
            no_room_results[str(dlt)] = {}

            for (t_thr, r_thr) in METRICS:
                # We'll build a user‐friendly key
                if t_thr == 1.0 and r_thr == 30:
                    key_str = "1.0m_30deg"
                elif t_thr == 0.1:
                    key_str = "0.1m_noRot"
                elif t_thr == 0.5:
                    key_str = "0.5m_noRot"
                elif t_thr == 1.0 and r_thr == 0:
                    key_str = "1.0m_noRot"
                else:
                    key_str = f"{t_thr}m_{r_thr}deg"  # fallback

                room_results[str(dlt)][key_str] = {}
                no_room_results[str(dlt)][key_str] = {}

                for K_val in K_LEVELS:
                    cnt_room = success_counters_room[dlt][(t_thr, r_thr)][K_val]
                    cnt_no = success_counters_no_room[dlt][(t_thr, r_thr)][K_val]
                    if total_valid_samples > 0:
                        rec_room = cnt_room / total_valid_samples
                        rec_no = cnt_no / total_valid_samples
                    else:
                        rec_room = 0.0
                        rec_no = 0.0
                    room_results[str(dlt)][key_str][f"top{K_val}"] = rec_room
                    no_room_results[str(dlt)][key_str][f"top{K_val}"] = rec_no

        # Store in final structures
        final_room_aware[wkey] = room_results
        final_no_room[wkey] = no_room_results

        # Also write out to JSON per weight
        with open(os.path.join(results_dir, f"topk_room_aware_{wkey}.json"), "w") as f:
            json.dump(room_results, f, indent=4)
        with open(os.path.join(results_dir, f"topk_no_room_{wkey}.json"), "w") as f:
            json.dump(no_room_results, f, indent=4)

        print(f"Weight combo {wkey} => wrote JSON in {results_dir}")

    # ----------------------------------------------------------
    # Optionally: also store an aggregated JSON for all weights
    # ----------------------------------------------------------
    summary_dir = os.path.join(results_dir, "topk_summary")
    os.makedirs(summary_dir, exist_ok=True)
    with open(os.path.join(summary_dir, "room_aware_all_weights.json"), "w") as f:
        json.dump(final_room_aware, f, indent=4)
    with open(os.path.join(summary_dir, "no_room_all_weights.json"), "w") as f:
        json.dump(final_no_room, f, indent=4)

    print("Done. Aggregated results saved to 'topk_summary' folder.")


def main():
    parser = argparse.ArgumentParser(description="Compute top‐K recall at different delta_res for 4 metrics.")
    parser.add_argument(
        "--config_file",
        type=str,
        default="evaluation/configuration/S3D/config_eval.yaml",
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    config = AttrDict(config_dict)

    evaluate_topk_deltas(config)


if __name__ == "__main__":
    main()
