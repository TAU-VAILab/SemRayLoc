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
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm

# == Local modules (adapt as needed) ==
from modules.mono.depth_net_pl import depth_net_pl
# from modules.mono.depth_net_pl_adaptive import depth_net_pl_adaptive
from modules.semantic.semantic_net_pl_maskformer_small_room_type import semantic_net_pl_maskformer_small_room_type
from modules.semantic.semantic_mapper import room_type_to_id, zind_room_type_to_id

# from data_utils.data_utils_for_laser_train import GridSeqDataset
from data_utils.data_utils import GridSeqDataset
from data_utils.prob_vol_data_utils import ProbVolDataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Helper utilities
from utils.data_loader_helper import load_scene_data
from utils.localization_utils import (
    get_ray_from_depth,
    get_ray_from_semantics_v2,    
    localize,
    finalize_localization,
    finalize_localization_soft_room_threshold,
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
    Compute an error measure combining:
      - Depth difference (L1)
      - Semantic mismatch with customized penalties

    Original semantic labels:
      1: door
      2: window
      3: wall
    Remap the labels so that:
      wall (3) -> 0, door remains 1, window remains 2.
      
    The penalty matrix (rows = GT, columns = prediction)
    is defined using the new label indices:
      GT wall (0):   [wall:0, door:1, window:1]
      GT door (1):   [wall:3, door:0, window:2]
      GT window (2): [wall:5, door:3, window:0]

    alpha in [0, 1] controls the mix between depth and semantic errors.
    """
    import numpy as np

    # Compute depth error (L1 loss)
    depth_error = np.mean(np.abs(pred_depths - cand_depths))
    
    # Convert semantic lists to numpy arrays if needed
    cand_semantics = np.array(cand_semantics)
    pred_semantics = np.array(pred_semantics)
    
    # Ensure semantic arrays are integer type
    cand_semantics_int = cand_semantics.astype(np.int32)
    pred_semantics_int = pred_semantics.astype(np.int32)
    
    # Remap semantic labels: any value 3 (wall) becomes 0.
    cand_semantics_int[cand_semantics_int == 3] = 0
    pred_semantics_int[pred_semantics_int == 3] = 0
    
    # Define the penalty matrix for semantic mismatches.
    # New indices: 0: wall, 1: door, 2: window.
    # For ground truth wall (0): penalty when predicted as wall=0, door=1, window=1.
    # For ground truth door (1): penalty when predicted as door=0, window=2, wall=3.
    # For ground truth window (2): penalty when predicted as window=0, door=3, wall=5.
    penalty_matrix = np.array([
        [0, 1, 1],  # GT wall (0)
        [2, 0, 0],  # GT door (1)
        [2, 0, 0],  # GT window (2)
    ])
    
    # Look up the penalty for each element based on the remapped labels.
    sem_penalties = penalty_matrix[pred_semantics_int, cand_semantics_int]
    sem_error = np.mean(sem_penalties)
    
    # Combine depth and semantic errors
    score = alpha * depth_error + (1.0 - alpha) * sem_error
    return score


def angular_difference_deg(ang1_rad, ang2_rad):
    """
    Minimal angular difference in degrees between two angles in radians.
    """
    diff_deg = abs(math.degrees(ang1_rad) - math.degrees(ang2_rad)) % 360
    return min(diff_deg, 360 - diff_deg)

# =============================================================================
# Main evaluation pipeline supporting multiple weight combinations
# =============================================================================
def evaluate_room_aware_with_refine(config):
    """
    Main pipeline:
      1) Load data & models.
      2) For each sample, compute probability volumes.
      3) For each weight combination, combine volumes and run localization,
         compute baseline and refined errors.
      4) Compute recalls and save results in a combined table.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # ---------- Load the dataset -----------
    with open(config.split_file, "r") as f:
        split = AttrDict(yaml.safe_load(f))

    scene_names = split.test[:config.num_of_scenes]
    # scene_names = split.train[:2]
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

    # Dictionaries to hold recall metrics per weight combination
    baseline_combined = {}
    baseline_n_combined = {}
    refine_combined = {}
    refine_n_combined = {}
    
    # Loop over each weight combination in the configuration
    for room_th in config.room_selection_thresholds:
        for depth_weight, semantic_weight in tqdm.tqdm(config.weight_combinations, desc="Weight combinations"):
            weight_key = f"{room_th}_{depth_weight}_{semantic_weight}"
            print(f"\nEvaluating weight combination: {weight_key}")

            baseline_trans_errors = []
            baseline_rot_errors = []
            refine_trans_errors = []
            refine_rot_errors = []

            baseline_trans_errors_n = []
            baseline_rot_errors_n = []
            refine_trans_errors_n = []
            refine_rot_errors_n = []

            # Hyperparams for refine
            top_k = config.top_k if hasattr(config, "top_k") else 2
            min_dist_m = config.min_dist_m if hasattr(config, "min_dist_m") else 0.1
            alpha_similarity = config.alpha_similarity if hasattr(config, "alpha_similarity") else 0.3
            resolution_m_per_pixel = config.resolution_m_per_pixel if hasattr(config, "resolution_m_per_pixel") else 0.1

            # Process each sample in the test set
            for data_idx in tqdm.tqdm(range(len(test_set)), desc="Samples", leave=False):
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
                # ref_pose_map= data["ref_pose"]
                # Get the reference image from the data dictionary
                # ref_img = data["original_pano"]

                # # Build the save path for the image
                # ref_img_save_path = os.path.join(results_type_dir, f"{scene}_{idx_within_scene}_ref_img.png")

                # # Save the image using matplotlib's imsave
                # plt.imsave(ref_img_save_path, ref_img)    
                
                # --- Get depth rays ---
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

                # --- Get semantic rays ---
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

                # --- Compute probability volumes ---
                if not config.use_saved_prob_vol:
                    prob_vol_pred_depth, prob_vol_dist_pred_depth, _, depth_pred = localize(
                        torch.tensor(desdfs[scene]["desdf"]),
                        torch.tensor(pred_rays_depth, device="cpu"),
                        return_np=False,
                    )                               
                    prob_vol_pred_semantic, _, _, _ = localize(
                        torch.tensor(semantics[scene]["desdf"]),
                        torch.tensor(pred_rays_semantic, device="cpu"),
                        return_np=False,
                        # localize_type="semantic",
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

                # --- Combine volumes with current weight combination ---
                combined_prob_vol = combine_prob_volumes(
                    prob_vol_pred_depth,
                    prob_vol_pred_semantic,
                    depth_weight,
                    semantic_weight
                )

                # --- Finalize localization (room aware vs. non-room aware) ---
                if config.use_room_aware:
                    # 1) compute softmax over rooms
                    class_probs = torch.softmax(room_logits, dim=1)[0]  # shape (R,)

                    # 2) collect every room’s polygons with its weight = class probability
                    room_poly_weights = []
                    id_to_room = {v:k for k,v in room_type_to_id.items()}
                    for room_idx, prob in enumerate(class_probs):
                        room_name = id_to_room[room_idx]
                        polys = data["room_polygons"].get(room_name, [])
                        for poly in polys:
                            room_poly_weights.append((poly, prob.item()))

                    # 3) finalize localization with both global and weighted room masks
                    final_prob_vol, prob_dist_pred, orientation_map, pose_pred = finalize_localization_soft_room_threshold(
                        combined_prob_vol,
                        data["room_polygons"],
                        room_poly_weights=room_poly_weights
                    )

                    # Non-room-aware version
                    final_prob_vol_n, prob_dist_pred_n, orientation_map_n, pose_pred_n = finalize_localization(
                        combined_prob_vol, data["room_polygons"]
                    )
                else:
                    # final_prob_vol, prob_dist_pred, orientation_map, pose_pred = finalize_localization(
                    #     combined_prob_vol, data["room_polygons"]
                    # )
                    final_prob_vol, prob_dist_pred, orientation_map, pose_pred = finalize_localization(
                        combined_prob_vol, data["room_polygons"]
                    )
                    final_prob_vol_n, prob_dist_pred_n, orientation_map_n, pose_pred_n = final_prob_vol, prob_dist_pred, orientation_map, pose_pred

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

                # --- Baseline error (non-room-aware) ---
                pose_pred_np_n = np.array(pose_pred_n, dtype=np.float32)
                pose_pred_np_n[0:2] = pose_pred_np_n[0:2] / 10.0
                pred_x_n, pred_y_n, pred_o_n = pose_pred_np_n
                baseline_trans_n = np.sqrt((pred_x_n - gt_x)**2 + (pred_y_n - gt_y)**2)
                rot_diff_deg_n = abs(pred_o_n - gt_o) % (2 * math.pi)
                baseline_rot_n = min(rot_diff_deg_n, 2*math.pi - rot_diff_deg_n) / math.pi * 180.0
                
                baseline_trans_errors_n.append(baseline_trans_n)
                baseline_rot_errors_n.append(baseline_rot_n)
                
                # depth_pred = np.array(depth_pred, dtype=np.float32)
                # depth_pred[0:2] = depth_pred[0:2] / 10.0
                # plot_prob_dist(prob_vol_dist_pred_depth, save_path=results_type_dir, file_name=f"{str(scene)}_{idx_within_scene}-depth" , pose_pred=depth_pred, ref_pose_map=ref_pose_map)
                # print(f"{idx_within_scene}-{str(scene)}_depth: {baseline_trans_n} oren: {baseline_rot_n}")
                    
                # if predicted_room != 'None' and predicted_room != 'bedroom' and predicted_room != 'bathroom':
                # plot_prob_dist(prob_dist_pred_n, save_path=results_type_dir, file_name=f"{str(scene)}_{idx_within_scene}-with_no_refine" , pose_pred=pose_pred_np_n, ref_pose_map=ref_pose_map)
                # print(f"{idx_within_scene}-{str(scene)}_with_no_refine: {baseline_trans_n} oren: {baseline_rot_n}")
                # plot_prob_dist(prob_dist_pred, save_path=results_type_dir, file_name=f"{idx_within_scene}-{str(scene)}_with_room_aware_{predicted_room}_prob_{int(max_prob.item()*100)}" , pose_pred=pose_pred_np, ref_pose_map=ref_pose_map)
                # print(f"{idx_within_scene}-{str(scene)}_with_room_aware: {baseline_trans} oren: {baseline_rot}")
                    
                # --- Candidate refinement ---
                top_k_candidates = extract_top_k_locations(
                    prob_dist_pred,
                    orientation_map,
                    K=top_k,
                    min_dist_m=min_dist_m,
                    resolution_m_per_pixel=resolution_m_per_pixel,
                    num_orientations=36
                )
                
                # top_k_candidates_n = extract_top_k_locations(
                #     prob_dist_pred_n,
                #     orientation_map_n,
                #     K=top_k,
                #     min_dist_m=min_dist_m,
                #     resolution_m_per_pixel=resolution_m_per_pixel,
                #     num_orientations=36
                # )

                # Augmentation offsets (in radians)
                augmentation_offsets = {
                    "0": 0,
                    # "1": np.deg2rad(1),
                    # "2": np.deg2rad(2),
                    # "3": np.deg2rad(3),
                    "5": np.deg2rad(5),                
                    # "-1": np.deg2rad(-1),
                    # "-2": np.deg2rad(-2),
                    # "-3": np.deg2rad(-3),
                    "-5": np.deg2rad(-5),                
                }

                # Define candidate processing function
                def process_candidate(cand):
                    cand_px = cand['x']
                    cand_py = cand['y']
                    cand_o = cand['orientation_radians']

                    cand_rays_dict = compute_rays_from_candidate(
                        walls[scene],
                        maps[scene],
                        cand,
                        augmentation_offsets
                    )

                    best_offset_score = 1e9
                    best_offset = 0.0

                    for off in augmentation_offsets:
                        cand_depths, cand_sems = cand_rays_dict[off]
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

                    refined_o = cand_o + np.deg2rad(int(best_offset))
                    cand_x_m = (cand_px / 10.0)
                    cand_y_m = (cand_py / 10.0)
                    return best_offset_score, cand_x_m, cand_y_m, refined_o

                # Process candidates in parallel (room-aware)
                best_candidate_score = 1e9
                best_candidate_location = None
                best_candidate_orientation = None
                for cand in top_k_candidates:
                    pick_score, cand_x_m, cand_y_m, refined_o = process_candidate(cand)
                    if pick_score < best_candidate_score:
                        best_candidate_score = pick_score
                        best_candidate_location = (cand_x_m, cand_y_m)
                        best_candidate_orientation = refined_o

                if best_candidate_location is not None:
                    refine_x, refine_y = best_candidate_location
                    refine_o = best_candidate_orientation
                    refine_trans = np.sqrt((refine_x - gt_x)**2 + (refine_y - gt_y)**2)
                    refine_rot = angular_difference_deg(refine_o, gt_o)
                    refine_trans_errors.append(refine_trans)
                    refine_rot_errors.append(refine_rot)                

                # Process candidates in parallel (non-room-aware)
            #     best_candidate_score_n = 1e9
            #     best_candidate_location_n = None
            #     best_candidate_orientation_n = None
            #     with ThreadPoolExecutor(max_workers=len(top_k_candidates_n)) as executor:
            #         futures = [executor.submit(process_candidate, cand) for cand in top_k_candidates_n]
            #         for future in as_completed(futures):
            #             pick_score, cand_x_m, cand_y_m, refined_o = future.result()
            #             if pick_score < best_candidate_score_n:
            #                 best_candidate_score_n = pick_score
            #                 best_candidate_location_n = (cand_x_m, cand_y_m)
            #                 best_candidate_orientation_n = refined_o

            #     if best_candidate_location_n is not None:
            #         refine_x, refine_y = best_candidate_location_n
            #         refine_o = best_candidate_orientation_n
            #         refine_trans_n = np.sqrt((refine_x - gt_x)**2 + (refine_y - gt_y)**2)
            #         refine_rot_n = angular_difference_deg(refine_o, gt_o)
            #         refine_trans_errors_n.append(refine_trans_n)
            #         refine_rot_errors_n.append(refine_rot_n)

            # After processing all samples, compute recalls
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

            # Save individual JSON files per weight combination
            with open(os.path.join(results_type_dir, f"baseline_recalls_{weight_key}.json"), "w") as f:
                json.dump(baseline_recalls, f, indent=4)
            with open(os.path.join(results_type_dir, f"baseline_recalls_n_{weight_key}.json"), "w") as f:
                json.dump(baseline_recalls_n, f, indent=4)
            with open(os.path.join(results_type_dir, f"refine_recalls_{weight_key}.json"), "w") as f:
                json.dump(refine_recalls, f, indent=4)
            with open(os.path.join(results_type_dir, f"refine_recalls_n_{weight_key}.json"), "w") as f:
                json.dump(refine_recalls_n, f, indent=4)

            # Store the recalls for each summary separately
            baseline_combined[weight_key] = baseline_recalls
            baseline_n_combined[weight_key] = baseline_recalls_n
            refine_combined[weight_key] = refine_recalls
            refine_n_combined[weight_key] = refine_recalls_n

            print(f"Weight {weight_key} recalls:")
            print("  baseline:", baseline_recalls)
            print("  baseline_n:", baseline_recalls_n)
            print("  refine:", refine_recalls)
            print("  refine_n:", refine_recalls_n)

    # Create separate directories for summary tables
    baseline_dir = os.path.join(results_type_dir, "summary_baseline")
    baseline_n_dir = os.path.join(results_type_dir, "summary_baseline_n")
    refine_dir = os.path.join(results_type_dir, "summary_refine")
    refine_n_dir = os.path.join(results_type_dir, "summary_refine_n")
    os.makedirs(baseline_dir, exist_ok=True)
    os.makedirs(baseline_n_dir, exist_ok=True)
    os.makedirs(refine_dir, exist_ok=True)
    os.makedirs(refine_n_dir, exist_ok=True)

    # Create and save the summary tables
    create_combined_results_table(baseline_combined, baseline_dir)
    create_combined_results_table(baseline_n_combined, baseline_n_dir)
    create_combined_results_table(refine_combined, refine_dir)
    create_combined_results_table(refine_n_combined, refine_n_dir)
    print("All combined summary tables created.")

def main():
    parser = argparse.ArgumentParser(description="Observation evaluation with multiple weight combinations.")
    parser.add_argument(
        "--config_file",
        type=str,
        default="evaluation/configuration/S3D/config_eval_soft_room_threshold_exp.yaml",
        # default="evaluation/configuration/zind/config_eval.yaml",
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    config = AttrDict(config_dict)

    evaluate_room_aware_with_refine(config)

if __name__ == "__main__":
    main()
