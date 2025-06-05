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
import matplotlib.pyplot as plt
import numpy as np
import math
from enum import Enum

# Define object types and their colors.
class ObjectType(Enum):
    WALL = 0
    WINDOW = 1
    DOOR = 2
    UNKNOWN = 3

object_to_color = {
    ObjectType.WALL: 'black',
    ObjectType.WINDOW: 'blue',
    ObjectType.DOOR: 'red',
    ObjectType.UNKNOWN: 'pink',
}


def plot_rays_comparison(center, orig_depth, orig_semantics, interp_depth, interp_semantics,
                           orientation_offset=0.0, save_path=None,
                           gt_depth=None, gt_semantics=None,
                           gt_interp_depth=None, gt_interp_semantics=None):
    """
    Generates four separate figures (one for each set of rays):
      1. Predicted full rays (e.g. 40 rays, 2° spacing).
      2. Predicted interpolated rays (e.g. 7/9 rays, 10° spacing).
      3. Ground-truth full rays (40 rays, 2° spacing), if provided.
      4. Ground-truth interpolated rays (e.g. 7/9 rays, 10° spacing), if provided.
    """
    def plot_single_rays(depths, semantics, angle_spacing_deg, fig_suffix):
        n = len(depths)
        angles = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * np.deg2rad(angle_spacing_deg) + orientation_offset
        center_x, center_y = center

        endpoints = []
        for d, angle in zip(depths, angles):
            end_x = center_x + d * math.cos(angle)
            end_y = center_y + d * math.sin(angle)
            endpoints.append((end_x, end_y))

        plt.figure(figsize=(6, 6))
        for (end_x, end_y), sem in zip(endpoints, semantics):
            try:
                color = object_to_color.get(ObjectType(sem), 'black')
            except ValueError:
                color = 'black'
            plt.plot([center_x, end_x], [center_y, end_y], linestyle='-', color=color, linewidth=1)
        for i in range(len(endpoints) - 1):
            A = np.array(endpoints[i])
            B = np.array(endpoints[i + 1])
            M = (A + B) / 2
            try:
                color_A = object_to_color.get(ObjectType(semantics[i]), 'black')
            except ValueError:
                color_A = 'black'
            plt.plot([A[0], M[0]], [A[1], M[1]], linestyle='-', color=color_A, linewidth=8, alpha=1)
            try:
                color_B = object_to_color.get(ObjectType(semantics[i + 1]), 'black')
            except ValueError:
                color_B = 'black'
            plt.plot([M[0], B[0]], [M[1], B[1]], linestyle='-', color=color_B, linewidth=8, alpha=1)
        plt.plot(center_x, center_y, 'ko')
        plt.axis('off')
        if save_path:
            plt.savefig(f"{save_path}_{fig_suffix}.png", bbox_inches='tight', pad_inches=0)
        plt.show()

    # Four separate figures.
    plot_single_rays(orig_depth, orig_semantics, angle_spacing_deg=2, fig_suffix="pred_full")
    plot_single_rays(interp_depth, interp_semantics, angle_spacing_deg=10, fig_suffix="pred_interp")
    if gt_depth is not None and gt_semantics is not None:
        plot_single_rays(gt_depth, gt_semantics, angle_spacing_deg=2, fig_suffix="gt_full")
    if gt_interp_depth is not None and gt_interp_semantics is not None:
        plot_single_rays(gt_interp_depth, gt_interp_semantics, angle_spacing_deg=10, fig_suffix="gt_interp")

def plot_all_rays_side_by_side(center, orig_depth, orig_semantics, interp_depth, interp_semantics,
                               gt_depth, gt_semantics, gt_interp_depth, gt_interp_semantics,
                               orientation_offset=0.0, save_path=None):
    """
    Generates a single figure with 4 subplots (side by side) showing:
      - Predicted full rays (2° spacing)
      - Predicted interpolated rays (10° spacing)
      - Ground-truth full rays (2° spacing)
      - Ground-truth interpolated rays (10° spacing)
    """
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    center_x, center_y = center

    def plot_rays_on_ax(ax, depths, semantics, angle_spacing_deg, title):
        n = len(depths)
        angles = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * np.deg2rad(angle_spacing_deg) + orientation_offset
        endpoints = []
        for d, angle in zip(depths, angles):
            end_x = center_x + d * math.cos(angle)
            end_y = center_y + d * math.sin(angle)
            endpoints.append((end_x, end_y))
        for (end_x, end_y), sem in zip(endpoints, semantics):
            try:
                color = object_to_color.get(ObjectType(sem), 'black')
            except ValueError:
                color = 'black'
            ax.plot([center_x, end_x], [center_y, end_y], linestyle='-', color=color, linewidth=1)
        for i in range(len(endpoints) - 1):
            A = np.array(endpoints[i])
            B = np.array(endpoints[i + 1])
            M = (A + B) / 2
            try:
                color_A = object_to_color.get(ObjectType(semantics[i]), 'black')
            except ValueError:
                color_A = 'black'
            ax.plot([A[0], M[0]], [A[1], M[1]], linestyle='-', color=color_A, linewidth=8, alpha=1)
            try:
                color_B = object_to_color.get(ObjectType(semantics[i + 1]), 'black')
            except ValueError:
                color_B = 'black'
            ax.plot([M[0], B[0]], [M[1], B[1]], linestyle='-', color=color_B, linewidth=8, alpha=1)
        ax.plot(center_x, center_y, 'ko')
        ax.set_title(title)
        ax.axis('off')

    plot_rays_on_ax(axs[0], gt_depth, gt_semantics, 2, "GT Full")
    plot_rays_on_ax(axs[1], orig_depth, orig_semantics, 2, "Pred Full")
    plot_rays_on_ax(axs[2], gt_interp_depth, gt_interp_semantics, 10, "GT Interp")
    plot_rays_on_ax(axs[3], interp_depth, interp_semantics, 10, "Pred Interp")

    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_side_by_side.png", bbox_inches='tight', pad_inches=0)
    plt.show()

def indices_to_radians(orientation_idx: int, num_orientations: int = 36) -> float:
    return orientation_idx / num_orientations * 2.0 * math.pi

def extract_top_k_locations(prob_dist: np.ndarray,
                            orientation_map: np.ndarray,
                            K: int = 10,
                            min_dist_m: float = 0.05,
                            resolution_m_per_pixel: float = 0.1,
                            num_orientations: int = 36):
    H, W = prob_dist.shape
    prob_dist_torch = torch.from_numpy(prob_dist)
    flat_prob = prob_dist_torch.view(-1)
    orientation_map = torch.from_numpy(orientation_map)
    flat_orient = orientation_map.view(-1)
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
    H = min(prob_vol_depth.shape[0], prob_vol_semantic.shape[0])
    W = min(prob_vol_depth.shape[1], prob_vol_semantic.shape[1])
    O = min(prob_vol_depth.shape[2], prob_vol_semantic.shape[2])
    depth_sliced = prob_vol_depth[:H, :W, :O]
    semantic_sliced = prob_vol_semantic[:H, :W, :O]
    return depth_weight * depth_sliced + semantic_weight * semantic_sliced

def get_max_over_orientation(prob_vol: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    prob_dist, orientation_map = torch.max(prob_vol, dim=2)
    return prob_dist, orientation_map

def compute_rays_from_candidate(walls_map: np.ndarray,
                                semantic_map: np.ndarray,
                                cand,
                                augmentation_offsets):
    candidate_cache = {}
    ray_n = 40  # number of rays per candidate per augmentation
    F_W = 1 / np.tan(0.698132) / 2
    depth_max = 15  # maximum depth in meters
    base_x = cand['x']
    base_y = cand['y']
    base_orientation = cand['orientation_radians']
    center_x = base_x * 10
    center_y = base_y * 10
    candidate_pos_pixels = np.array([center_x, center_y])
    center_angs = np.flip(np.arctan2((np.arange(ray_n) - np.arange(ray_n).mean()), ray_n * F_W))
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

def measure_similarity(pred_depths: np.ndarray,
                       pred_semantics: np.ndarray,
                       cand_depths: np.ndarray,
                       cand_semantics: np.ndarray,
                       alpha: float = 0.5):
    cand_semantics = cand_semantics.copy()
    cand_semantics[cand_semantics == 2] = -1
    cand_semantics[cand_semantics == 1] = 2
    cand_semantics[cand_semantics == -1] = 1
    depth_error = np.mean(np.abs(pred_depths - cand_depths))
    sem_error = np.mean((pred_semantics != cand_semantics).astype(float))
    score = alpha * depth_error + (1.0 - alpha) * sem_error
    return score

def angular_difference_deg(ang1_rad, ang2_rad):
    diff_deg = abs(math.degrees(ang1_rad) - math.degrees(ang2_rad)) % 360
    return min(diff_deg, 360 - diff_deg)

# =============================================================================
# Main evaluation pipeline supporting multiple weight combinations
# =============================================================================
def evaluate_room_aware_with_refine(config):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    with open(config.split_file, "r") as f:
        split = AttrDict(yaml.safe_load(f))
    scene_names = split.test[10:50]
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
    results_type_dir = os.path.join(config.results_dir, config.prediction_type)
    if config.use_ground_truth_depth or config.use_ground_truth_semantic:
        results_type_dir = os.path.join(results_type_dir, "gt")
    os.makedirs(results_type_dir, exist_ok=True)
    desdfs, semantics, maps, gt_poses, valid_scene_names, walls = load_scene_data(
        test_set, config.dataset_dir, config.desdf_path
    )
    baseline_n_combined = {}
    refine_n_combined = {}
    
    for depth_weight, semantic_weight in tqdm.tqdm(config.weight_combinations, desc="Weight combinations"):
        weight_key = f"{depth_weight}_{semantic_weight}"
        print(f"\nEvaluating weight combination: {weight_key}")
        baseline_trans_errors = []
        baseline_rot_errors = []
        baseline_trans_errors_n = []
        baseline_rot_errors_n = []
        refine_trans_errors_n = []
        refine_rot_errors_n = []
        top_k = config.top_k if hasattr(config, "top_k") else 2
        min_dist_m = config.min_dist_m if hasattr(config, "min_dist_m") else 0.05
        alpha_similarity = config.alpha_similarity if hasattr(config, "alpha_similarity") else 0.8
        resolution_m_per_pixel = config.resolution_m_per_pixel if hasattr(config, "resolution_m_per_pixel") else 0.1

        for data_idx in tqdm.tqdm(range(len(test_set)), desc="Samples", leave=False):
            data = test_set[data_idx]
            scene_idx = np.sum(data_idx >= np.array(test_set.scene_start_idx)) - 1
            scene = test_set.scene_names[scene_idx]
            if 'floor' not in scene:
                scene_number = int(scene.split("_")[1])
                scene = f"scene_{scene_number}"
            if scene not in valid_scene_names:
                continue
            idx_within_scene = data_idx - test_set.scene_start_idx[scene_idx]
            ref_pose_map = gt_poses[scene][idx_within_scene * (config.L + 1) + config.L, :]
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
    
            file_name = f"{str(scene)}-{idx_within_scene}_rays_plot.png"
            save_full_path = os.path.join(results_type_dir, file_name)
            
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
            combined_prob_vol = combine_prob_volumes(
                prob_vol_pred_depth,
                prob_vol_pred_semantic,
                depth_weight,
                semantic_weight
            )
            final_prob_vol, prob_dist_pred, orientation_map, pose_pred = finalize_localization(
                combined_prob_vol, data["room_polygons"])        

            final_prob_vol_n, prob_dist_pred_n, orientation_map_n, pose_pred_n = final_prob_vol, prob_dist_pred, orientation_map, pose_pred
            pose_pred_np = np.array(pose_pred, dtype=np.float32)
            pose_pred_np[0:2] = pose_pred_np[0:2] / 10.0
            gt_x, gt_y, gt_o = ref_pose_map[:3]
            pred_x, pred_y, pred_o = pose_pred_np
            baseline_trans = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
            rot_diff_deg = abs(pred_o - gt_o) % (2 * math.pi)
            baseline_rot = min(rot_diff_deg, 2*math.pi - rot_diff_deg) / math.pi * 180.0
            baseline_trans_errors.append(baseline_trans)
            baseline_rot_errors.append(baseline_rot)
            pose_pred_np_n = np.array(pose_pred_n, dtype=np.float32)
            pose_pred_np_n[0:2] = pose_pred_np_n[0:2] / 10.0
            pred_x_n, pred_y_n, pred_o_n = pose_pred_np_n
            baseline_trans_n = np.sqrt((pred_x_n - gt_x)**2 + (pred_y_n - gt_y)**2)
            rot_diff_deg_n = abs(pred_o_n - gt_o) % (2 * math.pi)
            baseline_rot_n = min(rot_diff_deg_n, 2*math.pi - rot_diff_deg_n) / math.pi * 180.0
            baseline_trans_errors_n.append(baseline_trans_n)
            baseline_rot_errors_n.append(baseline_rot_n)
                        
            top_k_candidates_n = extract_top_k_locations(
                prob_dist_pred_n,
                orientation_map_n,
                K=top_k,
                min_dist_m=min_dist_m,
                resolution_m_per_pixel=resolution_m_per_pixel,
                num_orientations=36
            )
            augmentation_offsets = {"0": 0}
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
                refine_trans = np.sqrt((cand_x_m - gt_x)**2 + (cand_y_m - gt_y)**2)
                refine_rot = angular_difference_deg(refined_o, gt_o)
                pick_score = refine_trans
                return pick_score, cand_x_m, cand_y_m, refined_o    
            best_candidate_score_n = 1e9
            best_candidate_location_n = None
            best_candidate_orientation_n = None
            # with ThreadPoolExecutor(max_workers=len(top_k_candidates_n)) as executor:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(process_candidate, cand) for cand in top_k_candidates_n]
                for future in as_completed(futures):
                    pick_score, cand_x_m, cand_y_m, refined_o = future.result()
                    if pick_score < best_candidate_score_n:
                        best_candidate_score_n = pick_score
                        best_candidate_location_n = (cand_x_m, cand_y_m)
                        best_candidate_orientation_n = refined_o
            if best_candidate_location_n is not None:
                refine_x, refine_y = best_candidate_location_n
                refine_o = best_candidate_orientation_n
                refine_trans_n = np.sqrt((refine_x - gt_x)**2 + (refine_y - gt_y)**2)
                refine_rot_n = angular_difference_deg(refine_o, gt_o)
                refine_trans_errors_n.append(refine_trans_n)
                refine_rot_errors_n.append(refine_rot_n)
                           
                # Compute GT rays (full) and then interpolate.
                gt_rays_depth = data["ref_depth"]
                gt_rays_semantics = data["ref_semantics"]
                gt_depth_interp = get_ray_from_depth(gt_rays_depth, V=config.V, F_W=config.F_W)
                gt_sem_interp = get_ray_from_semantics_v2(gt_rays_semantics)
                
                mismatch_count = np.sum(gt_rays_semantics != sampled_semantic_indices_np)

                if refine_trans_n < 1.5 and mismatch_count < 5 :
                # if baseline_trans_n < 1 and np.all(pose_pred != pose_pred_no_rooms):
                # if  baseline_trans_n < 0.01:
                    
                    # Plot the four individual figures.
                    plot_rays_comparison(center=(50, 50),
                                         orig_depth=pred_depths,
                                         orig_semantics=sampled_semantic_indices_np,
                                         interp_depth=pred_rays_depth,
                                         interp_semantics=pred_rays_semantic,
                                         orientation_offset=0.0,
                                         save_path=save_full_path,
                                         gt_depth=gt_rays_depth,
                                         gt_semantics=gt_rays_semantics,
                                         gt_interp_depth=gt_depth_interp,
                                         gt_interp_semantics=gt_sem_interp)
                    # Plot one additional composite figure with 4 subplots side by side.
                    plot_all_rays_side_by_side(center=(50, 50),
                                               orig_depth=pred_depths,
                                               orig_semantics=sampled_semantic_indices_np,
                                               interp_depth=pred_rays_depth,
                                               interp_semantics=pred_rays_semantic,
                                               gt_depth=gt_rays_depth,
                                               gt_semantics=gt_rays_semantics,
                                               gt_interp_depth=gt_depth_interp,
                                               gt_interp_semantics=gt_sem_interp,
                                               orientation_offset=0.0,
                                               save_path=save_full_path)
                    # plot_prob_dist(prob_dist_no_rooms, save_path=results_type_dir,
                    #                file_name=f"{str(scene)}-{idx_within_scene}_no_room",
                    #                pose_pred=pose_pred_np_no_room,
                    #                ref_pose_map=ref_pose_map)
                    plot_prob_dist(prob_dist_pred, save_path=results_type_dir,
                                   file_name=f"{str(scene)}-{idx_within_scene}_ab_with_refine",
                                   pose_pred=[refine_x, refine_y, refine_o],
                                   ref_pose_map=ref_pose_map)
                    print(f"{str(scene)}-{idx_within_scene}_with_zrefine acc: {refine_trans_n} oren: {refine_rot_n}")
                    plot_prob_dist(prob_dist_pred_n, save_path=results_type_dir,
                                   file_name=f"{str(scene)}-{idx_within_scene}_a_no_room_aware",
                                   pose_pred=pose_pred_np_n,
                                   ref_pose_map=ref_pose_map)
                    print(f"{idx_within_scene}-{str(scene)}_no_room_aware: {baseline_trans_n} oren: {baseline_rot_n}")

        baseline_trans_errors_n = np.array(baseline_trans_errors_n)
        baseline_rot_errors_n = np.array(baseline_rot_errors_n)
        refine_trans_errors_n = np.array(refine_trans_errors_n)
        refine_rot_errors_n = np.array(refine_rot_errors_n)
        baseline_recalls_n = calculate_recalls(baseline_trans_errors_n, baseline_rot_errors_n)
        refine_recalls_n = calculate_recalls(refine_trans_errors_n, refine_rot_errors_n)
        with open(os.path.join(results_type_dir, f"baseline_recalls_n_{weight_key}.json"), "w") as f:
            json.dump(baseline_recalls_n, f, indent=4)
        with open(os.path.join(results_type_dir, f"refine_recalls_{weight_key}.json"), "w") as f:
            json.dump(refine_recalls_n, f, indent=4)
        baseline_n_combined[weight_key] = baseline_recalls_n
        refine_n_combined[weight_key] = refine_recalls_n
        print(f"Weight {weight_key} recalls:")
        print("  baseline_n:", baseline_recalls_n)
        print("  refine_n:", refine_recalls_n)
    baseline_n_dir = os.path.join(results_type_dir, "summary_baseline_n")
    refine_n_dir = os.path.join(results_type_dir, "summary_refine_n")
    os.makedirs(baseline_n_dir, exist_ok=True)
    os.makedirs(refine_n_dir, exist_ok=True)
    create_combined_results_table(baseline_n_combined, baseline_n_dir)
    create_combined_results_table(refine_n_combined, refine_n_dir)
    print("All combined summary tables created.")

def main():
    parser = argparse.ArgumentParser(description="Observation evaluation with multiple weight combinations.")
    parser.add_argument(
        "--config_file",
        type=str,
        # default="evaluation/configuration/S3D/config_eval.yaml",
        default="evaluation/configuration/zind/config_eval.yaml",
        help="Path to the configuration file",
    )
    args = parser.parse_args()
    with open(args.config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    config = AttrDict(config_dict)
    evaluate_room_aware_with_refine(config)

if __name__ == "__main__":
    main()
