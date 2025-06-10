import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.raycast_utils import ray_cast

def compute_rays_from_candidate(
    walls_map: np.ndarray,
    semantic_map: np.ndarray,
    cand,
    augmentation_offsets,
):
    candidate_cache = {}
    ray_n = 40
    F_W = 1 / np.tan(0.698132) / 2
    depth_max = 15

    base_x, base_y, base_orientation = cand['x'], cand['y'], cand['orientation_radians']
    candidate_pos_pixels = np.array([base_x * 10, base_y * 10])
    center_angs = np.flip(np.arctan2((np.arange(ray_n) - np.arange(ray_n).mean()), ray_n * F_W))
    candidate_aug_rays = {}

    for aug_key, aug_offset in augmentation_offsets.items():
        ray_angles = center_angs + base_orientation + aug_offset
        depth_rays, semantic_rays = [], []
        for ang in ray_angles:
            cache_key = (round(base_x, 2), round(base_y, 2), round(math.degrees(ang)))
            if cache_key in candidate_cache:
                depth_val_m, prediction_class = candidate_cache[cache_key]
            else:
                depth_val_m, prediction_class, _, = ray_cast(
                    semantic_map, candidate_pos_pixels, ang, dist_max=depth_max * 100, min_dist=5
                )
                depth_val_m /= 100.0
                candidate_cache[cache_key] = (depth_val_m, prediction_class)
            depth_rays.append(depth_val_m)
            semantic_rays.append(prediction_class)
        candidate_aug_rays[aug_key] = (depth_rays, semantic_rays)
    return candidate_aug_rays

def measure_similarity(pred_depths, pred_semantics, cand_depths, cand_semantics, alpha=0.5):
    depth_error = np.mean(np.abs(np.array(pred_depths) - np.array(cand_depths)))
    
    cand_semantics_int = np.array(cand_semantics, dtype=np.int32)
    pred_semantics_int = np.array(pred_semantics, dtype=np.int32)
    cand_semantics_int[cand_semantics_int == 3] = 0
    pred_semantics_int[pred_semantics_int == 3] = 0
    
    penalty_matrix = np.array([[0, 1, 1], [2, 0, 0], [2, 0, 0]])
    sem_penalties = penalty_matrix[pred_semantics_int, cand_semantics_int]
    sem_error = np.mean(sem_penalties)
    
    return alpha * depth_error + (1.0 - alpha) * sem_error

def angular_difference_deg(ang1_rad, ang2_rad):
    diff_deg = abs(math.degrees(ang1_rad) - math.degrees(ang2_rad)) % 360
    return min(diff_deg, 360 - diff_deg)

def refine_and_select_best_candidate(top_k_candidates, walls_map, semantic_map, pred_depths, pred_semantics, alpha_similarity, use_multithreading=True):
    if not top_k_candidates:
        return None, None, None

    augmentation_offsets = {"0": 0, "5": np.deg2rad(5), "-5": np.deg2rad(-5)}

    def process_candidate(cand):
        cand_rays_dict = compute_rays_from_candidate(walls_map, semantic_map, cand, augmentation_offsets)
        best_offset_score, best_offset = 1e9, "0"
        for off_key in augmentation_offsets:
            score = measure_similarity(pred_depths, pred_semantics, *cand_rays_dict[off_key], alpha=alpha_similarity)
            if score < best_offset_score:
                best_offset_score, best_offset = score, off_key
        
        refined_o = cand['orientation_radians'] + augmentation_offsets[best_offset]
        return best_offset_score, (cand['x'] / 10.0, cand['y'] / 10.0), refined_o

    best_score, best_location, best_orientation = 1e9, None, None
    
    if use_multithreading and len(top_k_candidates) > 1:
        with ThreadPoolExecutor(max_workers=len(top_k_candidates)) as executor:
            futures = {executor.submit(process_candidate, cand): cand for cand in top_k_candidates}
            for future in as_completed(futures):
                score, location, orientation = future.result()
                if score < best_score:
                    best_score, best_location, best_orientation = score, location, orientation
    else:
        for cand in top_k_candidates:
            score, location, orientation = process_candidate(cand)
            if score < best_score:
                best_score, best_location, best_orientation = score, location, orientation
                
    return best_location, best_orientation, best_score 