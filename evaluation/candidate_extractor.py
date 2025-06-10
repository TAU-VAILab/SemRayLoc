import torch
import math
import numpy as np

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