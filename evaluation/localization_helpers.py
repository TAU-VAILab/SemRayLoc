import torch

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