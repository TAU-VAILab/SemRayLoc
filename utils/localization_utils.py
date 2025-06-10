from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import *
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from modules.semantic.semantic_mapper import ObjectType

def custom_loss_with_normalization(desdf, rays, semantic_weights):
    weights = torch.tensor(semantic_weights, device=rays.device)
    mismatches = (rays != desdf).float()  # Shape: (H, W, V)
    ray_weights = weights[desdf.long()]  # Shape: (H, W, V)
    weighted_errors = mismatches * ray_weights  # Shape: (H, W, V)
    total_penalty_per_pixel = weighted_errors.sum(dim=2)  # Shape: (H, W)
    loss = -total_penalty_per_pixel
    return loss

def localize(
    desdf: torch.tensor, rays: torch.tensor, orn_slice=36, return_np=True, lambd=40, localize_type = "depth", semantic_weights = [1.0, 5.0, 3.0, 0.0]
) -> Tuple[torch.tensor]:
    rays = torch.flip(rays, [0])
    O = desdf.shape[2]
    V = rays.shape[0]
    rays = rays.reshape((1, 1, -1))

    pad_front = V // 2
    pad_back = V - pad_front
    pad_desdf = F.pad(desdf, [pad_front, pad_back], mode="circular")

    if localize_type == "depth":
        prob_vol = torch.stack(
            [
                -torch.norm(pad_desdf[:, :, i : i + V] - rays, p=1.0, dim=2)
                for i in range(O)
            ],
            dim=2,
        )  # (H,W,O)
        prob_vol = torch.exp(prob_vol / lambd)
    else:
        prob_vol = torch.stack(
            [
                custom_loss_with_normalization(pad_desdf[:, :, i : i + V], rays, semantic_weights)
                for i in range(O)
            ],
            dim=2,
        )
        prob_vol = torch.exp(prob_vol / lambd)   
        
    # maxpooling
    prob_dist, orientations = torch.max(prob_vol, dim=2)
    
    pred_y_in_pixel, pred_x_in_pixel = torch.where(prob_dist == prob_dist.max())
    sampled_index = torch.randint(0, pred_y_in_pixel.shape[0], (1,))
    
    pred_y = pred_y_in_pixel[sampled_index]
    pred_x = pred_x_in_pixel[sampled_index]
    
    orn = orientations[pred_y, pred_x]
    orn = orn / orn_slice * 2 * torch.pi
    pred = torch.cat((pred_x, pred_y, orn))
    if return_np:
        return (
            prob_vol.detach().cpu().numpy(),
            prob_dist.detach().cpu().numpy(),
            orientations.detach().cpu().numpy(),
            pred.detach().cpu().numpy(),
        )
    else:
        return (
            prob_vol.to(torch.float32).detach().cpu(),
            prob_dist.to(torch.float32).detach().cpu(),
            orientations.to(torch.float32).detach().cpu(),
            pred.to(torch.float32).detach().cpu(),
        )

def finalize_localization(prob_vol: torch.Tensor, all_polygons, room_polygons=[]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    H, W, O = prob_vol.shape 
    scale_factor = 10  

    def apply_polygon(poly, mask):       
        if "coordinates" in poly:
            coords = np.array(poly["coordinates"], dtype=np.float32) / scale_factor
            coords = np.round(coords).astype(np.int32)
            tmp_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(tmp_mask, [coords], color=1)
            mask |= torch.tensor(tmp_mask, device=prob_vol.device, dtype=torch.bool)
        else:
            min_x_idx = max(int(math.floor(poly['min_x'] / scale_factor)), 0)
            min_y_idx = max(int(math.floor(poly['min_y'] / scale_factor)), 0)
            max_x_idx = min(int(math.ceil(poly['max_x'] / scale_factor)), W)
            max_y_idx = min(int(math.ceil(poly['max_y'] / scale_factor)), H)
            mask[min_y_idx:max_y_idx, min_x_idx:max_x_idx] = True
        return mask

    if all_polygons:
        mask_all = torch.zeros((H, W), dtype=torch.bool, device=prob_vol.device)
        if isinstance(all_polygons, dict):
            polygons_list = []
            for poly_list in all_polygons.values():
                polygons_list.extend(poly_list)
        else:
            polygons_list = all_polygons

        for poly in polygons_list:
            mask_all = apply_polygon(poly, mask_all)
        
        mask_all_expanded = mask_all.unsqueeze(2)  # shape (H, W, 1)
        prob_vol = prob_vol * mask_all_expanded

    if room_polygons:
        mask_room = torch.zeros((H, W), dtype=torch.bool, device=prob_vol.device)
        for poly in room_polygons:
            mask_room = apply_polygon(poly, mask_room)
        mask_room_expanded = mask_room.unsqueeze(2)  # shape (H, W, 1)
        prob_vol = prob_vol * mask_room_expanded

    prob_dist, orientations = torch.max(prob_vol, dim=2)  # shapes: (H, W)

    pred_y_indices, pred_x_indices = torch.where(prob_dist == prob_dist.max())
    sampled_index = torch.randint(0, pred_y_indices.shape[0], (1,))
    
    pred_y = pred_y_indices[sampled_index]
    pred_x = pred_x_indices[sampled_index]
    
    orn_idx = orientations[pred_y, pred_x]
    orn = orn_idx / 36 * 2 * torch.pi
    pred = torch.cat((pred_x, pred_y, orn))
    
    return (
        prob_vol.detach().cpu().numpy(),
        prob_dist.detach().cpu().numpy(),
        orientations.detach().cpu().numpy(),
        pred.detach().cpu().numpy()
    )


    
def get_ray_from_depth(d, V=9, dv=10, a0=None, F_W=1/np.tan(0.698132)/2):
    """
    Shoot the rays to the depths, from left to right
    Input:
        d: 1d depths from image
        V: number of rays
        dv: angle between two neighboring rays
        a0: camera intrisic
        F/W: focal length / image width
    Output:
        rays: interpolated rays
    """
    W = d.shape[0]
    angles = (np.arange(0, V) - np.arange(0, V).mean()) * dv / 180 * np.pi
    if a0 is None:
        # assume a0 is in the middle of the image
        w = np.tan(angles) * W * F_W + (W - 1) / 2  # desired width, left to right
    else:
        w = np.tan(angles) * W * F_W + a0  # left to right
    w = np.clip(w, 0, W - 1)
    interp_d = griddata(np.arange(W).reshape(-1, 1), d, w, method="linear")
    rays = interp_d / np.cos(angles)

    return rays

from collections import Counter

def get_ray_from_semantics(original_rays, angle_between_rays=80/40, desired_ray_count=9, window_size=1):
    desired_angle_step = 10  

    representative_rays = []
    
    # Assume the center of the FOV corresponds to the middle index.
    center_index = len(original_rays) // 2

    # For an odd number of desired rays, the middle one (index desired_ray_count//2) is 0°.
    # Thus, the desired angles (in degrees) are computed relative to 0.
    for i in range(desired_ray_count):
        # Compute desired angle relative to the center.
        desired_angle = (i - desired_ray_count // 2) * desired_angle_step
        
        # Compute the corresponding index offset (how many original rays away from the center).
        idx_offset = desired_angle / angle_between_rays
        
        # The target index is the center index plus the offset.
        idx = round(center_index + idx_offset)
        
        # Clamp the index so it remains within the valid range.
        idx = max(0, min(idx, len(original_rays) - 1))
        
        # If a window is provided, collect neighbors around the target index.
        neighbors = []
        for j in range(max(0, idx - window_size), min(len(original_rays), idx + window_size + 1)):
            neighbors.append(original_rays[j])
        
        # Use majority vote from the neighbors if there is a window; otherwise, just take the ray.
        count = Counter(neighbors)
        majority_class = count.most_common(1)[0][0]
        
        representative_rays.append(majority_class)
    
    return np.array(representative_rays)