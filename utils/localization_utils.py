from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import *
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torch
from modules.semantic.semantic_mapper import ObjectType

def custom_loss_with_normalization(desdf, rays, semantic_weights):
    # Define weights according to the enum: WALL=0, WINDOW=1, DOOR=2, UNKNOWN=3
    weights = torch.tensor(semantic_weights, device=rays.device)
    
    # Create a mask that is 1 when predicted and ground truth differ, else 0.
    mismatches = (rays != desdf).float()  # Shape: (H, W, V)
    
    # Look up the weight for each predicted ray.
    ray_weights = weights[desdf.long()]  # Shape: (H, W, V)
    
    # Compute the weighted error per ray.
    weighted_errors = mismatches * ray_weights  # Shape: (H, W, V)
    
    # Sum the weighted errors over the ray dimension.
    total_penalty_per_pixel = weighted_errors.sum(dim=2)  # Shape: (H, W)
    
    # Return the negative total penalty so that a perfect match yields 0.
    loss = -total_penalty_per_pixel
    return loss

def localize(
    desdf: torch.tensor, rays: torch.tensor, orn_slice=36, return_np=True, lambd=40, localize_type = "depth", semantic_weights = [1.0, 5.0, 3.0, 0.0]
) -> Tuple[torch.tensor]:
    """
    Localize in the desdf according to the rays
    Input:
        desdf: (H, W, O), counter clockwise
        rays: (V,) from left to right (clockwise)
        orn_slice: number of orientations
        return_np: return as ndarray instead of torch.tensor
        lambd: parameter for likelihood
    Output:
        prob_vol: probability volume (H, W, O), ndarray
        prob_dist: probability distribution, (H, W) maxpool the prob_vol along orientation, ndarray
        orientations: orientation with max likelihood at each position, (H, W), ndarray
        pred: (3, ) predicted state [x,y,theta], ndarray
    """

    # flip the ray, to make rotation direction mathematically positive
    rays = torch.flip(rays, [0])
    O = desdf.shape[2]
    V = rays.shape[0]
    # expand rays to have the same dimension as desdf
    rays = rays.reshape((1, 1, -1))

    # circular pad the desdf
    pad_front = V // 2
    pad_back = V - pad_front
    pad_desdf = F.pad(desdf, [pad_front, pad_back], mode="circular")

    if localize_type == "depth":
        # probablility is -l1norm
        prob_vol = torch.stack(
            [
                -torch.norm(pad_desdf[:, :, i : i + V] - rays, p=1.0, dim=2)
                for i in range(O)
            ],
            dim=2,
        )  # (H,W,O)
        prob_vol = torch.exp(prob_vol / lambd)  # NOTE: here make prob positive
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
    
    # get the prediction
    pred_y_in_pixel, pred_x_in_pixel = torch.where(prob_dist == prob_dist.max())
    sampled_index = torch.randint(0, pred_y_in_pixel.shape[0], (1,))
    
    pred_y = pred_y_in_pixel[sampled_index]
    pred_x = pred_x_in_pixel[sampled_index]
    
    orn = orientations[pred_y, pred_x]
    # from orientation indices to radians
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


def localize_soft_semantics(
    desdf: torch.tensor, rays: torch.tensor, orn_slice=36, return_np=True, lambd=40, localize_type = "depth"
) -> Tuple[torch.tensor]:
    H, W, O = desdf.shape
    device = desdf.device

    if localize_type == "depth":
        # — unchanged depth branch —
        V = rays.shape[0]
        rays_flipped = torch.flip(rays, [0]).view(1,1,-1)
        pad_front = V // 2
        pad_back  = V - pad_front
        pad_desdf = F.pad(desdf, (pad_front, pad_back), mode="circular")  # (H,W,O+V)

        prob_vol = torch.stack([
            -torch.norm(pad_desdf[:, :, i : i + V] - rays_flipped, p=1, dim=2)
            for i in range(O)
        ], dim=2)  # (H,W,O)
        prob_vol = torch.exp(prob_vol / lambd)

    elif localize_type == "sem":
        # — new semantic branch w/ mean predicted-prob score —
        # desdf: (H,W,O) ints; rays: (V,C) floats
        V, C = rays.shape

        # 1) circularly pad the orientation dimension
        pad_front = V // 2
        pad_back  = V - pad_front
        pad_desdf = F.pad(desdf, (pad_front, pad_back), mode="circular")  # (H,W,O+V)

        # 2) expand rays to (H, W, V, C)
        rays_exp = rays.view(1,1,V,C).expand(H, W, V, C)

        prob_vol = torch.zeros((H, W, O), device=device)
        eps = 1e-9

        for i in range(O):
            # 3) per‐orientation, grab GT labels for V rays
            gt = pad_desdf[:, :, i : i + V].long()            # (H,W,V)
            idx = gt.unsqueeze(-1)                            # (H,W,V,1)
            # gather predicted prob of the GT class
            p_gt = rays_exp.gather(dim=3, index=idx).squeeze(-1)  # (H,W,V)
            p_gt = p_gt.clamp(min=eps)

            # 4) score = mean(probabilities) in [0,1]
            prob_vol[:, :, i] = p_gt.mean(dim=2)

    else:
        raise ValueError(f"Unknown localize_type {localize_type!r}")

    # --- common post‐processing ---
    prob_dist, orientations = torch.max(prob_vol, dim=2)  # both (H,W)

    ys, xs = torch.where(prob_dist == prob_dist.max())
    choice = torch.randint(len(ys), (1,))
    y, x = ys[choice], xs[choice]

    theta = orientations[y, x].float() / orn_slice * 2 * torch.pi
    pred = torch.cat([x.float(), y.float(), theta])

    if return_np:
        return (
            prob_vol.cpu().numpy(),
            prob_dist.cpu().numpy(),
            orientations.cpu().numpy(),
            pred.cpu().numpy(),
        )
    else:
        return (
            prob_vol.float().cpu(),
            prob_dist.float().cpu(),
            orientations.float().cpu(),
            pred.float().cpu(),
        )

import math
import cv2
import numpy as np
import torch
from typing import Tuple

def finalize_localization_soft_room_threshold(prob_vol,all_polygons, room_poly_weights=None):
    """
    Apply the global mask, then weight every room polygon by its softmax probability.

    Args:
      prob_vol: torch.Tensor of shape (H, W, O) on device
      all_polygons: either a dict mapping names to lists of polygons, or a list of polygon dicts
      room_poly_weights: list of (poly_dict, room_prob) tuples (or None)

    Returns:
      - masked probability volume (numpy array)
      - per-pixel max probability (numpy array)
      - per-pixel best orientation index (numpy array)
      - predicted [x, y, theta] (numpy array)
    """
    if room_poly_weights is None:
        room_poly_weights = []

    H, W, O = prob_vol.shape
    scale = 10
    device = prob_vol.device

    def apply_polygon(poly, mask):
        """
        Update the given mask with the area defined by poly.
        poly can either be:
          - A bounding box (keys 'min_x', 'min_y', 'max_x', 'max_y')
          - A full polygon (key "coordinates": list of [x, y] vertices)
        """
        if "coordinates" in poly:
            # Full polygon: scale coordinates and fill using OpenCV.
            coords = np.array(poly["coordinates"], dtype=np.float32) / scale
            coords = np.round(coords).astype(np.int32)
            tmp_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(tmp_mask, [coords], color=1)
            mask |= torch.tensor(tmp_mask, device=prob_vol.device, dtype=torch.bool)
        else:
            # Bounding box: convert to grid indices.
            min_x_idx = max(int(math.floor(poly['min_x'] / scale)), 0)
            min_y_idx = max(int(math.floor(poly['min_y'] / scale)), 0)
            max_x_idx = min(int(math.ceil(poly['max_x'] / scale)), W)
            max_y_idx = min(int(math.ceil(poly['max_y'] / scale)), H)
            mask[min_y_idx:max_y_idx, min_x_idx:max_x_idx] = True
        return mask

    # --- Apply mask for all_polygons ---
    if all_polygons:
        mask_all = torch.zeros((H, W), dtype=torch.bool, device=prob_vol.device)
        # If all_polygons is a dict, flatten the list of polygon dicts.
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

    def poly_mask(poly):
        if "coordinates" in poly:
            coords = (np.array(poly["coordinates"], float) / scale).round().astype(np.int32)
            tmp = np.zeros((H, W), np.uint8)
            cv2.fillPoly(tmp, [coords], 1)
            return torch.from_numpy(tmp.astype(bool)).to(device)
        else:
            min_x = max(int(poly["min_x"] // scale), 0)
            min_y = max(int(poly["min_y"] // scale), 0)
            max_x = min(int(np.ceil(poly["max_x"] / scale)), W)
            max_y = min(int(np.ceil(poly["max_y"] / scale)), H)
            m = torch.zeros((H, W), dtype=torch.bool, device=device)
            m[min_y:max_y, min_x:max_x] = True
            return m

    mask_rooms = torch.zeros((H, W), dtype=torch.float32, device=device)
    for poly, w in room_poly_weights:
        m = poly_mask(poly).float() * w
        mask_rooms = torch.maximum(mask_rooms, m)

    # --- Apply that mask to every orientation plane ---
    prob_vol = prob_vol * mask_rooms.unsqueeze(2)

    # --- Pick best pixel & orientation ---
    prob_dist, orientations = torch.max(prob_vol, dim=2)
    ys, xs = torch.where(prob_dist == prob_dist.max())
    idx = torch.randint(0, len(xs), (1,))
    y, x = ys[idx], xs[idx]

    orn_idx = orientations[y, x]
    orn = orn_idx / O * 2 * torch.pi

    pred = torch.stack((x.float(), y.float(), orn))

    return (
        prob_vol.detach().cpu().numpy(),
        prob_dist.detach().cpu().numpy(),
        orientations.detach().cpu().numpy(),
        pred.detach().cpu().numpy()
    )


def finalize_localization(prob_vol: torch.Tensor, all_polygons, room_polygons=[]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Finalize localization using the combined probability volume.
    
    Inputs:
        prob_vol: combined probability volume with shape (H, W, O),
                  where H and W are on a low-resolution grid (e.g. 82 x 144)
                  and O is the orientation dimension (e.g. 36).
        all_polygons: list (or dict) of polygon definitions.
                      Each polygon can be either a bounding box with keys:
                      'min_x', 'min_y', 'max_x', 'max_y'
                      OR a full polygon with a "coordinates" key (list of [x,y] vertices).
                      Coordinates are defined in the high-resolution semantic map 
                      (e.g. 820 x 1440) and will be scaled down by a factor of 10.
        room_polygons: (optional) list of polygon definitions for a room.
                       Format is the same as for all_polygons.
    
    Outputs:
        Returns a tuple:
          - prob_vol (masked), as a NumPy array.
          - prob_dist: probability distribution (H, W), as a NumPy array.
          - orientations: (H, W), as a NumPy array.
          - pred: predicted state [x, y, theta] as a NumPy array.
    """
    H, W, O = prob_vol.shape  # e.g., 82, 144, 36
    scale_factor = 10  # semantic map resolution is 10x that of the probability grid

    def apply_polygon(poly, mask):
        """
        Update the given mask with the area defined by poly.
        poly can either be:
          - A bounding box (keys 'min_x', 'min_y', 'max_x', 'max_y')
          - A full polygon (key "coordinates": list of [x, y] vertices)
        """
        if "coordinates" in poly:
            # Full polygon: scale coordinates and fill using OpenCV.
            coords = np.array(poly["coordinates"], dtype=np.float32) / scale_factor
            coords = np.round(coords).astype(np.int32)
            tmp_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(tmp_mask, [coords], color=1)
            mask |= torch.tensor(tmp_mask, device=prob_vol.device, dtype=torch.bool)
        else:
            # Bounding box: convert to grid indices.
            min_x_idx = max(int(math.floor(poly['min_x'] / scale_factor)), 0)
            min_y_idx = max(int(math.floor(poly['min_y'] / scale_factor)), 0)
            max_x_idx = min(int(math.ceil(poly['max_x'] / scale_factor)), W)
            max_y_idx = min(int(math.ceil(poly['max_y'] / scale_factor)), H)
            mask[min_y_idx:max_y_idx, min_x_idx:max_x_idx] = True
        return mask

    # --- Apply mask for all_polygons ---
    if all_polygons:
        mask_all = torch.zeros((H, W), dtype=torch.bool, device=prob_vol.device)
        # If all_polygons is a dict, flatten the list of polygon dicts.
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

    # --- Optionally, apply mask for room_polygons ---
    if room_polygons:
        mask_room = torch.zeros((H, W), dtype=torch.bool, device=prob_vol.device)
        for poly in room_polygons:
            mask_room = apply_polygon(poly, mask_room)
        mask_room_expanded = mask_room.unsqueeze(2)  # shape (H, W, 1)
        prob_vol = prob_vol * mask_room_expanded

    # Compute per-pixel max probability (over orientations) and the corresponding orientation indices.
    prob_dist, orientations = torch.max(prob_vol, dim=2)  # shapes: (H, W)

    # Find indices of the pixel(s) with maximum probability.
    pred_y_indices, pred_x_indices = torch.where(prob_dist == prob_dist.max())
    sampled_index = torch.randint(0, pred_y_indices.shape[0], (1,))
    
    pred_y = pred_y_indices[sampled_index]
    pred_x = pred_x_indices[sampled_index]
    
    # Get the orientation index at the chosen pixel and convert it to radians.
    orn_idx = orientations[pred_y, pred_x]
    # Assuming 36 bins (each representing 10° or 2pi/36 radians)
    orn = orn_idx / 36 * 2 * torch.pi
    # Concatenate x, y, and orientation to form the predicted state.
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

def get_ray_from_semantics(semantics, V=7, dv=10, a0=None, F_W=1/np.tan(0.698132)/2):
    """
    Shoot the rays to the semantics, from left to right
    Input:
        V: number of rays
        dv: angle between two neighboring rays (in degrees)
        a0: camera intrinsic (center of the image by default)
        F/W: focal length / image width ratio
    Output:
        rays: interpolated rays for semantics
    """
    W = semantics.shape[0]
    angles = (np.arange(0, V) - np.arange(0, V).mean()) * dv / 180 * np.pi

    if a0 is None:
        # assume a0 is in the middle of the image
        w = np.tan(angles) * W * F_W + (W - 1) / 2  # desired width, left to right
    else:
        w = np.tan(angles) * W * F_W + a0  # left to right
    # w=np.linspace(0, 39, 21)
    # w = np.clip(w, 0, W-1)
    # Interpolating semantics across the desired angles
    interp_semantics = griddata(np.arange(W).reshape(-1, 1), semantics, w, method="linear", fill_value=0)
    rays = np.round(interp_semantics).astype(int)  # Convert interpolated values to nearest integer to get the semantics

    return rays

from collections import Counter

def get_ray_from_semantics_v2(original_rays, angle_between_rays=80/40, desired_ray_count=9, window_size=1):
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



def get_ray_from_semantics_soft(
    original_probs: torch.Tensor,
    angle_between_rays: float = 80/40,
    desired_ray_count: int = 9,
    window_size: int = 1,
) -> torch.Tensor:
    """
    original_probs: [num_rays, num_classes] probability tensor
    angle_between_rays: degrees per original ray
    desired_ray_count: how many output rays you want
    window_size: how many neighbors to include on each side
    returns: [desired_ray_count, num_classes] tensor of mean probabilities
    """
    num_rays, num_classes = original_probs.shape
    center_index = num_rays // 2
    desired_angle_step = 10.0

    out = original_probs.new_zeros((desired_ray_count, num_classes))

    for i in range(desired_ray_count):
        # angle offset from center
        desired_angle = (i - desired_ray_count // 2) * desired_angle_step
        idx_offset = desired_angle / angle_between_rays
        idx = int(round(center_index + idx_offset))
        idx = max(0, min(idx, num_rays - 1))

        # window boundaries
        start = max(idx - window_size, 0)
        end   = min(idx + window_size + 1, num_rays)

        # average the probability vectors in the window
        window_probs = original_probs[start:end]         # [window_length, num_classes]
        out[i] = window_probs.mean(dim=0)                # [num_classes]

    return out