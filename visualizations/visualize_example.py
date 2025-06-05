import os
import torch
import gzip
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import re
from utils.localization_utils import (
    finalize_localization,
)
import shutil  # Import shutil for file copying

def extract_camera_number(file_name):
    match = re.search(r'\d+', file_name)
    if match:
        return int(match.group())
    return float('inf')

def load_compressed_tensor(file_path):
    with gzip.open(file_path, 'rb') as f:
        tensor = torch.load(f)
    return tensor

def read_poses(file_path):
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            x, y, orientation = map(float, line.strip().split())
            poses.append((x, y, orientation))
    return poses

def plot_single_map(prob_vol, ref_pose_map, resolution, output_path, device='cpu'):
    """
    Plots a heatmap using the original probability volume and calculates accuracy metrics.
    """
    prob_vol = torch.tensor(prob_vol, device=device)
    prob_vol_pred, prob_dist_pred, orientations_pred, pose_pred = finalize_localization(prob_vol,[])
    
    # Generate the probability map (2D projection of prob_vol)
    prob_map = prob_dist_pred.astype(np.float32)  # Convert to NumPy array for visualization
    prob_map = np.flipud(cv2.resize(prob_map, (prob_map.shape[1] * 10, prob_map.shape[0] * 10), interpolation=cv2.INTER_LINEAR))
    H, W = prob_map.shape

    plt.figure(figsize=(8,8))
    plt.imshow(prob_map, extent=[0, W, 0, H], cmap='jet', alpha=0.8, origin='lower')  # "jet" colormap

    acc = acc_orn = 0  # Initialize metrics
    if ref_pose_map is not None:
        # Convert predicted pose to appropriate scale
        pose_pred = torch.tensor(pose_pred, device=device, dtype=torch.float32)
        pose_pred[:2] = pose_pred[:2] / 10  # Scale poses to match ground truth

        # Calculate accuracy metrics
        acc = torch.norm(pose_pred[:2] - torch.tensor(ref_pose_map[:2], device=device), p=2).item()
        acc_orn = ((pose_pred[2] - torch.tensor(ref_pose_map[2], device=device)) % (2 * np.pi)).item()
        acc_orn = min(acc_orn, 2 * np.pi - acc_orn) / np.pi * 180

        # Plot Ground Truth (GT) pose as an arrow
        gt_pose_x = ref_pose_map[0] * (1 / resolution) * 10
        gt_pose_y = H - ref_pose_map[1] * (1 / resolution) * 10
        gt_dx = np.cos(ref_pose_map[2]) * 30
        gt_dy = np.sin(ref_pose_map[2]) * 30
        plt.arrow(
            gt_pose_x, gt_pose_y, gt_dx, gt_dy,
            width=30,head_width=80, head_length=60, fc='magenta', ec='black', alpha=1  
        )

        # Plot predicted pose as an arrow
        pred_pose_x = pose_pred[0].item() * (1 / resolution) * 10
        pred_pose_y = H - pose_pred[1].item() * (1 / resolution) * 10
        pred_dx = np.cos(pose_pred[2].item()) * 30
        pred_dy = np.sin(pose_pred[2].item()) * 30
        plt.arrow(
            pred_pose_x, pred_pose_y, pred_dx, pred_dy,
            width=30,head_width=80, head_length=60, fc='white', ec='black', alpha=1 
        )

    # Save the plot without additional headers
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return acc, acc_orn  # Return accuracy metrics



def create_scale_bar(save_dir, example_heatmap):
    """
    Creates a probability scale bar image without numeric scale, only 'Low' and 'High' outside the bar.
    """
    scale_bar_path = os.path.join(save_dir, "scale_bar.png")
    gradient = np.linspace(0, 1, example_heatmap.shape[1]).reshape(1, -1)
    gradient = np.vstack([gradient] * 50)  # Create a scale bar

    fig, ax = plt.subplots(figsize=(5, 1.4))
    im = ax.imshow(gradient, aspect="auto", cmap="jet", origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])

    # Position texts outside the bar
    # "Low" to the left, "High" to the right
    # We'll slightly adjust the plot limits to allow space
    ax.set_xlim(-0.001*example_heatmap.shape[1], 1.0001*example_heatmap.shape[1])
    # The image is from 0 to example_heatmap.shape[1]-1 in x-axis. 
    # We'll place Low at a negative x and High at beyond the length.
    low_x = - (0.05 * example_heatmap.shape[1])
    high_x = (1.01 * example_heatmap.shape[1])
    mid_y = 25  # middle in the vertical direction (since height=50)
    ax.text(low_x, mid_y, 'Low', va='center', ha='right', fontsize=20)
    ax.text(high_x, mid_y, 'High', va='center', ha='left', fontsize=20)

    plt.tight_layout()
    plt.savefig(scale_bar_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    return scale_bar_path

def create_legend_arrows(save_dir):
    """
    Creates an image with two arrows facing right (stacked vertically) and text below each arrow.
    Uses plt.arrow so that the legend arrows match the plot arrows.
    The arrows are now wider and shorter.
    """
    legend_arrows_path = os.path.join(save_dir, "legend_arrows.png")

    plt.figure(figsize=(3, 1.4))
    
    # Ground Truth Arrow (Green with black outline)
    # plt.arrow(
    #     0.1, 0.6,    # starting point (x, y)
    #     0.2, 0,       # dx, dy (arrow extends from x=0.1 to 0.3, making it shorter than before)
    #     width=0.1,   # increased width for a thicker (wider) arrow shaft
    #     head_width=0.24,  # larger arrowhead to match the increased width
    #     head_length=0.08, # shorter head length, keeping proportion with the overall shorter arrow
    #     fc='green',
    #     ec='black',
    #     length_includes_head=True
    # )
    # Text label next to the Ground Truth arrow
    plt.text(0.32, 0.6, "Ground Truth", color='black', fontsize=16,
             ha='left', va='center')

    # Predicted Arrow (White with black outline)
    # plt.arrow(
    #     0.1, 0.3,    # starting point (x, y)
    #     0.2, 0,       # dx, dy (arrow extends from x=0.1 to 0.3)
    #     width=0.1,   # wider arrow shaft
    #     head_width=0.24,
    #     head_length=0.08,
    #     fc='white',
    #     ec='black',
    #     length_includes_head=True
    # )
    # Text label next to the Predicted arrow
    plt.text(0.32, 0.3, "Predicted", color='black', fontsize=16,
             ha='left', va='center')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(legend_arrows_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return legend_arrows_path


def generate_maps_and_latex(scene_dir, prob_vol_dir, save_dir, poses, num_cameras=5, device='cpu'):
    """
    Generates heatmaps and creates a LaTeX table with accuracy metrics.
    """
    scene_save_dir = os.path.join(save_dir, f"figures/examples/{os.path.basename(scene_dir)}")
    os.makedirs(scene_save_dir, exist_ok=True)

    # Ensure floorplan_semantic.png is present
    floorplan_src = os.path.join(scene_dir, "floorplan_semantic.png")
    floorplan_dest = os.path.join(scene_save_dir, "floorplan_semantic.png")
    if not os.path.exists(floorplan_src):
        raise FileNotFoundError(f"floorplan_semantic.png not found in {scene_dir}")
    if not os.path.exists(floorplan_dest):
        shutil.copy(floorplan_src, floorplan_dest)

    heatmaps = []
    for file_name in os.listdir(prob_vol_dir):
        if file_name.endswith("pred_depth_prob_vol.pt.gz"):
            file_path = os.path.join(prob_vol_dir, file_name)
            depth_prob_vol = load_compressed_tensor(file_path)

            semantic_file_name = file_name.replace("pred_depth_prob_vol.pt.gz", "pred_semantic_prob_vol.pt.gz")
            semantic_file_path = os.path.join(prob_vol_dir, semantic_file_name)

            if os.path.exists(semantic_file_path):
                semantic_prob_vol = load_compressed_tensor(semantic_file_path)

                combined_prob_vol = 0.5 * depth_prob_vol + 0.5 * semantic_prob_vol
                heatmaps.append((file_name, depth_prob_vol, semantic_prob_vol, combined_prob_vol))

    heatmaps.sort(key=lambda x: extract_camera_number(x[0]))
    heatmaps = heatmaps[:num_cameras]

    # Create the scale bar using one of the heatmaps
    example_heatmap = heatmaps[0][1] if heatmaps else np.random.rand(10, 10)  # Use depth_prob_vol
    scale_bar_path = create_scale_bar(scene_save_dir, example_heatmap)

    # Create the arrows image
    legend_arrows_path = create_legend_arrows(scene_save_dir)

    output_images = []
    metrics = []  # Store accuracy metrics for each plot
    camera_images = []
    for i, (file_name, depth_prob_vol, semantic_prob_vol, combined_prob_vol) in enumerate(heatmaps):
        ref_pose_map = poses[i] if i < len(poses) else None

        # Extract camera number
        camera_number = extract_camera_number(file_name)
        camera_path = os.path.join(scene_dir, "rgb", f"{camera_number}.png")

        # Generate plots and calculate metrics
        depth_map_path = os.path.join(scene_save_dir, f"depth_map_{i}.png")
        depth_acc, depth_acc_orn = plot_single_map(depth_prob_vol, ref_pose_map, 0.1, depth_map_path, device)

        semantic_map_path = os.path.join(scene_save_dir, f"semantic_map_{i}.png")
        semantic_acc, semantic_acc_orn = plot_single_map(semantic_prob_vol, ref_pose_map, 0.1, semantic_map_path, device)

        combined_map_path = os.path.join(scene_save_dir, f"combined_map_{i}.png")
        combined_acc, combined_acc_orn = plot_single_map(combined_prob_vol, ref_pose_map, 0.1, combined_map_path, device)

        metrics.append([
            (depth_acc, depth_acc_orn),
            (semantic_acc, semantic_acc_orn),
            (combined_acc, combined_acc_orn)
        ])
        output_images.append((depth_map_path, semantic_map_path, combined_map_path))
        camera_images.append(camera_path)

    latex_content = r"""
\documentclass[a4paper]{article}
\usepackage{graphicx}
\usepackage[margin=0.5in]{geometry}
\usepackage{amsmath,amssymb}
\begin{document}

\begin{center}
\small
\begin{tabular}{ccccc}
\hline
\textbf{Camera} & \textbf{Floor Plan} & \textbf{Depth Map} & \textbf{Semantic Map} & \textbf{Combined Map} \\
\hline
"""

    floorplan_rel = f"figures/examples/{os.path.basename(scene_dir)}/floorplan_semantic.png"

    for i, (depth_map_path, semantic_map_path, combined_map_path) in enumerate(output_images):
        depth_acc, depth_acc_orn = metrics[i][0]
        semantic_acc, semantic_acc_orn = metrics[i][1]
        combined_acc, combined_acc_orn = metrics[i][2]

        depth_info = f"({depth_acc:.2f}m, {depth_acc_orn:.0f}^\circ)"
        semantic_info = f"({semantic_acc:.2f}m, {semantic_acc_orn:.0f}^\circ)"
        combined_info = f"({combined_acc:.2f}m, {combined_acc_orn:.0f}^\circ)"

        camera_image_dest = os.path.join(scene_save_dir, f"camera_{i}.png")
        if not os.path.exists(camera_image_dest):
            shutil.copy(camera_images[i], camera_image_dest)
        camera_rel = f"figures/examples/{os.path.basename(scene_dir)}/camera_{i}.png"

        latex_content += f"""
\\includegraphics[width=0.25\\textwidth]{{{camera_rel}}} &
\\includegraphics[width=0.15\\textwidth]{{{floorplan_rel}}} &
\\includegraphics[width=0.15\\textwidth]{{figures/examples/{os.path.basename(scene_dir)}/{os.path.basename(depth_map_path)}}} &
\\includegraphics[width=0.15\\textwidth]{{figures/examples/{os.path.basename(scene_dir)}/{os.path.basename(semantic_map_path)}}} &
\\includegraphics[width=0.15\\textwidth]{{figures/examples/{os.path.basename(scene_dir)}/{os.path.basename(combined_map_path)}}} \\\\

& & {depth_info} & {semantic_info} & {combined_info} \\\\
\\hline
"""

    # Add legend and scale bar right below the last image row
    latex_content += r"""
\multicolumn{5}{c}{
\includegraphics[width=0.2\textwidth]{figures/examples/""" + os.path.basename(scene_dir) + r"""/scale_bar.png}
\hspace{1em}
\includegraphics[width=0.2\textwidth]{figures/examples/""" + os.path.basename(scene_dir) + r"""/legend_arrows.png}
} \\
\end{tabular}
\end{center}

\end{document}
"""

    latex_file = os.path.join(scene_save_dir, "output_table.tex")
    with open(latex_file, "w") as f:
        f.write(latex_content)


# Example usage (adapt paths if needed)
# scene_name = "scene_1068_floor_01"
# save_dir = "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/results/final_results/zind/visualtizations/saved_maps"
# scene_dir = f"/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/zind/zind_data_set_80_fov/{scene_name}"
# prob_vol_dir = f"/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/zind/prob_vol/{scene_name}"
scene_name = "scene_3358"
save_dir = "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/results/final_results/visualtizations/saved_maps"
scene_dir = f"/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/test_data_set_full/structured3d_perspective_full/{scene_name}"
prob_vol_dir = f"/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/test_data_set_full/prob_vols/{scene_name}"
poses_file = os.path.join(scene_dir, "poses.txt")
poses = read_poses(poses_file)
generate_maps_and_latex(scene_dir, prob_vol_dir, save_dir, poses, num_cameras=20, device='cuda:1' if torch.cuda.is_available() else 'cpu')
