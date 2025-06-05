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
    finalize_localization_acc_only
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
    prob_vol_pred, prob_dist_pred, orientations_pred, pose_pred = finalize_localization(prob_vol)
    
    # Generate the probability map (2D projection of prob_vol)
    prob_map = prob_dist_pred.astype(np.float32)  # Convert to NumPy array for visualization
    prob_map = np.flipud(cv2.resize(prob_map, (prob_map.shape[1] * 10, prob_map.shape[0] * 10), interpolation=cv2.INTER_LINEAR))
    H, W = prob_map.shape

    plt.figure(figsize=(3,3))
    plt.imshow(prob_map, extent=[0, W, 0, H], cmap='viridis', alpha=0.6, origin='lower')

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
            head_width=15, head_length=10, fc='green', ec='green', alpha=0.8
        )

        # Plot predicted pose as an arrow
        pred_pose_x = pose_pred[0].item() * (1 / resolution) * 10
        pred_pose_y = H - pose_pred[1].item() * (1 / resolution) * 10
        pred_dx = np.cos(pose_pred[2].item()) * 30
        pred_dy = np.sin(pose_pred[2].item()) * 30
        plt.arrow(
            pred_pose_x, pred_pose_y, pred_dx, pred_dy,
            head_width=15, head_length=10, fc='blue', ec='blue', alpha=0.8
        )

    # Save the plot without additional headers
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return acc, acc_orn  # Return accuracy metrics

def create_scale_bar(save_dir, example_heatmap):
    """
    Creates a probability scale bar image without numeric scale, only 'Low' and 'High' text.
    """
    scale_bar_path = os.path.join(save_dir, "scale_bar.png")
    gradient = np.linspace(0, 1, example_heatmap.shape[1]).reshape(1, -1)
    gradient = np.vstack([gradient] * 50)  # Create a scale bar

    fig, ax = plt.subplots(figsize=(2, 0.5))
    ax.imshow(gradient, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])

    # Add "Low" and "High" at the ends, no numeric scale
    ax.text(0.0, 0.5, 'Low', va='center', ha='left', fontsize=8, transform=ax.transAxes)
    ax.text(1.0, 0.5, 'High', va='center', ha='right', fontsize=8, transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(scale_bar_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    return scale_bar_path

def create_legend_arrows(save_dir):
    """
    Creates a legend for the arrows, stacked vertically and both facing right,
    with the legend to the right of them.
    """
    legend_arrows_path = os.path.join(save_dir, "legend_arrows.png")

    plt.figure(figsize=(4,2))
    # Ground Truth Arrow, facing right
    plt.quiver(
        [0.2], [0.6], [0.3], [0],
        angles='xy', scale_units='xy', scale=1, color='green', label='Ground Truth Pose'
    )
    # Predicted Arrow, facing right, below the GT
    plt.quiver(
        [0.2], [0.4], [0.3], [0],
        angles='xy', scale_units='xy', scale=1, color='blue', label='Predicted Pose'
    )

    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.axis('off')
    # Place legend to the right (not overlapping arrows)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

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

    # Create legend arrows
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

    # Generate LaTeX table
    # Columns: Camera Image | Floor Plan | Depth Map | Semantic Map | Combined Map
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

    # Assuming a floorplan_semantic.png exists in the same folder
    floorplan_path = os.path.join(scene_save_dir, "floorplan_semantic.png")
    floorplan_rel = f"figures/examples/{os.path.basename(scene_dir)}/floorplan_semantic.png"

    for i, (depth_map_path, semantic_map_path, combined_map_path) in enumerate(output_images):
        depth_acc, depth_acc_orn = metrics[i][0]
        semantic_acc, semantic_acc_orn = metrics[i][1]
        combined_acc, combined_acc_orn = metrics[i][2]

        # Convert to desired format: (2.3m, 30°)
        depth_info = f"({depth_acc:.2f}m, {depth_acc_orn:.0f}^\circ)"
        semantic_info = f"({semantic_acc:.2f}m, {semantic_acc_orn:.0f}^\circ)"
        combined_info = f"({combined_acc:.2f}m, {combined_acc_orn:.0f}^\circ)"

        # Copy the camera image into scene_save_dir for LaTeX referencing
        camera_image_dest = os.path.join(scene_save_dir, f"camera_{i}.png")
        if not os.path.exists(camera_image_dest):
            shutil.copy(camera_images[i], camera_image_dest)

        camera_rel = f"figures/examples/{os.path.basename(scene_dir)}/camera_{i}.png"

        latex_content += f"""
\\includegraphics[width=0.15\\textwidth]{{{camera_rel}}} &
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
scene_name = "scene_3250"
save_dir = "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/results/final_results/visualtizations/saved_maps"
scene_dir = f"/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/test_data_set_full/structured3d_perspective_full/{scene_name}"
prob_vol_dir = f"/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/test_data_set_full/prob_vols/{scene_name}"
poses_file = os.path.join(scene_dir, "poses.txt")
poses = read_poses(poses_file)
generate_maps_and_latex(scene_dir, prob_vol_dir, save_dir, poses, num_cameras=2, device='cuda' if torch.cuda.is_available() else 'cpu')

