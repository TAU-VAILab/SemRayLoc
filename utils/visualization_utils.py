# utils/visualization_utils.py

import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# utils/visualization_utils.py

import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def plot_dict_relationship(data_dict, save_path=None, weight_key='', xlabel='Max Probability', ylabel='Distance (acc)', title='Relationship Between Max Probability and Distance'):
    """
    Plots the relationship between the keys and values in a dictionary and optionally saves the plot.

    Parameters:
    - data_dict (dict): A dictionary where keys are scalar values and values are lists of numbers.
    - save_path (str): Directory to save the plot. If None, the plot is displayed but not saved.
    - weight_key (str): A string identifying the weight key to append to the plot title and file name.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - title (str): Title of the plot.
    """
    # Prepare data for plotting, filtering out distances > 1 meter
    keys = []
    values = []
    for key, value_list in data_dict.items():
        for value in value_list:  # Unpack lists of values
            # if value <= 1:  # Only include distances <= 1 meter
            keys.append(key)
            values.append(value)
    
    # Convert data to numpy arrays for calculations
    keys = np.array(keys)
    values = np.array(values)

    # Calculate correlation
    if len(keys) > 1 and len(values) > 1:  # Ensure enough data points for correlation
        correlation, _ = pearsonr(keys, values)
    else:
        correlation = np.nan  # Not enough data points

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.scatter(keys, values, alpha=0.7, label='Data Points')
    
    # Add a trend line (linear regression)
    if len(keys) > 1 and len(values) > 1:
        coeffs = np.polyfit(keys, values, deg=1)  # Linear fit
        trendline = np.poly1d(coeffs)
        plt.plot(keys, trendline(keys), color='red', linestyle='--', label='Trend Line')
    
    # Add correlation to the plot
    plt.text(0.05, 0.95, f"Correlation: {correlation:.2f}", fontsize=12, transform=plt.gca().transAxes, verticalalignment='top')
    
    # Update the title to include weights
    full_title = f"{title} (Weights: {weight_key})"
    plt.title(full_title)
    
    # Labels and legend
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))  # Place legend on the right side
    plt.grid(True)
    
    # Save or show the plot
    if save_path:
        os.makedirs(save_path, exist_ok=True)  # Ensure directory exists
        save_file_name = f"{full_title.replace(' ', '_')}.png"
        full_save_path = os.path.join(save_path, save_file_name)
        plt.savefig(full_save_path, bbox_inches='tight')
        print(f"Plot saved to {full_save_path}")
    else:
        plt.show()
    plt.close()


def plot_prob_dist(prob_dist, resolution: float = 0.1, save_path: str = 'temp_figs', file_name: str = 'prob_dist_map.png', occ=None, pose_pred=None, ref_pose_map=None, plot_type="combined", acc=None, acc_orn=None, acc_only=False, prob_vol_gt=None):
    """
    Plots the probability distribution map overlaid on the occupancy map (floorplan) and saves it to a specified folder.
    If prob_vol_gt is provided, it plots the predicted and ground truth maps side by side.

    Args:
        prob_dist: (H, W) probability distribution tensor or NumPy array.
        resolution: resolution of the map in meters (default is 0.1m).
        save_path: folder where the figure will be saved (default is 'temp_figs').
        file_name: name of the file to save the figure as (default is 'prob_dist_map.png').
        occ: occupancy map (floorplan) as a NumPy array.
        pose_pred: predicted pose [x, y, orientation].
        ref_pose_map: actual pose [x, y, orientation].
        plot_type: type of plot (depth, semantic, or combined).
        acc: accuracy metric (optional).
        acc_orn: orientation accuracy metric (optional).
        acc_only: flag to only show position markers without arrows.
        prob_vol_gt: ground truth probability distribution (optional).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    import os

    # Helper function to plot an individual probability map
    def plot_single_map(prob_map, occ, poses, H, W, resolution, title, arrow_colors):
        fig, ax = plt.subplots(figsize=(8, 6))
        if occ is not None:
            occ_flipped = np.flipud(occ)
            occ_mask = occ_flipped < 250  # Keep pixels that are not white
            occ_colored = np.ma.masked_where(~occ_mask, occ_flipped)
            ax.imshow(occ_colored, cmap='gray', extent=[0, W * resolution, 0, H * resolution], origin='lower', alpha=0.6)
        ax.imshow(prob_map, extent=[0, W * resolution, 0, H * resolution], cmap='jet', alpha=0.8, origin='lower')

        if poses is not None:
            arrow_length = 4.0  # Adjust length of the orientation arrow
            arrow_width = 1.6   # Arrow width

            for pose, color in zip(poses, arrow_colors):
                pos_x = pose[0] * (1 / resolution)
                pos_y = H * resolution - pose[1] * (1 / resolution)
                
                if len(pose) > 2:
                    ax.arrow(pos_x, pos_y,
                             arrow_length * np.cos(pose[2]), arrow_length * np.sin(pose[2]),
                             width=4,head_width=8, head_length=6, fc=color, ec='black', alpha=1, label=f'{title} Pose')
                else:
                    ax.plot(pos_x, pos_y, marker='o', color=color, markersize=10, label=f'{title} Pose')

        # Add the title and remove the axis
        # ax.set_title(title, fontsize=14)
        ax.axis('off')
        return fig, ax

    # Convert torch tensor to NumPy if necessary
    if isinstance(prob_dist, torch.Tensor):
        prob_dist = prob_dist.detach().cpu().numpy()
    if prob_vol_gt is not None and isinstance(prob_vol_gt, torch.Tensor):
        prob_vol_gt = prob_vol_gt.detach().cpu().numpy()

    # Convert pose_pred and ref_pose_map to NumPy if they are tensors
    if isinstance(pose_pred, torch.Tensor):
        pose_pred = pose_pred.detach().cpu().numpy()
    if isinstance(ref_pose_map, torch.Tensor):
        ref_pose_map = ref_pose_map.detach().cpu().numpy()

    # Resize probability distributions to match the occupancy map shape
    if occ is not None:
        prob_dist_resized = cv2.resize(prob_dist, (occ.shape[1], occ.shape[0]), interpolation=cv2.INTER_LINEAR)
        if prob_vol_gt is not None:
            prob_vol_gt_resized = cv2.resize(prob_vol_gt, (occ.shape[1], occ.shape[0]), interpolation=cv2.INTER_LINEAR)
    else:
        prob_dist_resized = cv2.resize(prob_dist, (prob_dist.shape[1]*10, prob_dist.shape[0]*10), interpolation=cv2.INTER_LINEAR)
        if prob_vol_gt is not None:
            prob_vol_gt_resized = cv2.resize(prob_vol_gt, (prob_dist_resized.shape[1], prob_dist_resized.shape[0]), interpolation=cv2.INTER_LINEAR)

    prob_dist_resized = np.flipud(prob_dist_resized)
    if prob_vol_gt is not None:
        prob_vol_gt_resized = np.flipud(prob_vol_gt_resized)

    H, W = prob_dist_resized.shape

    if prob_vol_gt is not None:
        # Plot the predicted probability map with both predicted and ground truth locations
        fig_pred, _ = plot_single_map(
            prob_dist_resized, occ, [pose_pred, ref_pose_map], H, W, resolution, 'Prediction', arrow_colors=['white', 'magenta']
        )

        # Plot the ground truth probability map with only the ground truth location
        fig_gt, _ = plot_single_map(
            prob_vol_gt_resized, occ, [ref_pose_map], H, W, resolution, 'Ground Truth', arrow_colors=['green']
        )

        # Create a combined image with both figures side by side
        fig_combined, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig_combined.subplots_adjust(wspace=0, hspace=0)  # Reduce spacing between plots

        # Plot the images in each axis
        for ax, fig in zip(axes, [fig_pred, fig_gt]):
            canvas = fig.canvas
            canvas.draw()
            width, height = canvas.get_width_height()
            img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
            ax.imshow(img)
            ax.axis('off')
            plt.close(fig)  # Close individual figures to prevent display

        # Add orientation and accuracy text
        if acc is not None and acc_orn is not None:
            axes[0].text(0.5, -0.1, f'Acc: {acc:.2f}m\nAcc Orn: {acc_orn:.2f}°', 
                         transform=axes[0].transAxes, fontsize=12, ha='center', va='top')

        # Ensure the save directory exists
        os.makedirs(save_path, exist_ok=True)

        # Save the combined image
        save_full_path = os.path.join(save_path, file_name)
        fig_combined.savefig(save_full_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig_combined)

        print(f"Figure saved to {save_full_path}")

    else:
        # Plot and save the single predicted probability map if prob_vol_gt is not provided
        fig, _ = plot_single_map(prob_dist_resized, occ, [ref_pose_map, pose_pred], H, W, resolution, 'Prediction', arrow_colors=['magenta', 'white'])

        os.makedirs(save_path, exist_ok=True)
        save_full_path = os.path.join(save_path, file_name)
        fig.savefig(save_full_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        print(f"Figure saved to {save_full_path}")

def plot_prob_dist_comparison(cond_map, pred_map, gt_map, resolution: float = 0.1, save_path: str = 'temp_figs', file_name: str = 'prob_dist_comparison.png', occ=None, pose_pred_cond=None, pose_pred=None, pose_pred_gt=None, ref_pose_map=None, acc_cond=None, acc_pred=None, acc_gt=None, model_type= ""):
    """
    Plots the conditional map, predicted map, and ground truth map side by side with individual accuracy metrics.
    Each image shows the predicted pose by this map in red and the reference pose in green.

    Args:
        cond_map: (H, W) conditional map used as input.
        pred_map: (H, W) predicted map output from the model.
        gt_map: (H, W) ground truth probability map.
        resolution: resolution of the map in meters (default is 0.1m).
        save_path: folder where the figure will be saved (default is 'temp_figs').
        file_name: name of the file to save the figure as (default is 'prob_dist_comparison.png').
        occ: occupancy map (floorplan) as a NumPy array.
        pose_pred_cond: predicted pose from the conditional map [x, y, orientation].
        pose_pred: predicted pose from the predicted map [x, y, orientation].
        pose_pred_gt: predicted pose from the ground truth map [x, y, orientation].
        ref_pose_map: actual pose [x, y, orientation].
        acc_cond: accuracy metric for the conditional map.
        acc_pred: accuracy metric for the predicted map.
        acc_gt: accuracy metric for the ground truth map.
    """

    def plot_single_map(ax, prob_map, occ, poses, resolution, title, arrow_colors, acc):
        if occ is not None:
            occ_flipped = np.flipud(occ)
            occ_mask = occ_flipped < 250  # Keep pixels that are not white
            occ_colored = np.ma.masked_where(~occ_mask, occ_flipped)
            ax.imshow(occ_colored, cmap='gray', extent=[0, W * resolution, 0, H * resolution], origin='lower', alpha=0.6)
        ax.imshow(prob_map, extent=[0, W * resolution, 0, H * resolution], cmap='viridis', alpha=0.6, origin='lower')

        if poses is not None:
            arrow_length = 2.0  # Adjust length of the orientation arrow
            arrow_width = 0.8   # Arrow width

            for pose, color in zip(poses, arrow_colors):
                pos_x = pose[0] * (1 / resolution)
                pos_y = H * resolution - pose[1] * (1 / resolution)
                
                if len(pose) > 2:
                    ax.arrow(pos_x, pos_y,
                             arrow_length * np.cos(pose[2]), arrow_length * np.sin(pose[2]),
                             color=color, width=arrow_width, head_width=2, head_length=2, label=f'{title} Pose')
                else:
                    marker_size = 8 if color == 'red' else 10  # Reduce the size of the red dot
                    ax.plot(pos_x, pos_y, marker='o', color=color, markersize=marker_size, label=f'{title} Pose')

        ax.set_title(f'{title}\nAcc: {acc:.2f}m', fontsize=14)
        ax.axis('off')

    # Convert inputs to NumPy arrays if necessary
    def convert_tensor_to_numpy(tensor):
        return tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor

    cond_map = convert_tensor_to_numpy(cond_map)
    pred_map = convert_tensor_to_numpy(pred_map)
    gt_map = convert_tensor_to_numpy(gt_map)
    pose_pred_cond = convert_tensor_to_numpy(pose_pred_cond)
    pose_pred = convert_tensor_to_numpy(pose_pred)
    pose_pred_gt = convert_tensor_to_numpy(pose_pred_gt)
    ref_pose_map = convert_tensor_to_numpy(ref_pose_map)

    # Resize maps to match occupancy map if provided
    if occ is not None:
        def resize_and_flip_map(map):
            return np.flipud(cv2.resize(map, (occ.shape[1], occ.shape[0]), interpolation=cv2.INTER_LINEAR))
        
        cond_map_resized = resize_and_flip_map(cond_map)
        pred_map_resized = resize_and_flip_map(pred_map)
        gt_map_resized = resize_and_flip_map(gt_map)
    else:
        def resize_and_flip_map(map):
            return np.flipud(cv2.resize(map, (cond_map.shape[1]*10, cond_map.shape[0]*10), interpolation=cv2.INTER_LINEAR))
        
        cond_map_resized = resize_and_flip_map(cond_map)
        pred_map_resized = resize_and_flip_map(pred_map)
        gt_map_resized = resize_and_flip_map(gt_map)

    H, W = cond_map_resized.shape

    # Create subplots for conditional map, predicted map, and ground truth map
    fig, axes = plt.subplots(1, 3, figsize=(20, 4))  # Reduced height for narrower plots

    # Plot conditional map
    plot_single_map(axes[0], cond_map_resized, occ, [pose_pred_cond, ref_pose_map], resolution, f'{model_type} Map', arrow_colors=['red', 'green'], acc=acc_cond)

    # Plot predicted map
    plot_single_map(axes[1], pred_map_resized, occ, [pose_pred, ref_pose_map], resolution, f'{model_type} after diffusion', arrow_colors=['red', 'green'], acc=acc_pred)

    # Plot ground truth map
    plot_single_map(axes[2], gt_map_resized, occ, [pose_pred_gt, ref_pose_map], resolution, f'{model_type} Ground Truth', arrow_colors=['red', 'green'], acc=acc_gt)

    # Add legend below the plots
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.1), fontsize=12)

    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Save the combined plot
    save_full_path = os.path.join(save_path, file_name)
    plt.savefig(save_full_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    print(f"Figure saved to {save_full_path}")
def plot_weight_pred_comparison(cond_map, pred_map, gt_map, gt_pred_map, w_depth, w_semantic,prob_vol_depth,prob_vol_semantic, resolution: float = 0.1, save_path: str = 'temp_figs', file_name: str = 'prob_dist_comparison.png', occ=None, pose_pred_cond=None, pose_pred=None, pose_pred_gt=None, pose_gt_with_weight_pred=None, ref_pose_map=None, acc_cond=None, acc_pred=None, acc_gt=None, model_type=""):
    def plot_single_map(ax, prob_map, occ, poses, resolution, title, arrow_colors, acc):
        if occ is not None:
            occ_flipped = np.flipud(occ)
            occ_mask = occ_flipped < 250  # Keep pixels that are not white
            occ_colored = np.ma.masked_where(~occ_mask, occ_flipped)
            ax.imshow(occ_colored, cmap='gray', extent=[0, W * resolution, 0, H * resolution], origin='lower', alpha=0.6)
        ax.imshow(prob_map, extent=[0, W * resolution, 0, H * resolution], cmap='viridis', alpha=0.6, origin='lower')

        if poses is not None:
            arrow_length = 2.0  # Adjust length of the orientation arrow
            arrow_width = 0.8   # Arrow width

            for pose, color in zip(poses, arrow_colors):
                pos_x = pose[0] * (1 / resolution)
                pos_y = H * resolution - pose[1] * (1 / resolution)
                
                if len(pose) > 2:
                    ax.arrow(pos_x, pos_y,
                             arrow_length * np.cos(pose[2]), arrow_length * np.sin(pose[2]),
                             color=color, width=arrow_width, head_width=2, head_length=2, label=f'{title} Pose')
                else:
                    marker_size = 8 if color == 'red' else 10  # Reduce the size of the red dot
                    ax.plot(pos_x, pos_y, marker='o', color=color, markersize=marker_size, label=f'{title} Pose')

        ax.set_title(f'{title}\nAcc: {acc:.2f}m', fontsize=14)
        ax.axis('off')

    # Convert inputs to NumPy arrays if necessary
    def convert_tensor_to_numpy(tensor):
        return tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor

    cond_map = convert_tensor_to_numpy(cond_map)
    pred_map = convert_tensor_to_numpy(pred_map)
    gt_map = convert_tensor_to_numpy(gt_map)
    gt_pred_map = convert_tensor_to_numpy(gt_pred_map)
    w_depth = convert_tensor_to_numpy(w_depth)
    w_semantic = convert_tensor_to_numpy(w_semantic)
    prob_vol_depth = convert_tensor_to_numpy(prob_vol_depth)
    prob_vol_semantic = convert_tensor_to_numpy(prob_vol_semantic)
    pose_pred_cond = convert_tensor_to_numpy(pose_pred_cond)
    pose_pred = convert_tensor_to_numpy(pose_pred)
    pose_pred_gt = convert_tensor_to_numpy(pose_pred_gt)
    pose_gt_with_weight_pred = convert_tensor_to_numpy(pose_gt_with_weight_pred)
    ref_pose_map = convert_tensor_to_numpy(ref_pose_map)

    # Resize maps to match occupancy map if provided
    if occ is not None:
        def resize_and_flip_map(map):
            return np.flipud(cv2.resize(map, (occ.shape[1], occ.shape[0]), interpolation=cv2.INTER_LINEAR))
    else:
        def resize_and_flip_map(map):
            return np.flipud(cv2.resize(map, (cond_map.shape[1]*10, cond_map.shape[0]*10), interpolation=cv2.INTER_LINEAR))
        
    cond_map_resized = resize_and_flip_map(cond_map)
    pred_map_resized = resize_and_flip_map(pred_map)
    gt_map_resized = resize_and_flip_map(gt_map)
    gt_pred_map_resized = resize_and_flip_map(gt_pred_map)
    w_depth_resized = resize_and_flip_map(w_depth)
    w_semantic_resized = resize_and_flip_map(w_semantic)
    prob_vol_depth_resized = resize_and_flip_map(prob_vol_depth)
    prob_vol_semantic_resized = resize_and_flip_map(prob_vol_semantic)

    H, W = cond_map_resized.shape

    # Create subplots for all maps
    fig, axes = plt.subplots(2, 4, figsize=(24, 8))  # Adjusted for eight plots (2 rows, 4 columns)

    # Plot conditional map
    plot_single_map(axes[0, 0], cond_map_resized, occ, [pose_pred_cond, ref_pose_map], resolution, f'{model_type} original_maps__weight-0.5', arrow_colors=['red', 'green'], acc=acc_cond)

    # Plot predicted map
    plot_single_map(axes[0, 1], pred_map_resized, occ, [pose_pred, ref_pose_map], resolution, f'{model_type} original_maps__weight-Pred', arrow_colors=['red', 'green'], acc=acc_pred)

    # Plot ground truth map
    plot_single_map(axes[0, 2], gt_map_resized, occ, [pose_pred_gt, ref_pose_map], resolution, f'{model_type} GT__weight-0.5', arrow_colors=['red', 'green'], acc=acc_gt)

    # Plot GT-predicted map
    plot_single_map(axes[0, 3], gt_pred_map_resized, occ, [pose_gt_with_weight_pred, ref_pose_map], resolution, f'{model_type} GT__weight-Pred', arrow_colors=['red', 'green'], acc=acc_gt)

    plot_single_map(axes[1, 0], prob_vol_depth_resized, occ, [pose_pred, ref_pose_map], resolution, f'{model_type} Depth original', arrow_colors=['red', 'green'], acc=0)
    plot_single_map(axes[1, 1], prob_vol_semantic_resized, occ, [pose_pred, ref_pose_map], resolution, f'{model_type} Semantic original', arrow_colors=['red', 'green'], acc=0)    
    plot_single_map(axes[1, 2], w_depth_resized, occ, [pose_pred, ref_pose_map], resolution, f'{model_type} Depth Weight (w_depth)', arrow_colors=['red', 'green'], acc=0)
    plot_single_map(axes[1, 3], w_semantic_resized, occ, [pose_pred, ref_pose_map], resolution, f'{model_type} Semantic Weight (w_semantic)', arrow_colors=['red', 'green'], acc=0)

    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Save the combined plot
    save_full_path = os.path.join(save_path, file_name)
    plt.savefig(save_full_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    print(f"Figure saved to {save_full_path}")
