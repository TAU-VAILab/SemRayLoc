import os
import json
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm
from attrdict import AttrDict
import torch.nn.functional as F

# Imports from our project – ensure your PYTHONPATH is set appropriately.
from modules.mono.depth_net_pl import depth_net_pl
from modules.semantic.semantic_net_pl import semantic_net_pl
from modules.semantic.semantic_net_pl_maskformer_small import semantic_net_pl_maskformer_small
from modules.semantic.semantic_mapper import ObjectType, object_to_color
from data_utils.data_utils import GridSeqDataset
from data_utils.prob_vol_data_utils import ProbVolDataset
from utils.localization_utils import (
    get_ray_from_depth, get_ray_from_semantics_v2, localize, finalize_localization
)
from utils.data_loader_helper import load_scene_data
from utils.visualization_utils import plot_dict_relationship
from scipy.ndimage import gaussian_filter1d
import seaborn as sns
import pickle

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
config = AttrDict({
    'dataset_dir': '/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/test_data_set_full/structured3d_perspective_full',
    'desdf_path': '/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/test_data_set_full/desdf',
    'log_dir_depth': '/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/modules/Final_wights/depth/final_depth_model_checkpoint.ckpt',
    'log_dir_semantic': '/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/modules/Final_wights/semantic/final_semantic_model_checkpoint.ckpt',
    'split_file': '/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/test_data_set_full/structured3d_perspective_full/split.yaml',
    'prob_vol_path': '/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/test_data_set_full/prob_vols',
    
    'results_dir': '/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/results/full/GT',
    'combined_prob_vols_net_path': '/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/logs/combined/expected_pose_pred/combined_prob_vols_net_type-expected_pose_pred_net_size-6k_dataset_size-medium_epochs-99/combined_net-epoch=58-loss-valid=7.16.ckpt',
    'combined_net_size': '6k',
    
    'L': 0,
    'D': 128,
    'd_min': 0.1,
    'd_max': 15.0,
    'd_hyp': -0.2,
    'F_W': 0.59587643422,
    'V': 7,
    'num_classes': 4,
    'prediction_type': 'combined',
    'use_ground_truth_depth': False,
    'use_ground_truth_semantic': False,
    'use_saved_prob_vol': False,
    'num_of_scenes': -1,
    'pad_to_max': True,
    'max_h': 1760,
    'max_w': 1760,
    'weight_combinations': [
        [1.0, 0.0],
        # [0, 1.0],
        [0.8, 0.2]
    ],    
    'use_semantic_mask2foremr': True,
    'semantic_mask2former_path': '/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/logs/mask2Former_semantics_with_tiny_small_version_no_augmentaion/semantic/semantic_net-epoch=19-loss-valid=0.34.ckpt'
})

# -------------------------------------------------------------------------
# Function to plot camera positions and rays
# -------------------------------------------------------------------------
def plot_camera_positions_and_rays(camera_positions, img, ray_data, output_path, resolution=0.01, dpi=100,
                                   position_key='semantic', camera_labels=None, camera_dot_colors=None,
                                   filename_suffix=None, object_color = None):
    """
    Plots selected camera positions and their corresponding rays on the provided image and saves the plot.
    
    Parameters:
        camera_positions: list of dictionaries with keys 'vx_semantic', 'vy_semantic', 'vx_walls', and 'vy_walls'.
        img: Image array to use as background.
        ray_data: JSON data containing rays for each camera.
        output_path: Directory path where the output image will be saved.
        resolution: Plot resolution.
        dpi: Dots per inch for the figure.
        position_key: 'semantic' or 'walls' to select which positions to plot.
        camera_labels: List of labels (numbers) corresponding to each camera.
        camera_dot_colors: List of colors for the camera dots.
        filename_suffix: Suffix for the output filename. If provided, the file will be named using this suffix.
    """
    img_height, img_width = img.shape[:2]
    fig, ax = plt.subplots(figsize=(img_width / dpi, img_height / dpi), dpi=dpi)
    # ax.imshow(img)

    # Plot each camera as a dot with a label.
    for i, camera_info in enumerate(camera_positions):
        if position_key == 'semantic':
            camera_x = camera_info['vx_semantic']
            camera_y = camera_info['vy_semantic']
        elif position_key == 'walls':
            camera_x = camera_info['vx_walls']
            camera_y = camera_info['vy_walls']
        else:
            raise ValueError(f"Invalid position_key: {position_key}. Expected 'semantic' or 'walls'.")
        
        dot_color = camera_dot_colors[i] if camera_dot_colors and i < len(camera_dot_colors) else 'blue'
        ax.plot(camera_x, camera_y, 'o', markersize=30, color=dot_color)
        
        label = camera_labels[i] if camera_labels and i < len(camera_labels) else str(i)
        # ax.text(camera_x + 2, camera_y, str(label), color='green', fontsize=16)

    # Plot each ray. The color is chosen via the mapping in object_to_color based on the ray's prediction_class.
    for k ,camera_data in enumerate(ray_data['cameras']):
        for j ,ray in enumerate(camera_data['rays']):
            if position_key == 'semantic':
                start_x = ray['start_position_semantic']['x']
                start_y = ray['start_position_semantic']['y']
            elif position_key == 'walls':
                start_x = ray['start_position_walls']['x']
                start_y = ray['start_position_walls']['y']
            else:
                raise ValueError(f"Invalid position_key: {position_key}. Expected 'semantic' or 'walls'.")
            
            end_x = ray['end_position_semantic']['x']
            end_y = ray['end_position_semantic']['y']
            if object_color:
                object_type = ObjectType(object_color[k][j])
            else:
                object_type = ObjectType(ray['prediction_class'])

            
    
                
            
            color = object_to_color.get(object_type, 'black')
            ax.plot([start_x, end_x], [start_y, end_y], color=color, lw=0.5)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.axis('equal')
    ax.axis('off')

    os.makedirs(output_path, exist_ok=True)
    if filename_suffix is None:
        output_image_path = os.path.join(output_path, f'camera_positions_with_rays_{position_key}.png')
    else:
        output_image_path = os.path.join(output_path, f'camera_positions_with_rays_{filename_suffix}.png')
    plt.savefig(output_image_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    print(f"Plot saved to: {output_image_path}")
    plt.close(fig)

# -------------------------------------------------------------------------
# Main execution
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Hardcoded paths for input data ---
    input_folder = "/home/yuvalg/projects/Semantic_Floor_plan_localization/data/structured3d_perspective/test_data_set_full/scene_0"
    output_folder = "/home/yuvalg/projects/Semantic_Floor_plan_localization/results/visualizations/rays/scene_0"

    # --- Load the semantic floorplan image ---
    semantic_img_path = os.path.join(input_folder, 'floorplan_semantic.png')
    if not os.path.exists(semantic_img_path):
        print(f"Semantic floorplan image not found at {semantic_img_path}")
        exit(1)
    img_semantic = plt.imread(semantic_img_path)

    # --- Load the raycast JSON data ---
    ray_data_path = os.path.join(input_folder, 'camera_rays.json')
    if not os.path.exists(ray_data_path):
        print(f"Camera rays JSON not found at {ray_data_path}")
        exit(1)
    with open(ray_data_path, 'r') as f:
        ray_data = json.load(f)

    # --- Extract camera positions ---
    # Each camera should provide positions for both the semantic floorplan and walls.
    camera_positions = []
    for cam in ray_data['cameras']:
        camera_positions.append({
            'vx_semantic': cam['camera_position_pixel_semantic']['x'],
            'vy_semantic': cam['camera_position_pixel_semantic']['y'],
            'vx_walls': cam['camera_position_pixel_walls']['x'],
            'vy_walls': cam['camera_position_pixel_walls']['y']
        })

    # --- Filter to only the selected cameras (by index) ---
    selected_indices = [9]
    selected_camera_positions = [camera_positions[i] for i in selected_indices if i < len(camera_positions)]
    selected_ray_data = {"cameras": [ray_data['cameras'][i] for i in selected_indices if i < len(ray_data['cameras'])]}

    unique_colors = ['#FF00FF', '#00FFFF', '#FFA500', '#00FF00']
    camera_labels = [selected_indices[i] for i in range(len(selected_indices))]

    # --- Plot and save the GT (ground truth) camera positions with rays ---
    plot_camera_positions_and_rays(selected_camera_positions, img_semantic, selected_ray_data, output_folder,
                                   position_key='semantic', camera_labels=camera_labels,
                                   camera_dot_colors=unique_colors, filename_suffix="GT")

    # # --- Load the semantic network and dataset for predicted labels ---
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # scene_names = ["scene_00000"]
    # test_set = GridSeqDataset(config.dataset_dir, scene_names, L=config.L)
    # semantic_net = semantic_net_pl_maskformer_small.load_from_checkpoint(
    #     config.semantic_mask2former_path, num_classes=config.num_classes
    # ).to(device)
    # semantic_net.eval()

    # # --- Compute predicted labels for each selected camera ---
    # # For each camera (by index in the test set), run the semantic net on its reference image and take the mode
    # # of the pixelwise predictions as the camera's predicted label.
    # predicted_labels = {}  # key: camera index, value: predicted label (int)
    # i=0
    # for idx in selected_indices:
    #     if idx >= len(test_set):
    #         print(f"Index {idx} out of range for test_set with length {len(test_set)}")
    #         continue
    #     data = test_set[idx]
    #     ref_img = data["ref_img"]
    #     ref_img_t = torch.tensor(ref_img, device=device).unsqueeze(0).float()
    #     with torch.no_grad():
    #         # The encoder returns (_, _, prob) and we assume prob is (1, num_classes, H, W)
    #         ref_img_t = torch.tensor(data["ref_img"], device=device).unsqueeze(0)
    #         _, _, prob = semantic_net.encoder(ref_img_t, None)
    #         sampled_indices_np = torch.multinomial(prob.squeeze(0), 1, replacement=True).squeeze(1).cpu().detach().numpy()
    #         predicted_labels[i] =  sampled_indices_np #[40,]
    #         i+=1
      
    # plot_camera_positions_and_rays(selected_camera_positions, img_semantic, selected_ray_data, output_folder,
    #                             position_key='semantic', camera_labels=camera_labels,
    #                             camera_dot_colors=unique_colors, filename_suffix="Predicted", object_color=predicted_labels)

