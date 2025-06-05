import os
import numpy as np
import torch
import tqdm
import yaml
from attrdict import AttrDict
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict

from modules.mono.depth_net_pl import depth_net_pl
from modules.semantic.semantic_net_pl import semantic_net_pl
from modules.semantic.semantic_net_pl_maskformer_small import semantic_net_pl_maskformer_small
from modules.semantic.semantic_mapper import ObjectType
from data_utils.data_utils import GridSeqDataset
from data_utils.prob_vol_data_utils import ProbVolDataset
from utils.localization_utils import (
    get_ray_from_depth, get_ray_from_semantics_v2, localize, finalize_localization
)
from utils.data_loader_helper import load_scene_data
from utils.visualization_utils import plot_dict_relationship
from scipy.ndimage import gaussian_filter1d
import seaborn as sns
import os
import pickle

# Hard-coded configuration
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

def save_data(data, path):
    """Saves data to a specified path using pickle."""
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_data(path):
    """Loads data from a specified path using pickle."""
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def pad_to_max(prob_vol, max_H, max_W):
    return F.pad(prob_vol, (0, max_W - prob_vol.shape[1], 0, max_H - prob_vol.shape[0]))


def plot_acc_vs_count(count_dict_for_all_weights, count_name, save_path, data_type):
    """
    Plots accuracy vs number of objects for different weight combinations with smoothing and range shading.
    :param count_dict_for_all_weights: Dictionary containing accuracy values for different weights.
    :param count_name: Type of object (e.g., windows, doors, doors_and_walls).
    :param save_path: Path to save the plot.
    :param data_type: Data type ('pred' or 'gt').
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 7))
    
    # Define the mapping from weight_key to legend label
    label_mapping = {
        "1.0_0.0": "Depth",
        "0.8_0.2": "Depth + Semantic"
    }
    
    for weight_key in count_dict_for_all_weights.keys():
        sorted_counts = sorted(count_dict_for_all_weights[weight_key].keys())
        mean_acc = [np.mean(count_dict_for_all_weights[weight_key][c]) for c in sorted_counts]
        std_acc = [np.std(count_dict_for_all_weights[weight_key][c]) for c in sorted_counts]
        
        smoothed_mean = gaussian_filter1d(mean_acc, sigma=1)
        
        # Upper and lower bounds
        lower_bound = np.maximum(np.array(mean_acc) - np.array(std_acc), 0)  # Ensure lower bound is >= 0
        upper_bound = np.array(mean_acc) + np.array(std_acc)
        
        # Get the legend label from mapping, defaulting to the weight_key if not found
        legend_label = label_mapping.get(weight_key, weight_key)
        
        # Plot the smoothed mean
        sns.lineplot(x=sorted_counts, y=smoothed_mean, label=legend_label, linewidth=6)
        sns.lineplot(x=sorted_counts, y=mean_acc, linewidth=1, alpha=0.4)
        
        # Uncomment below to add shaded range for bounds (unsmoothed)
        # plt.fill_between(sorted_counts, lower_bound, upper_bound, alpha=0.05)

    plt.xlabel(f'Number of {count_name} Rays')
    plt.ylabel('Mean Accuracy (m)')
    # Uncomment if you want a title
    # plt.title(f'Mean Accuracy vs Number of {count_name} Rays ({data_type.upper()})')
    
    # Create the legend without a header
    plt.legend(title=None)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{data_type}_acc_vs_{count_name}.png'))
    plt.close()




def evaluate_combined_model(
    depth_net, semantic_net, desdfs, semantics, test_set, gt_poses, maps,
    device, results_type_dir, valid_scene_names, config
):
    # Define file paths to save intermediate results
    save_path = '/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/results/final_results/mask2former/visualtizations/semantic_vs_acc'
    
    os.makedirs(save_path, exist_ok=True)
    counts_file = os.path.join(save_path, 'count_dict_for_all_weights.pkl')
    
    if os.path.exists(counts_file):
        # Load existing data if it exists
        print(f"Loading existing count data from {counts_file}")
        data = load_data(counts_file)
        windows_acc = data['windows_acc']
        doors_acc = data['doors_acc']
        doors_and_windows_acc = data['doors_and_windows_acc']
        gt_windows_acc = data['gt_windows_acc']
        gt_doors_acc = data['gt_doors_acc']
        gt_doors_and_windows_acc = data['gt_doors_and_windows_acc']
    else:
        print(f"No existing data found at {counts_file}, running calculations.")
        wc = config.weight_combinations
        acc_records = {f"{dw}_{sw}": [] for dw, sw in wc}
        acc_orn_records = {f"{dw}_{sw}": [] for dw, sw in wc}
        all_weight_comb_dict = {f"{dw}_{sw}": {} for dw, sw in wc}
        windows_acc = {f"{dw}_{sw}": defaultdict(list) for dw, sw in wc}
        doors_acc = {f"{dw}_{sw}": defaultdict(list) for dw, sw in wc}
        doors_and_windows_acc = {f"{dw}_{sw}": defaultdict(list) for dw, sw in wc}

        # Dictionaries for GT data
        gt_windows_acc = {f"{dw}_{sw}": defaultdict(list) for dw, sw in wc}
        gt_doors_acc = {f"{dw}_{sw}": defaultdict(list) for dw, sw in wc}
        gt_doors_and_windows_acc = {f"{dw}_{sw}": defaultdict(list) for dw, sw in wc}

        for data_idx in tqdm.tqdm(range(len(test_set))):
            data = test_set[data_idx]
            scene_idx = np.sum(data_idx >= np.array(test_set.scene_start_idx)) - 1
            scene = test_set.scene_names[scene_idx]
            if 'floor' not in scene:
                scene = f"scene_{int(scene.split('_')[1])}"
            if scene not in valid_scene_names:
                continue

            idx_within_scene = data_idx - test_set.scene_start_idx[scene_idx]
            desdf, semantic = desdfs[scene], semantics[scene]
            ref_pose_map = gt_poses[scene][idx_within_scene * (config.L + 1) + config.L]

            pred_depths = depth_net.encoder(torch.tensor(data["ref_img"], device=device).unsqueeze(0), None)[0].squeeze(0).cpu().detach().numpy()
            pred_rays_depth = get_ray_from_depth(pred_depths, V=config.V, F_W=config.F_W)
            
            gt_pred_depths = data["ref_depth"]
            gt_rays_depth = get_ray_from_depth(gt_pred_depths, V=config.V, F_W=config.F_W)

            ref_img_t = torch.tensor(data["ref_img"], device=device).unsqueeze(0)
            _, _, prob = semantic_net.encoder(ref_img_t, None)
            sampled_indices_np = torch.multinomial(prob.squeeze(0), 1, replacement=True).squeeze(1).cpu().detach().numpy()
            pred_rays_semantic = get_ray_from_semantics_v2(sampled_indices_np)
            
            gt_sampled_indices_np = data["ref_semantics"]
            gt_rays_semantic = get_ray_from_semantics_v2(gt_sampled_indices_np)

            prob_vol_pred_depth = localize(torch.tensor(desdf["desdf"]), torch.tensor(pred_rays_depth), return_np=False)[0]
            prob_vol_pred_semantic = localize(torch.tensor(semantic["desdf"]), torch.tensor(pred_rays_semantic), return_np=False,localize_type="semantic")[0]
            
            gt_prob_vol_pred_depth = localize(torch.tensor(desdf["desdf"]), torch.tensor(gt_rays_depth), return_np=False)[0]
            gt_prob_vol_pred_semantic = localize(torch.tensor(semantic["desdf"]), torch.tensor(gt_rays_semantic), return_np=False,localize_type="semantic")[0]

            # Separate GT and semantic counts
            num_windows_pred = np.sum(sampled_indices_np == ObjectType.WINDOW.value)
            num_doors_pred = np.sum(sampled_indices_np == ObjectType.DOOR.value)
            num_doors_and_windows_pred = np.sum((sampled_indices_np == ObjectType.DOOR.value) | (sampled_indices_np == ObjectType.WINDOW.value))

            num_windows_gt = np.sum(gt_sampled_indices_np == ObjectType.WINDOW.value)
            num_doors_gt = np.sum(gt_sampled_indices_np == ObjectType.DOOR.value)
            num_doors_and_windows_gt = np.sum((gt_sampled_indices_np == ObjectType.DOOR.value) | (gt_sampled_indices_np == ObjectType.WINDOW.value))

            for dw, sw in wc:
                wkey = f"{dw}_{sw}"
                min_shape = [min(d, s) for d, s in zip(prob_vol_pred_depth.shape, prob_vol_pred_semantic.shape)]
                d_sliced = prob_vol_pred_depth[tuple(slice(0, m) for m in min_shape)]
                s_sliced = prob_vol_pred_semantic[tuple(slice(0, m) for m in min_shape)]
                combined_prob = dw * d_sliced + sw * s_sliced
                _, pdp, _, pose_pred = finalize_localization(combined_prob)
                pose_pred = torch.tensor(pose_pred, device=device, dtype=torch.float32)
                pose_pred[:2] = pose_pred[:2] / 10
                acc = torch.norm(pose_pred[:2] - torch.tensor(ref_pose_map[:2], device=device), 2).item()
                acc_orn = ((pose_pred[2] - ref_pose_map[2]) % (2 * np.pi))
                acc_orn = min(acc_orn, 2 * np.pi - acc_orn) / np.pi * 180
                key = pdp.max().item()
                all_weight_comb_dict[wkey].setdefault(key, []).append(acc)
                acc_records[wkey].append(acc)
                acc_orn_records[wkey].append(acc_orn)
                windows_acc[wkey][num_windows_pred].append(acc)
                doors_acc[wkey][num_doors_pred].append(acc)
                doors_and_windows_acc[wkey][num_doors_and_windows_pred].append(acc)
                
                min_shape = [min(d, s) for d, s in zip(gt_prob_vol_pred_depth.shape, gt_prob_vol_pred_semantic.shape)]
                d_gt_sliced = gt_prob_vol_pred_depth[tuple(slice(0, m) for m in min_shape)]
                s_gt_sliced = gt_prob_vol_pred_semantic[tuple(slice(0, m) for m in min_shape)]
                gt_combined_prob = dw * d_gt_sliced + sw * s_gt_sliced
                _, gt_pdp, _, gt_pose_pred = finalize_localization(gt_combined_prob)
                gt_pose_pred = torch.tensor(gt_pose_pred, device=device, dtype=torch.float32)
                gt_pose_pred[:2] = gt_pose_pred[:2] / 10
                gt_acc = torch.norm(gt_pose_pred[:2] - torch.tensor(ref_pose_map[:2], device=device), 2).item()
                gt_windows_acc[wkey][num_windows_gt].append(gt_acc)
                gt_doors_acc[wkey][num_doors_gt].append(gt_acc)
                gt_doors_and_windows_acc[wkey][num_doors_and_windows_gt].append(gt_acc)
                
        # Save the computed results
        data_to_save = {
            'windows_acc': windows_acc,
            'doors_acc': doors_acc,
            'doors_and_windows_acc': doors_and_windows_acc,
            'gt_windows_acc': gt_windows_acc,
            'gt_doors_acc': gt_doors_acc,
            'gt_doors_and_windows_acc': gt_doors_and_windows_acc
        }
        save_data(data_to_save, counts_file)


    save_path = '/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/results/final_results/mask2former/visualtizations/semantic_vs_acc'
    os.makedirs(save_path, exist_ok=True)

    plot_acc_vs_count(windows_acc, 'windows', save_path, 'pred')
    plot_acc_vs_count(doors_acc, 'doors', save_path, 'pred')
    plot_acc_vs_count(doors_and_windows_acc, 'doors_and_windows', save_path, 'pred')
    
    plot_acc_vs_count(gt_windows_acc, 'windows', save_path, 'gt')
    plot_acc_vs_count(gt_doors_acc, 'doors', save_path, 'gt')
    plot_acc_vs_count(gt_doors_and_windows_acc, 'doors_and_windows', save_path, 'gt')



def evaluate_observation(prediction_type, config, device):
    with open(config.split_file, "r") as f:
        split = AttrDict(yaml.safe_load(f))
    scene_names = split.test[:config.num_of_scenes] if config.num_of_scenes>0 else split.test
    test_set = ProbVolDataset(config.dataset_dir, scene_names, L=config.L, prob_vol_path=config.prob_vol_path, acc_only=False) if config.use_saved_prob_vol \
               else GridSeqDataset(config.dataset_dir, scene_names, L=config.L)

    depth_net, semantic_net = None, None
    if not config.use_ground_truth_depth or not config.use_ground_truth_semantic:
        if prediction_type in ["depth", "combined"]:
            depth_net = depth_net_pl.load_from_checkpoint(config.log_dir_depth, d_min=config.d_min, d_max=config.d_max, d_hyp=config.d_hyp, D=config.D).to(device)
        if prediction_type in ["semantic", "combined"]:
            if config.use_semantic_mask2foremr:            
                semantic_net = semantic_net_pl_maskformer_small.load_from_checkpoint(config.semantic_mask2former_path, num_classes=config.num_classes).to(device)
            else:
                semantic_net = semantic_net_pl.load_from_checkpoint(config.log_dir_semantic, num_classes=config.num_classes).to(device)

    results_type_dir = os.path.join(config.results_dir, prediction_type)
    if config.use_ground_truth_depth or config.use_ground_truth_semantic:
        results_type_dir = os.path.join(results_type_dir, "gt")
    os.makedirs(results_type_dir, exist_ok=True)

    desdfs, semantics, maps, gt_poses, valid_scene_names,_ = load_scene_data(test_set, config.dataset_dir, config.desdf_path)
    evaluate_combined_model(depth_net, semantic_net, desdfs, semantics, test_set, gt_poses, maps, device, results_type_dir, valid_scene_names, config)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if config.prediction_type == "all":
        for pt in ["depth", "semantic", "combined"]:
            evaluate_observation(pt, config, device)
    else:
        evaluate_observation(config.prediction_type, config, device)

if __name__ == "__main__":
    main()
