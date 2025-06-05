import argparse
import os
import numpy as np
import torch
import tqdm
import yaml
from attrdict import AttrDict
import cv2
import torch.nn as nn
from PIL import Image

# Import your models
from modules.mono.depth_net_pl import depth_net_pl
from modules.semantic.semantic_net_pl import semantic_net_pl
from modules.combined.map_prediction_nets.map_predictor_pl import MapPredictorPL


# Import your dataset
from data_utils.bets_map_vector_data_utils import BestMapVectorDataset
import torch.nn.functional as F

# Import helper functions from utils
from utils.localization_utils import (
    get_ray_from_depth,
    get_ray_from_semantics,
    localize,
    finalize_localization,
    finalize_localization_acc_only
)
from utils.result_utils import (
    save_acc_and_orn_records,
    calculate_recalls,
    save_recalls,
    create_combined_results_table,
)
from utils.data_loader_helper import load_scene_data
from utils.visualization_utils import plot_prob_dist, plot_dict_relationship

from data_utils.prob_vol_data_utils import ProbVolDataset
from torch.utils.data import DataLoader

def pad_to_max(prob_vol,max_H, max_W):
    prob_vol= F.pad(prob_vol,(0, max_W - prob_vol.shape[1],0, max_H - prob_vol.shape[0]))
    return prob_vol

def evaluate_combined_model(
    map_predictor_pl,
    test_set,
    gt_poses,
    device,
    results_type_dir,
    valid_scene_names,
    config,  
):
    weight_combinations = config.weight_combinations
    combined_recalls = {}

    # Initialize records for weight combinations
    acc_records_for_all_weights = {
        f"{depth_weight}_{semantic_weight}": []
        for depth_weight, semantic_weight in weight_combinations
    }
    acc_orn_records_for_all_weights = {
        f"{depth_weight}_{semantic_weight}": []
        for depth_weight, semantic_weight in weight_combinations
    }
    all_weight_comb_dict = {f"{depth_weight}_{semantic_weight}": {} for depth_weight, semantic_weight in weight_combinations}
    
    for data_idx in tqdm.tqdm(range(len(test_set))):
        data = test_set[data_idx]
        scene_idx = np.sum(data_idx >= np.array(test_set.scene_start_idx)) - 1
        scene = test_set.scene_names[scene_idx]
        if 'floor' in scene:
            pass
        else:
            scene_number = int(scene.split("_")[1])
            scene = f"scene_{scene_number}"
        
        if not scene in valid_scene_names:
            continue
            
        idx_within_scene = data_idx - test_set.scene_start_idx[scene_idx]
        ref_pose_map = gt_poses[scene][idx_within_scene * (config.L + 1) + config.L, :]                
        prob_vol_pred_depth = data['prob_vol_depth'].to(device)
        prob_vol_pred_semantic = data['prob_vol_semantic'].to(device)

        # if config.pad_to_max:
        #     prob_vol_pred_depth = pad_to_max(prob_vol_pred_depth, config.max_h//10,config.max_w//10)
            # prob_vol_pred_semantic = pad_to_max(prob_vol_pred_semantic, config.max_h//10,config.max_w//10)
            
        # Combine probabilities using weights
        for depth_weight, semantic_weight in weight_combinations:
            weight_key = f"{depth_weight}_{semantic_weight}"
            min_shape = [min(d, s) for d, s in zip(prob_vol_pred_depth.shape, prob_vol_pred_semantic.shape)]
            prob_vol_pred_depth_sliced = prob_vol_pred_depth[tuple(slice(0, min_shape[i]) for i in range(len(min_shape)))]
            prob_vol_pred_semantic_sliced = prob_vol_pred_semantic[tuple(slice(0, min_shape[i]) for i in range(len(min_shape)))]
            if depth_weight == "use_best_map_net":
                logits = map_predictor_pl(prob_vol_pred_depth_sliced.unsqueeze(0), prob_vol_pred_semantic_sliced.unsqueeze(0))
                pred_class_indices = logits.argmax(dim=1) 
                depth_weight, semantic_weight = config.best_map_weight_combinations[pred_class_indices[0].item()]
                combined_prob_vol_pred = depth_weight * prob_vol_pred_depth_sliced + semantic_weight * prob_vol_pred_semantic_sliced  # [H, W, O]
            else:
                combined_prob_vol_pred = depth_weight * prob_vol_pred_depth_sliced + semantic_weight * prob_vol_pred_semantic_sliced    
                            
            prob_vol_pred, prob_dist_pred, orientations_pred, pose_pred = finalize_localization(combined_prob_vol_pred)
            
            # Convert pose_pred to torch.Tensor before calling .cpu()
            pose_pred = torch.tensor(pose_pred, device=device, dtype=torch.float32)
            pose_pred[:2] = pose_pred[:2] / 10  # Scale poses
            acc = torch.norm(pose_pred[:2] - torch.tensor(ref_pose_map[:2], device=device), p=2).item()
            acc_orn = ((pose_pred[2] - torch.tensor(ref_pose_map[2], device=device)) % (2 * np.pi)).item()
            acc_orn = min(acc_orn, 2 * np.pi - acc_orn) / np.pi * 180                
            
            acc_records_for_all_weights[weight_key].append(acc)
            acc_orn_records_for_all_weights[weight_key].append(acc_orn)        

    # Process and save results for weight combinations
    for depth_weight, semantic_weight in weight_combinations:
        weight_key = f"{depth_weight}_{semantic_weight}"
        acc_record = np.array(acc_records_for_all_weights[weight_key])
        acc_orn_record = np.array(acc_orn_records_for_all_weights[weight_key])

        weight_dir = os.path.join(
            results_type_dir, f"depth{depth_weight}_semantic{semantic_weight}"
        )
        os.makedirs(weight_dir, exist_ok=True)
        save_acc_and_orn_records(acc_record, acc_orn_record, weight_dir)

        recalls = calculate_recalls(acc_record, acc_orn_record)
        combined_recalls[weight_key] = recalls
        save_recalls(recalls, weight_dir, weight_key)

    create_combined_results_table(combined_recalls, results_type_dir)



def evaluate_observation(config, device):
    # Extract configuration parameters
    dataset_dir = config.dataset_dir
    desdf_path = config.desdf_path
    results_dir = config.results_dir
    split_file = config.split_file



    # split_file = os.path.join(dataset_dir, "split.yaml")
    with open(split_file, "r") as f:
        split = AttrDict(yaml.safe_load(f))

    scene_names = split.test[:config.num_of_scenes] 
    test_set = BestMapVectorDataset(
        dataset_dir=dataset_dir,
        scene_names=scene_names,
        prob_vol_path=config.prob_vol_path,
        best_map_path=config.best_map_path,
        L=0,
    )

    map_predictor_pl = MapPredictorPL.load_from_checkpoint(
        checkpoint_path=config.map_predictor_pl_log_dir,
        config=config,
    ).to(device)
    
    os.makedirs(results_dir, exist_ok=True)

    # Load desdf, semantics, maps, and ground truth poses
    _, _, _, gt_poses, valid_scene_names = load_scene_data(
        test_set, dataset_dir, desdf_path
    )

    evaluate_combined_model(
        map_predictor_pl,
        test_set,
        gt_poses,
        device,
        results_dir,
        valid_scene_names,
        config=config,  
    )


def main():
    parser = argparse.ArgumentParser(description="Observation evaluation.")
    parser.add_argument(
        "--config_file",
        type=str,
        default="evaluation/configuration/S3D/config_best_map_eval.yaml",
        # default="evaluation/configuration/zind/config_eval.yaml",
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    # Load configuration from file
    with open(args.config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    config = AttrDict(config_dict)

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("======= USING DEVICE : ", device, " =======")

    evaluate_observation(config, device)

if __name__ == "__main__":
    main()
