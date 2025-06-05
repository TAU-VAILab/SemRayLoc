import os
import argparse
import yaml
import torch
import tqdm
import numpy as np
from attrdict import AttrDict
from torch.utils.data import DataLoader
from data_utils.prob_vol_data_utils import ProbVolDataset
from utils.localization_utils import finalize_localization
from multiprocessing import Pool
from utils.result_utils import (
    save_acc_and_orn_records,
    calculate_recalls,
    save_recalls,
    create_combined_results_table,
)

# Global variables for multiprocessing
dataset = None
config = None

def save_combine_map_vector(vector, save_path, file_name):
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, file_name + ".pt")
    torch.save(vector, file_path)

def calculate_best_combination(prob_vol_pred_depth, prob_vol_pred_semantic, prob_vol_gt_depth, prob_vol_gt_semantic, weight_combinations, ref_pose, device):
    pred_accs = []
    pred_acc_orns = []
    gt_accs = []
    gt_acc_orns = []

    for depth_weight, semantic_weight in weight_combinations:
        # Combine volumes
        min_shape = [min(d, s) for d, s in zip(prob_vol_pred_depth.shape, prob_vol_pred_semantic.shape)]
        prob_vol_pred_depth_sliced = prob_vol_pred_depth[tuple(slice(0, min_shape[i]) for i in range(len(min_shape)))]
        prob_vol_pred_semantic_sliced = prob_vol_pred_semantic[tuple(slice(0, min_shape[i]) for i in range(len(min_shape)))]
        combined_pred = depth_weight * prob_vol_pred_depth_sliced + semantic_weight * prob_vol_pred_semantic_sliced

        min_shape = [min(d, s) for d, s in zip(prob_vol_gt_depth.shape, prob_vol_gt_semantic.shape)]
        prob_vol_pred_gt_depth_sliced = prob_vol_gt_depth[tuple(slice(0, min_shape[i]) for i in range(len(min_shape)))]
        prob_vol_pred_gt_semantic_sliced = prob_vol_gt_semantic[tuple(slice(0, min_shape[i]) for i in range(len(min_shape)))]
        combined_gt = depth_weight * prob_vol_pred_gt_depth_sliced + semantic_weight * prob_vol_pred_gt_semantic_sliced

        # Localize and calculate accuracy for prediction
        _, _, _, pose_pred = finalize_localization(combined_pred)
        pose_pred = torch.tensor(pose_pred, device=device, dtype=torch.float32)
        pose_pred[:2] = pose_pred[:2] / 10  # Scale poses
        acc_pred = torch.norm(pose_pred[:2] - ref_pose[:2], p=2).item()

        acc_orn_pred = ((pose_pred[2] - ref_pose[2]) % (2 * np.pi)).item()
        acc_orn_pred = min(acc_orn_pred, 2 * np.pi - acc_orn_pred) / np.pi * 180

        pred_accs.append(acc_pred)
        pred_acc_orns.append(acc_orn_pred)

        # Localize and calculate accuracy for GT
        _, _, _, pose_gt = finalize_localization(combined_gt)
        pose_gt = torch.tensor(pose_gt, device=device, dtype=torch.float32)
        pose_gt[:2] = pose_gt[:2] / 10
        acc_gt = torch.norm(pose_gt[:2] - ref_pose[:2], p=2).item()

        acc_orn_gt = ((pose_gt[2] - ref_pose[2]) % (2 * np.pi)).item()
        acc_orn_gt = min(acc_orn_gt, 2 * np.pi - acc_orn_gt) / np.pi * 180

        gt_accs.append(acc_gt)
        gt_acc_orns.append(acc_orn_gt)

    # Convert to tensors
    pred_accs_tensor = torch.tensor(pred_accs)
    gt_accs_tensor = torch.tensor(gt_accs)

    # Find the index of the minimum accuracy
    pred_best_idx = pred_accs_tensor.argmin()
    gt_best_idx = gt_accs_tensor.argmin()

    # Find the minimal accuracy value
    pred_best_acc = pred_accs_tensor.min()
    gt_best_acc = gt_accs_tensor.min()

    # Create vectors
    pred_best_vector = torch.zeros(len(weight_combinations))
    if pred_best_acc < 1 and abs(pred_best_acc-pred_accs_tensor[-1]) > 0.1:
        pred_best_vector[pred_best_idx] = 1
    else:
        pred_best_vector[-1] = 1         

    gt_best_vector = torch.zeros(len(weight_combinations))
    if gt_best_acc < 1.0:
        gt_best_vector[gt_best_idx] = 1
    else:
        gt_best_vector[-1] = 1 

    # Return the best vectors and the accuracy records
    return pred_best_vector, gt_best_vector, pred_accs, pred_acc_orns, gt_accs, gt_acc_orns

def process_data_idx(data_idx):
    global dataset
    global config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset is None:
        # Instantiate the dataset
        dataset = ProbVolDataset(
            dataset_dir=config.dataset_dir,
            scene_names=config.scene_names,
            L=config.L,
            prob_vol_path=config.prob_vol_path,
            acc_only=False,
        )

    data = dataset[data_idx]
    ref_pose = torch.tensor(data["ref_pose"], device=device, dtype=torch.float32)
    prob_vol_pred_depth = data['prob_vol_depth'].to(device)
    prob_vol_pred_semantic = data['prob_vol_semantic'].to(device)
    prob_vol_gt_depth = data['prob_vol_depth_gt'].to(device)
    prob_vol_gt_semantic = data['prob_vol_semantic_gt'].to(device)

    # Calculate the best combinations and accuracy records
    weight_combinations = config.weight_combinations
    pred_best_vector, gt_best_vector, pred_accs, pred_acc_orns, gt_accs, gt_acc_orns = calculate_best_combination(
        prob_vol_pred_depth, prob_vol_pred_semantic, prob_vol_gt_depth, prob_vol_gt_semantic, weight_combinations, ref_pose, device
    )

    # Save vectors
    scene_idx = np.sum(data_idx >= np.array(dataset.scene_start_idx)) - 1
    scene_name = dataset.scene_names[scene_idx]
    scene_number = int(scene_name.split("_")[1])
    scene = f"scene_{scene_number}"
    scene_save_dir = os.path.join(config.prob_vol_save_dir, "combine_map_vector", scene)
    idx_within_scene = data_idx - dataset.scene_start_idx[scene_idx]

    if config.save_vectors:
        camera_id = f"camera_{idx_within_scene}"
        save_combine_map_vector(pred_best_vector, scene_save_dir, f"{camera_id}_pred_best")
        save_combine_map_vector(gt_best_vector, scene_save_dir, f"{camera_id}_gt_best")

    # Return the best vectors and accuracy records
    return pred_best_vector, gt_best_vector, pred_accs, pred_acc_orns, gt_accs, gt_acc_orns

def generate_and_save_combine_map_vectors(config, dataset, prob_vol_save_dir):
    all_pred_vectors = []
    all_gt_vectors = []
    weight_combinations = config.weight_combinations

    # Initialize dictionaries to collect accuracy records
    pred_acc_records_for_all_weights = {f"{depth_weight}_{semantic_weight}": [] for depth_weight, semantic_weight in weight_combinations}
    pred_acc_orn_records_for_all_weights = {f"{depth_weight}_{semantic_weight}": [] for depth_weight, semantic_weight in weight_combinations}
    gt_acc_records_for_all_weights = {f"{depth_weight}_{semantic_weight}": [] for depth_weight, semantic_weight in weight_combinations}
    gt_acc_orn_records_for_all_weights = {f"{depth_weight}_{semantic_weight}": [] for depth_weight, semantic_weight in weight_combinations}

    total = len(dataset)
    data_indices = list(range(total))
    num_processes = os.cpu_count()

    with Pool(processes=num_processes) as pool:
        results = list(tqdm.tqdm(pool.imap(process_data_idx, data_indices), total=total, desc="Processing Data"))

    for result in results:
        pred_vector, gt_vector, pred_accs, pred_acc_orns, gt_accs, gt_acc_orns = result
        if pred_vector is not None and gt_vector is not None:
            all_pred_vectors.append(pred_vector)
            all_gt_vectors.append(gt_vector)
            # Collect accuracy records
            for idx, (depth_weight, semantic_weight) in enumerate(weight_combinations):
                weight_key = f"{depth_weight}_{semantic_weight}"
                pred_acc_records_for_all_weights[weight_key].append(pred_accs[idx])
                pred_acc_orn_records_for_all_weights[weight_key].append(pred_acc_orns[idx])
                gt_acc_records_for_all_weights[weight_key].append(gt_accs[idx])
                gt_acc_orn_records_for_all_weights[weight_key].append(gt_acc_orns[idx])

    # Stack all vectors
    pred_tensor = torch.stack(all_pred_vectors)
    gt_tensor = torch.stack(all_gt_vectors)

    # Calculate percentage of times each entry is 1
    pred_percentages = (pred_tensor.sum(dim=0) / total) * 100
    gt_percentages = (gt_tensor.sum(dim=0) / total) * 100

    # Prepare table as a string
    header = f"{'Weights':<30} {'Prediction %':<15} {'Ground Truth %':<15}"
    divider = "-" * len(header)
    rows = []
    for idx, (depth_weight, semantic_weight) in enumerate(weight_combinations):
        weights_str = f"Depth={depth_weight}, Semantic={semantic_weight}"
        pred_str = f"{pred_percentages[idx]:.2f}%"
        gt_str = f"{gt_percentages[idx]:.2f}%"
        rows.append(f"{weights_str:<30} {pred_str:<15} {gt_str:<15}")

    # Print the table
    print(f"\nTotal number of images processed: {total}\n")
    print(header)
    print(divider)
    for row in rows:
        print(row)

    # Process and save results for weight combinations
    combined_recalls_pred = {}
    combined_recalls_gt = {}
    results_type_dir = os.path.join(prob_vol_save_dir, "combine_map_vector", "results")

    for depth_weight, semantic_weight in weight_combinations:
        weight_key = f"{depth_weight}_{semantic_weight}"
        # For predictions
        acc_record_pred = np.array(pred_acc_records_for_all_weights[weight_key])
        acc_orn_record_pred = np.array(pred_acc_orn_records_for_all_weights[weight_key])

        weight_dir_pred = os.path.join(
            results_type_dir, "predictions", f"depth{depth_weight}_semantic{semantic_weight}"
        )
        os.makedirs(weight_dir_pred, exist_ok=True)
        save_acc_and_orn_records(acc_record_pred, acc_orn_record_pred, weight_dir_pred)

        recalls_pred = calculate_recalls(acc_record_pred, acc_orn_record_pred)
        combined_recalls_pred[weight_key] = recalls_pred
        save_recalls(recalls_pred, weight_dir_pred, weight_key)

        # For ground truth
        acc_record_gt = np.array(gt_acc_records_for_all_weights[weight_key])
        acc_orn_record_gt = np.array(gt_acc_orn_records_for_all_weights[weight_key])

        weight_dir_gt = os.path.join(
            results_type_dir, "ground_truth", f"depth{depth_weight}_semantic{semantic_weight}"
        )
        os.makedirs(weight_dir_gt, exist_ok=True)
        save_acc_and_orn_records(acc_record_gt, acc_orn_record_gt, weight_dir_gt)

        recalls_gt = calculate_recalls(acc_record_gt, acc_orn_record_gt)
        combined_recalls_gt[weight_key] = recalls_gt
        save_recalls(recalls_gt, weight_dir_gt, weight_key)

    # Create combined results tables
    create_combined_results_table(combined_recalls_pred, os.path.join(results_type_dir, "predictions"))
    create_combined_results_table(combined_recalls_gt, os.path.join(results_type_dir, "ground_truth"))

def main():
    global config
    parser = argparse.ArgumentParser(description="Create combine map vectors.")
    parser.add_argument(
        "--config_file",
        type=str,
        default="evaluation/configuration/config_create_best_map_vector.yaml",
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    # Load configuration from file
    with open(args.config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    config = AttrDict(config_dict)

    # Set scene_numbers and scene_names in config
    config.scene_numbers = range(config.start_scene, config.end_scene + 1)
    config.scene_names = [f"scene_{str(i).zfill(5)}" for i in config.scene_numbers]

    # Load test dataset
    test_set = ProbVolDataset(
        dataset_dir=config.dataset_dir,
        scene_names=config.scene_names,
        L=config.L,
        prob_vol_path=config.prob_vol_path,
        acc_only=False,
    )
    generate_and_save_combine_map_vectors(config, test_set, config.prob_vol_save_dir)

if __name__ == "__main__":
    main()
