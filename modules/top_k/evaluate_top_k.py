import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from attrdict import AttrDict
from datetime import datetime
import numpy as np
from collections import defaultdict
import logging
import random
import json  # Added for loading metadata
from glob import glob  # Added for handling file paths

from best_k_net_pl import best_k_net_pl
from best_k_net_pl_with_fp import best_k_net_pl_with_fp
from top_k_dataset import TopKDataset

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return AttrDict(yaml.safe_load(f))

def load_split(split_file):
    """Load the dataset split YAML file."""
    with open(split_file, "r") as f:
        split = AttrDict(yaml.safe_load(f))
    return split

def process_scene_names(scene_list):
    """
    Process scene names according to the dataset type.
    If the scene name contains 'floor', it is assumed to be Zind dataset and
    processed accordingly; otherwise for S3D, remove leading zeros.
    """
    processed = []
    for scene in scene_list:
        # Default is_zind flag false
        is_zind = False
        if 'floor' in scene:
            is_zind = True
            processed.append(scene)
        else:
            # Assuming scene format "scene_00000", remove leading zeros.
            try:
                scene_number = int(scene.split('_')[1])
                scene_name = f"scene_{scene_number}"
            except Exception as e:
                logging.error(f"Error processing scene: {scene} with error: {e}")
                scene_name = scene
            processed.append(scene_name)
    return processed

def setup_dataset_and_loader(config, scenes):
    """
    Set up the TopKDataset and corresponding DataLoader using the specified split.
    dataset_split: 'train', 'val', or 'test'
    """
    dataset_dir = os.path.join(config.dataset_path, config.dataset)
    
    # Process scene names based on the dataset type
    scenes = process_scene_names(scenes)[:-1]
    
    dataset = TopKDataset(
        scene_names=scenes,
        image_base_dir=dataset_dir,
        top_k_dir=config.top_k_results_dir,
        poses_filename=config.poses_filename,
        enforce_fixed_resolution=True,
        target_resolution=(360, 640),
        desired_candidate_dim=(10, 40)
    )
    
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.get("num_workers", 8)
    )
    
    return dataset, loader

def calculate_recalls(acc_record, acc_orn_record):
    recalls = {
        "1m": np.sum(acc_record < 1) / acc_record.shape[0],
        "0.5m": np.sum(acc_record < 0.5) / acc_record.shape[0],
        "0.1m": np.sum(acc_record < 0.1) / acc_record.shape[0],
        "1m 30 deg": np.sum(np.logical_and(acc_record < 1, acc_orn_record < 30)) / acc_record.shape[0] if acc_orn_record is not None else 0,
    }
    return recalls

def angular_difference(pred_o, gt_o):
    """
    Calculate the minimal angular difference between predictions and ground truth.
    Assumes angles are in degrees.
    """
    diff = torch.abs(pred_o - gt_o) % 360
    diff = torch.where(diff > 180, 360 - diff, diff)
    return diff

def load_metadata(metadata_path):
    """
    Load metadata JSON file.
    """
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        logging.error(f"Error loading metadata from {metadata_path}: {e}")
        return None

def compare_rays(k1_depth_rays, k_depth_rays, k1_semantic_rays, k_semantic_rays):
    counter_depth = 0
    counter_semantic = 0
    if len(k1_depth_rays) != len(k_depth_rays) or len(k1_semantic_rays) != len(k_semantic_rays):
        return False  
    for ray1, ray2 in zip(k1_depth_rays, k_depth_rays):
        if abs(ray1['distance_m']- ray2['distance_m']) > 0.2:
            counter_depth += 1
    for ray1, ray2 in zip(k1_semantic_rays, k_semantic_rays):
        if ray1['prediction_class'] != ray2['prediction_class']:
            counter_semantic += 1
    return True if counter_depth < 1 and counter_semantic < 5 else False

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Argument parsing
    parser = argparse.ArgumentParser(description="Evaluation script for best_k_net.")
    parser.add_argument(
        "--config",
        type=str,
        default="modules/top_k/config_evaluate_top_k.yaml",
        help="Path to the YAML config file.",
    )
    args = parser.parse_args()

    # Enable CUDA_LAUNCH_BLOCKING for better error tracing
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Set random seed for reproducibility in the new baseline
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Load configuration.
    config = load_config(args.config)
    
    # Load split file.
    split = load_split(config.split_file)
    
    # Create log directory name if needed, not necessary for evaluation
    # You can skip this step or create a separate directory for evaluation logs
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    default_root_dir = config.ckpt_path
    os.makedirs(default_root_dir, exist_ok=True)
    logging.info(f"Logging evaluation to: {default_root_dir}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup dataset and DataLoader for test split.
    test_dataset, test_loader = setup_dataset_and_loader(config, split.test)
    
    # # Initialize the model.
    # if config.use_fp:
    #     model = best_k_net_pl_with_fp.load_from_checkpoint(config.checkpoint, lr=config.lr)
    # else:
    #     model = best_k_net_pl.load_from_checkpoint(config.checkpoint, lr=config.lr)
    
    # # Ensure the model is in evaluation mode
    # model.eval()
    # model.freeze()  # Freeze model parameters
    
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model.to(device)
    
    # Initialize metric containers for model
    acc_records_for_all_weights = defaultdict(list)
    acc_orn_records_for_all_weights = defaultdict(list)
    
    # Initialize metric containers for baseline (always picking k=0)
    acc_records_baseline = []
    acc_orn_records_baseline = []
    
    # Initialize metric containers for random depth ray baseline
    acc_records_random_depth = []
    acc_orn_records_random_depth = []
    

    acc_records_random_score = []
    acc_orn_records_random_score = []
    
    # Initialize metric containers for Top 2, Top 3, Top 4, and Top 5
    acc_records_top2 = []
    acc_orn_records_top2 = []
    
    acc_records_top3 = []
    acc_orn_records_top3 = []
    
    acc_records_top10 = []
    acc_orn_records_top10 = []
    
    acc_records_top5 = []
    acc_orn_records_top5 = []
    
    # If you have different weights, manage them accordingly
    # For simplicity, assuming one weight_key, e.g., 'default'
    weight_key = 'default'
    random_count = 0
    random_score_counter = 0
    random_choices_len = []
    random_selected = []
    # Iterate over test data
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Move data to device
            ref_imgs = batch["ref_img"].to(device)              # (N, 3, 360, 640)
            depth_vecs = batch["depth_vec"].to(device)          # (N, k, 40)
            sem_vecs = batch["sem_vec"].to(device)              # (N, k, 40)
            semantic_maps = batch["semantic_map"].to(device)    # (N, 1, 300, 300)
            gt_locations = batch["gt_location"].to(device)      # (N, 3)
            k_positions = batch["k_positions"]                  # list of N elements, each list of k [x, y, o]
            k_scores_batch = batch["k_scores"]                  # list of N elements, each list of k scores
            metadata_paths = batch["metadata_path"]              # list of N metadata file paths
            best_index = batch["best_index"]                  # (N,)
            k_scores = batch["k_scores"]                        # (N, k)
            
            # Convert k_positions from list to tensor
            k_positions = [[list(p) for p in pos] for pos in k_positions]  
            k_positions_tensor = torch.tensor(k_positions, dtype=torch.float32, device=device) 

            # Forward pass: get logits
            # logits = model(ref_imgs, depth_vecs, sem_vecs, semantic_maps, k_positions_tensor)  # (N, K)
            K = len(k_scores)
            
            # **Compute Top 1 Prediction**
            # Get the predicted candidate indices (0-indexed)
            # preds = torch.argmax(logits, dim=1)  # (N,)
            
            # Validate preds are within [0, K-1]
            # valid_mask = (preds >= 0) & (preds < K)
            # if not torch.all(valid_mask):
            #     invalid_preds = preds[~valid_mask]
            #     logging.warning(f"Batch {batch_idx}: Predicted indices out of bounds: {invalid_preds}")
            #     logging.warning(f"k_positions_tensor size: {k_positions_tensor.size()}")
            #     # Optionally, set invalid preds to a default value (e.g., 0)
            #     preds = torch.where(valid_mask, preds, torch.zeros_like(preds))
            
            # Gather predicted positions and orientations using advanced indexing
            batch_size = gt_locations.shape[0]
            batch_indices = torch.arange(batch_size, device=device)
            
            k_positions_tensor = k_positions_tensor.permute(1, 0, 2)  # (N, K, 3)
            # pred_positions = k_positions_tensor[batch_indices,preds]  # (N, 3)
            # pred_x, pred_y, pred_o = pred_positions[:, 0], pred_positions[:, 1], pred_positions[:, 2]
            
            # Gather ground truth positions and orientations
            gt_x, gt_y, gt_o = gt_locations[:, 0], gt_locations[:, 1], gt_locations[:, 2]
            
            # Calculate Euclidean distance for position accuracy
            # acc = torch.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)  # (N,)
            
            # Calculate orientation difference
            # acc_orn = angular_difference(pred_o, gt_o)  # (N,)
            
            # Move to CPU and convert to NumPy
            # acc_np = acc.cpu().numpy()
            # acc_orn_np = acc_orn.cpu().numpy()
            
            # Store the metrics for the model
            # acc_records_for_all_weights[weight_key].extend(acc_np.tolist())
            # acc_orn_records_for_all_weights[weight_key].extend(acc_orn_np.tolist())
            
            # **Compute Baseline Metrics (Always Select k=0)**
            # Select the first candidate
            # baseline_preds = torch.zeros(batch_size, dtype=torch.long, device=device)  # (N,)
            
            # Gather baseline predicted positions and orientations
            baseline_positions = k_positions_tensor[:, 0, :].to(device)  # (N, 3)
            baseline_x, baseline_y, baseline_o = baseline_positions[:, 0], baseline_positions[:, 1], baseline_positions[:, 2]
            
            # Calculate Euclidean distance for baseline
            acc_baseline = torch.sqrt((baseline_x - gt_x) ** 2 + (baseline_y - gt_y) ** 2)  # (N,)
            
            # Convert both baseline and ground truth orientations from radians to degrees
            baseline_o_deg = torch.rad2deg(baseline_o)
            gt_o_deg = torch.rad2deg(gt_o)

            # Calculate orientation difference for baseline using the function that expects degrees
            acc_orn_baseline_deg = angular_difference(baseline_o_deg, gt_o_deg)
        
            # Move to CPU and convert to NumPy    
            acc_baseline_np = acc_baseline.cpu().numpy()
            acc_orn_baseline_deg = acc_orn_baseline_deg.cpu().numpy()
            
            # Store the baseline metrics
            acc_records_baseline.extend(acc_baseline_np.tolist())
            acc_orn_records_baseline.extend(acc_orn_baseline_deg.tolist())
            
            # **Compute Random Depth Ray Baseline Metrics**
            # Initialize list to hold selected indices
            selected_indices = []
            
            for i in range(batch_size):
                metadata_path = metadata_paths[i]
                metadata = load_metadata(metadata_path)
                if metadata is None:
                    # If metadata cannot be loaded, default to k=0
                    selected_indices.append(0)
                    continue
                
                # Assuming metadata contains entries for each K, e.g., "K1", "K2", ..., "K5"
                k1_depth_rays = metadata.get("K1", {}).get("depth_rays", [])
                k1_semantic_rays = metadata.get("K1", {}).get("semantic_rays", [])
                
                # Compare each K with K1
                eligible_indices = [0]
                for k in range(1,K):
                    k_key = f"K{k+1}"  # "K1", "K2", ..., "K5"
                    if k_key == "K1":
                        continue  # Skip K1
                    k_depth_rays = metadata.get(k_key, {}).get("depth_rays", [])
                    k_semantic_rays = metadata.get(k_key, {}).get("semantic_rays", [])
                    if compare_rays(k1_depth_rays, k_depth_rays, k1_semantic_rays, k_semantic_rays):
                        eligible_indices.append(k)
                
                if len(eligible_indices)>1:
                    random_count +=1
                    random_choices_len.append(len(eligible_indices))
                    selected_idx = random.choice(eligible_indices)
                    random_selected.append(selected_idx)
                else:
                    selected_idx = 0
                selected_indices.append(selected_idx)
            
            # Convert selected indices to tensor
            selected_indices_tensor = torch.tensor(selected_indices, dtype=torch.long, device=device)  # (N,)
            
            # Gather randomly selected positions and orientations
            random_positions = torch.gather(k_positions_tensor, 1, selected_indices_tensor.unsqueeze(1).unsqueeze(2).expand(-1,1,3)).squeeze(1)  # (N, 3)
            random_x, random_y, random_o = random_positions[:, 0], random_positions[:, 1], random_positions[:, 2]
            
            # Calculate Euclidean distance for random depth ray baseline
            acc_random = torch.sqrt((random_x - gt_x) ** 2 + (random_y - gt_y) ** 2)  # (N,)
            
            # Calculate orientation difference for random depth ray baseline
            acc_orn_random = angular_difference(random_o, gt_o)  # (N,)
            
            # Move to CPU and convert to NumPy
            acc_random_np = acc_random.cpu().numpy()
            acc_orn_random_np = acc_orn_random.cpu().numpy()
            
            # Store the random depth ray baseline metrics
            acc_records_random_depth.extend(acc_random_np.tolist())
            acc_orn_records_random_depth.extend(acc_orn_random_np.tolist())
            
            #random score
            selected_indices = []
            for i in range(batch_size):
                k0_score = k_scores[0][i]                      
                # Compare each K with K1
                eligible_indices = [0]
                for k in range(1,5):
                    k_i_score = k_scores[k][i]
                    if abs(k_i_score - k0_score) < 0.001:
                        eligible_indices.append(k)    
                
                if len(eligible_indices)>1:   
                    random_score_counter +=1                       
                    selected_idx = random.choice(eligible_indices)
                else:
                    selected_idx = 0
                selected_indices.append(selected_idx)
            
            # Convert selected indices to tensor
            selected_indices_tensor = torch.tensor(selected_indices, dtype=torch.long, device=device)  # (N,)
            
            # Gather randomly selected positions and orientations
            random_positions = torch.gather(k_positions_tensor, 1, selected_indices_tensor.unsqueeze(1).unsqueeze(2).expand(-1,1,3)).squeeze(1)  # (N, 3)
            random_x, random_y, random_o = random_positions[:, 0], random_positions[:, 1], random_positions[:, 2]
            
            # Calculate Euclidean distance for random depth ray baseline
            acc_random = torch.sqrt((random_x - gt_x) ** 2 + (random_y - gt_y) ** 2)  # (N,)
            
            # Calculate orientation difference for random depth ray baseline
            acc_orn_random = angular_difference(random_o, gt_o)  # (N,)
            
            # Move to CPU and convert to NumPy
            acc_random_np = acc_random.cpu().numpy()
            acc_orn_random_np = acc_orn_random.cpu().numpy()
            
            # Store the random depth ray baseline metrics
            acc_records_random_score.extend(acc_random_np.tolist())
            acc_orn_records_random_score.extend(acc_orn_random_np.tolist())
            
            # **Compute Top K Metrics (Top 2 to Top K)**
            for top_k, acc_records, acc_orn_records in [
                (2, acc_records_top2, acc_orn_records_top2),
                (3, acc_records_top3, acc_orn_records_top3),
                (5, acc_records_top5, acc_orn_records_top5),
                (10, acc_records_top10, acc_orn_records_top10),
            ]:
                if top_k > K:
                    logging.warning(f"Requested top_k={top_k} exceeds available candidates K={K}. Skipping.")
                    continue

                # Get top K predictions
                topk_indices = torch.arange(0, top_k, device=device)
                # Gather top K positions and orientations            
                topk_positions = torch.gather(
                    k_positions_tensor,
                    1,
                    topk_indices.unsqueeze(0).unsqueeze(-1).expand(k_positions_tensor.size(0), -1, 3)
                )  # Resulting shape: (N, top_k, 3) 
                
                topk_x = topk_positions[:, :, 0]  # (N, top_k)
                topk_y = topk_positions[:, :, 1]  # (N, top_k)
                topk_o = topk_positions[:, :, 2]  # (N, top_k)
                
                # Calculate Euclidean distance for top K
                acc_topk = torch.sqrt((topk_x - gt_x.unsqueeze(1)) ** 2 + (topk_y - gt_y.unsqueeze(1)) ** 2)  # (N, top_k)
                
                # Find the minimum acc and corresponding index
                min_acc, min_indices = acc_topk.min(dim=1)  # Both are (N,)
                
                # Gather the orientation corresponding to the minimum acc
                min_o = torch.gather(topk_o, 1, min_indices.unsqueeze(1)).squeeze(1)  # (N,)
                
                # Convert both baseline and ground truth orientations from radians to degrees
                min_o_deg = torch.rad2deg(min_o)
                gt_o_deg = torch.rad2deg(gt_o)

                # Calculate orientation difference for baseline using the function that expects degrees
                acc_orn = angular_difference(min_o_deg, gt_o_deg)
                
                # Store the minimum acc and corresponding orientation
                acc_records.extend(min_acc.cpu().numpy().tolist())
                acc_orn_records.extend(acc_orn.cpu().numpy().tolist())
    
    # After the loop, process all metrics
    
    # Convert lists to numpy arrays
    for key in acc_records_for_all_weights.keys():
        acc_records_for_all_weights[key] = np.array(acc_records_for_all_weights[key])
    for key in acc_orn_records_for_all_weights.keys():
        acc_orn_records_for_all_weights[key] = np.array(acc_orn_records_for_all_weights[key])
    
    acc_records_baseline = np.array(acc_records_baseline)
    acc_orn_records_baseline = np.array(acc_orn_records_baseline)
    
    acc_records_random_depth = np.array(acc_records_random_depth)
    acc_orn_records_random_depth = np.array(acc_orn_records_random_depth)
    
    acc_records_random_score = np.array(acc_records_random_score)
    acc_orn_records_random_score= np.array(acc_orn_records_random_score)
    
    acc_records_top2 = np.array(acc_records_top2)
    acc_orn_records_top2 = np.array(acc_orn_records_top2)
    
    acc_records_top3 = np.array(acc_records_top3)
    acc_orn_records_top3 = np.array(acc_orn_records_top3)
        
    acc_records_top5 = np.array(acc_records_top5)
    acc_orn_records_top5 = np.array(acc_orn_records_top5)
    
    acc_records_top10 = np.array(acc_records_top10)
    acc_orn_records_top10 = np.array(acc_orn_records_top10)
    
    # Calculate recalls for the model
    recalls_model = {}
    for key in acc_records_for_all_weights.keys():
        acc_record = acc_records_for_all_weights[key]
        acc_orn_record = acc_orn_records_for_all_weights.get(key, None)
        recalls = calculate_recalls(acc_record, acc_orn_record)
        recalls_model[key] = recalls
        logging.info(f"Recalls for model (weight key '{key}'):")
        for metric, value in recalls.items():
            logging.info(f"  {metric}: {value*100:.2f}%")
    
    # Calculate recalls for the baseline (always selecting k=0)
    recalls_baseline = calculate_recalls(acc_records_baseline, acc_orn_records_baseline)
    logging.info("Recalls for baseline (always selecting k=0):")
    for metric, value in recalls_baseline.items():
        logging.info(f"  {metric}: {value*100:.2f}%")
    
    # Calculate recalls for the random depth ray baseline
    recalls_random_depth = calculate_recalls(acc_records_random_depth, acc_orn_records_random_depth)
    logging.info("Recalls for random depth ray baseline (selecting randomly based on depth ray differences):")
    for metric, value in recalls_random_depth.items():
        logging.info(f"  {metric}: {value*100:.2f}%")
    
    recalls_random_score = calculate_recalls(acc_records_random_score, acc_orn_records_random_score)
    logging.info("Recalls for random depth ray baseline (selecting randomly based on depth ray differences):")
    for metric, value in recalls_random_score.items():
        logging.info(f"  {metric}: {value*100:.2f}%")
    
    # Calculate recalls for Top 2, Top 3, Top 4, and Top 5
    recalls_top = {}
    for top_k, acc_records, acc_orn_records, label in [
        (2, acc_records_top2, acc_orn_records_top2, "Top 2"),
        (3, acc_records_top3, acc_orn_records_top3, "Top 3"),
        (5, acc_records_top5, acc_orn_records_top5, "Top 5"),
        (10, acc_records_top10, acc_orn_records_top10, "Top 10"),
    ]:
        recalls = calculate_recalls(acc_records, acc_orn_records)
        recalls_top[label] = recalls
        logging.info(f"Recalls for {label}:")
        for metric, value in recalls.items():
            logging.info(f"  {metric}: {value*100:.2f}%")
    
    # **Generate LaTeX Table**
    # Define the table header
    latex_table = r"""
\begin{table}[ht]
    \centering
    \begin{tabular}{lcccc}
        \hline
        \textbf{Method} & \textbf{0.1m} & \textbf{0.5m} & \textbf{1m} & \textbf{1m 30°} \\
        \hline
    """    
    # Add baseline recalls
    row_baseline = "Baseline (Top 1) & " + " & ".join([
        f"{recalls_baseline['0.1m']*100:.2f}",
        f"{recalls_baseline['0.5m']*100:.2f}",
        f"{recalls_baseline['1m']*100:.2f}",
        f"{recalls_baseline['1m 30 deg']*100:.2f}"
    ]) + r" \\"
    latex_table += "\n    " + row_baseline
    
    for top_label in ["Top 2", "Top 3", "Top 5", "Top 10"]:
        recalls = recalls_top.get(top_label, None)
        if recalls is None:
            continue
        row = f"{top_label} & " + " & ".join([
            f"{recalls['0.1m']*100:.2f}",
            f"{recalls['0.5m']*100:.2f}",
            f"{recalls['1m']*100:.2f}",
            f"{recalls['1m 30 deg']*100:.2f}"
        ]) + r" \\"
        latex_table += "\n    " + row
    
    # Add model recalls
    for key, recalls in recalls_model.items():
        row = f"Model ({key}) & " + " & ".join([
            f"{recalls['0.1m']*100:.2f}",
            f"{recalls['0.5m']*100:.2f}",
            f"{recalls['1m']*100:.2f}",
            f"{recalls['1m 30 deg']*100:.2f}"
        ]) + r" \\"
        latex_table += "\n    " + row
        
    # Add random depth ray baseline recalls
    row_random = "Random Ray Baseline & " + " & ".join([
        f"{recalls_random_depth['1m']*100:.2f}",
        f"{recalls_random_depth['0.5m']*100:.2f}",
        f"{recalls_random_depth['0.1m']*100:.2f}",
        f"{recalls_random_depth['1m 30 deg']*100:.2f}"
    ]) + r" \\"
    latex_table += "\n    " + row_random
    
    
    row_random_score = "Random score Baseline & " + " & ".join([
        f"{recalls_random_score['1m']*100:.2f}",
        f"{recalls_random_score['0.5m']*100:.2f}",
        f"{recalls_random_score['0.1m']*100:.2f}",
        f"{recalls_random_score['1m 30 deg']*100:.2f}"
    ]) + r" \\"
    latex_table += "\n    " + row_random_score
    
    # Close the table
    latex_table += """
        \hline
    \end{tabular}
    \caption{Recall Metrics for Different Methods}
    \label{tab:recall_metrics}
\end{table}
"""

    print(latex_table)
    
    # Optionally, save the metrics to a file along with the LaTeX table
    metrics_path = os.path.join(default_root_dir, "evaluation_metrics.txt")
    latex_path = os.path.join(default_root_dir, "recall_metrics_table.tex")
    try:
        with open(metrics_path, "w") as f:
            for key in acc_records_for_all_weights.keys():
                acc_record = acc_records_for_all_weights[key]
                acc_orn_record = acc_orn_records_for_all_weights.get(key, None)
                recalls = calculate_recalls(acc_record, acc_orn_record)
                f.write(f"Recalls for model (weight key '{key}'):\n")
                for metric, value in recalls.items():
                    f.write(f"  {metric}: {value*100:.2f}%\n")
            # Write baseline metrics (always selecting k=0)
            f.write("Recalls for baseline (always selecting k=0):\n")
            for metric, value in recalls_baseline.items():
                f.write(f"  {metric}: {value*100:.2f}%\n")
            # Write random depth ray baseline metrics
            f.write("Recalls for random depth ray baseline (selecting randomly based on depth ray differences):\n")
            for metric, value in recalls_random_depth.items():
                f.write(f"  {metric}: {value*100:.2f}%\n")
            # Write Top 2, Top 3, Top 4, Top 5 metrics
            for top_label in ["Top 2", "Top 3", "Top 4", "Top 5"]:
                recalls = recalls_top.get(top_label, None)
                if recalls is None:
                    continue
                f.write(f"Recalls for {top_label}:\n")
                for metric, value in recalls.items():
                    f.write(f"  {metric}: {value*100:.2f}%\n")
        logging.info(f"Evaluation metrics saved to {metrics_path}")
    except Exception as e:
        logging.error(f"Error saving evaluation metrics: {e}")
    
    # Save the LaTeX table to a file
    try:
        with open(latex_path, "w") as f:
            f.write(latex_table)
        logging.info(f"LaTeX table saved to {latex_path}")
    except Exception as e:
        logging.error(f"Error saving LaTeX table: {e}")

if __name__ == "__main__":
    main()
