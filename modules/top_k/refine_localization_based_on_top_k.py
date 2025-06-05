import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
from attrdict import AttrDict
from datetime import datetime
import numpy as np
import logging
import random
import json 
from modules.mono.depth_net_pl import depth_net_pl
from modules.semantic.semantic_net_pl import semantic_net_pl
from top_k_dataset import TopKDataset
NUM_OF_SCENES = -1
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
    scenes = process_scene_names(scenes)[:NUM_OF_SCENES]
    
    dataset = TopKDataset(
        scene_names=scenes,
        image_base_dir=dataset_dir,
        top_k_dir=config.top_k_results_dir,
        poses_filename=config.poses_filename,
        enforce_fixed_resolution=True,
        target_resolution=(360, 640),
        desired_candidate_dim=(5, 40)
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

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Argument parsing
    parser = argparse.ArgumentParser(description="Evaluation script for best_k_net with new candidate selection baselines.")
    parser.add_argument(
        "--config",
        type=str,
        default="modules/top_k/config_refine_top_k.yaml",
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
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    default_root_dir = config.ckpt_path
    os.makedirs(default_root_dir, exist_ok=True)
    logging.info(f"Logging evaluation to: {default_root_dir}")
    
    # Setup dataset and DataLoader for test split.
    test_dataset, test_loader = setup_dataset_and_loader(config, split.test)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    depth_net = depth_net_pl.load_from_checkpoint(
        checkpoint_path=config.log_dir_depth,
        d_min=config.d_min,
        d_max=config.d_max,
        d_hyp=config.d_hyp,
        D=config.D,
    ).to(device)
    semantic_net = semantic_net_pl.load_from_checkpoint(
        checkpoint_path=config.log_dir_semantic,
        num_classes=config.num_classes,
    ).to(device)
    
    # Initialize metric containers for baseline (always picking k=0) and the new comparisons.
    acc_records_baseline = []
    acc_orn_records_baseline = []
    
    acc_records_depth_only = []
    acc_orn_records_depth_only = []
    
    acc_records_sem_only = []
    acc_orn_records_sem_only = []
    
    acc_records_combined = []
    acc_orn_records_combined = []

    acc_records_combined_proportional = []
    acc_orn_records_combined_proportional = []
    
    # Iterate over test data
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Move data to device
            ref_imgs = batch["ref_img"].to(device)              # (N, 3, 360, 640)
            depth_vecs = batch["depth_vec"].to(device)            # (N, 5, 40)
            sem_vecs = batch["sem_vec"].to(device)                # (N, 5, 40)
            semantic_maps = batch["semantic_map"].to(device)      # (N, 1, 300, 300)
            gt_locations = batch["gt_location"].to(device)        # (N, 3)
            k_positions = batch["k_positions"]                    # list of N elements, each list of 5 [x, y, o]
            k_scores_batch = batch["k_scores"]                    # list of N elements, each list of 5 scores
            metadata_paths = batch["metadata_path"]               # list of N metadata file paths
            best_index = batch["best_index"]                      # (N,)
            k_scores = batch["k_scores"]                          # (N, 5)
            
            # Compute network predictions for depth and semantic rays
            # (pred_depths: (N, 40))
            ref_img_torch = ref_imgs  # already on device
            pred_depths, _, _ = depth_net.encoder(ref_img_torch, None)
            # Compute semantic network output and sample an index per ray.
            _, _, prob = semantic_net.encoder(ref_img_torch, None)
            # prob shape assumed to be (N, 40, num_classes) or similar.
            # Squeeze extra dimensions if needed and sample indices.
            prob_squeezed = prob.squeeze(dim=0) if prob.shape[0] == 1 else prob
            sampled_indices = torch.multinomial(prob_squeezed.view(-1, prob_squeezed.shape[-1]), num_samples=1, replacement=True)
            sampled_indices = sampled_indices.view(prob_squeezed.shape[:-1])  # shape (N, 40)
            sampled_indices_np = sampled_indices.cpu().numpy() 
                    
            # Convert k_positions from list to tensor and permute so that candidate dimension is first.
            # Original k_positions is list of N elements, each is list of 5 candidates [x,y,o].
            k_positions = [[list(p) for p in pos] for pos in k_positions]  
            k_positions_tensor = torch.tensor(k_positions, dtype=torch.float32, device=device)        
            batch_size = len(batch["ref_img"])
            
            # Gather ground truth positions and orientations
            gt_x, gt_y, gt_o = gt_locations[:, 0], gt_locations[:, 1], gt_locations[:, 2]                        
            
   
            # ==============================
            # **Baseline: Always Select k=0**
            # ==============================
            baseline_positions = k_positions_tensor[0, :, :].to(device)  # candidate 0 for all images, shape (N, 3)
            baseline_x, baseline_y, baseline_o = baseline_positions[:, 0], baseline_positions[:, 1], baseline_positions[:, 2]

            # Calculate Euclidean distance for baseline
            acc_baseline = torch.sqrt((baseline_x - gt_x) ** 2 + (baseline_y - gt_y) ** 2)  # (N,)
            
            # Calculate orientation difference for baseline
            acc_orn_baseline = angular_difference(baseline_o, gt_o)  # (N,)
            
            # Move to CPU and convert to NumPy and store
            acc_records_baseline.extend(acc_baseline.cpu().numpy().tolist())
            acc_orn_records_baseline.extend(acc_orn_baseline.cpu().numpy().tolist())
            
            # ==============================================================
            # **New Comparisons: Candidate Selection using Predicted Rays**
            # ==============================================================
            # For each image in the batch, load the metadata and compare the 40 rays.
            # We assume the metadata JSON contains keys like "K1", "K2", ... corresponding to the order in k_positions.
            for i in range(batch_size):
                metadata = load_metadata(metadata_paths[i])
                if metadata is None:
                    continue  # skip if metadata loading fails
                # Sort candidate keys (e.g., "K1", "K2", ...) in ascending order.
                candidate_keys = sorted(metadata.keys(), key=lambda x: int(x[1:]))
                
                errors_depth = []
                errors_sem = []
                errors_comb = []
                errors_proportional = []
                # For each candidate, compute the error between the network–predicted rays and candidate rays.
                for candidate_key in candidate_keys:
                    candidate_info = metadata[candidate_key]
                    # Extract candidate's 40 depth ray distances.
                    candidate_depth_rays = np.array([ray["distance_m"] for ray in candidate_info["depth_rays"]])
                    # Extract candidate's 40 semantic ray predictions.
                    candidate_sem_rays = np.array([ray["prediction_class"] for ray in candidate_info["semantic_rays"]])
                    candidate_sem_rays[candidate_sem_rays == 2] = -1
                    candidate_sem_rays[candidate_sem_rays == 1] = 2
                    candidate_sem_rays[candidate_sem_rays == -1] = 1
                    # Get network predicted depth and semantic for image i (both are length 40)
                    pred_depth_i = pred_depths[i].cpu().numpy()      # (40,)
                    pred_sem_i = sampled_indices_np[i]                 # (40,)
                    
                    # Compute mean absolute error for depth
                    error_depth = np.mean(np.abs(pred_depth_i - candidate_depth_rays) > 1.5)
                    # Compute fraction of mismatches for semantic rays
                    error_sem = np.mean((pred_sem_i != candidate_sem_rays).astype(np.float32))
                    # Combined error: simple average of the two (you may wish to normalize if needed)
                    error_comb = 0.4 * error_depth + 0.6 * error_sem
                    error_proportional = 0.5 * error_depth + 0.5 * error_sem
                    
                    errors_depth.append(error_depth)
                    errors_sem.append(error_sem)
                    errors_comb.append(error_comb)
                    errors_proportional.append(error_proportional)
                
                # Choose the candidate (i.e. index) with the minimum error for each method.
                best_depth_idx = int(np.argmin(errors_depth))
                best_sem_idx   = int(np.argmin(errors_sem))
                best_comb_idx  = int(np.argmin(errors_comb))
                best_proportional_idx  = int(np.argmin(errors_proportional))
                
                # Retrieve the candidate predicted position (from k_positions_tensor) for each selection.
                # k_positions_tensor has shape (5, N, 3) where index 0 corresponds to candidate "K1", etc.
                candidate_pos_depth = k_positions_tensor[best_depth_idx, i, :]  # (3,)
                candidate_pos_sem   = k_positions_tensor[best_sem_idx, i, :]
                candidate_pos_comb  = k_positions_tensor[best_comb_idx, i, :]
                candidate_pos_proportional  = k_positions_tensor[best_proportional_idx, i, :]
                
                # Get ground truth location for image i.
                gt_pos = gt_locations[i]
                gt_x_i = gt_pos[0].item()
                gt_y_i = gt_pos[1].item()
                gt_o_i = gt_pos[2].item()
                
                # ----- Depth Only Selection -----
                cand_x_depth = candidate_pos_depth[0].item()
                cand_y_depth = candidate_pos_depth[1].item()
                cand_o_depth = candidate_pos_depth[2].item()
                error_dist_depth = np.sqrt((cand_x_depth - gt_x_i)**2 + (cand_y_depth - gt_y_i)**2)
                error_orient_depth = angular_difference(torch.tensor(cand_o_depth), torch.tensor(gt_o_i)).item()
                acc_records_depth_only.append(error_dist_depth)
                acc_orn_records_depth_only.append(error_orient_depth)
                
                # ----- Semantic Only Selection -----
                cand_x_sem = candidate_pos_sem[0].item()
                cand_y_sem = candidate_pos_sem[1].item()
                cand_o_sem = candidate_pos_sem[2].item()
                error_dist_sem = np.sqrt((cand_x_sem - gt_x_i)**2 + (cand_y_sem - gt_y_i)**2)
                error_orient_sem = angular_difference(torch.tensor(cand_o_sem), torch.tensor(gt_o_i)).item()
                acc_records_sem_only.append(error_dist_sem)
                acc_orn_records_sem_only.append(error_orient_sem)
                
                # ----- Combined Selection (0.5*depth + 0.5*semantic) -----
                cand_x_comb = candidate_pos_comb[0].item()
                cand_y_comb = candidate_pos_comb[1].item()
                cand_o_comb = candidate_pos_comb[2].item()
                error_dist_comb = np.sqrt((cand_x_comb - gt_x_i)**2 + (cand_y_comb - gt_y_i)**2)
                error_orient_comb = angular_difference(torch.tensor(cand_o_comb), torch.tensor(gt_o_i)).item()
                acc_records_combined.append(error_dist_comb)
                acc_orn_records_combined.append(error_orient_comb)

                # ----- Combined Selection (0.5*depth + 0.5*semantic) -----
                cand_x_proportional = candidate_pos_proportional[0].item()
                cand_y_proportional = candidate_pos_proportional[1].item()
                cand_o_proportional = candidate_pos_proportional[2].item()
                error_dist_proportional = np.sqrt((cand_x_proportional - gt_x_i)**2 + (cand_y_proportional - gt_y_i)**2)
                error_orient_proportional = angular_difference(torch.tensor(cand_o_proportional), torch.tensor(gt_o_i)).item()
                acc_records_combined_proportional.append(error_dist_proportional)
                acc_orn_records_combined_proportional.append(error_orient_proportional)
    
    # After all batches, convert error lists to numpy arrays.
    acc_records_baseline = np.array(acc_records_baseline)
    acc_orn_records_baseline = np.array(acc_orn_records_baseline)
    
    acc_records_depth_only = np.array(acc_records_depth_only)
    acc_orn_records_depth_only = np.array(acc_orn_records_depth_only)
    
    acc_records_sem_only = np.array(acc_records_sem_only)
    acc_orn_records_sem_only = np.array(acc_orn_records_sem_only)
    
    acc_records_combined = np.array(acc_records_combined)
    acc_orn_records_combined = np.array(acc_orn_records_combined)
    
    acc_records_combined_proportional = np.array(acc_records_combined_proportional)
    acc_orn_records_combined_proportional = np.array(acc_orn_records_combined_proportional)
    
    # Calculate recalls for each method.
    recalls_baseline = calculate_recalls(acc_records_baseline, acc_orn_records_baseline)
    recalls_depth = calculate_recalls(acc_records_depth_only, acc_orn_records_depth_only)
    recalls_sem = calculate_recalls(acc_records_sem_only, acc_orn_records_sem_only)
    recalls_combined = calculate_recalls(acc_records_combined, acc_orn_records_combined)
    recalls_combined_proportional = calculate_recalls(acc_records_combined_proportional, acc_orn_records_combined_proportional)
    
    logging.info("Recalls for baseline (always selecting k=0):")
    for metric, value in recalls_baseline.items():
        logging.info(f"  {metric}: {value*100:.2f}%")
    
    logging.info("Recalls for Depth Only selection:")
    for metric, value in recalls_depth.items():
        logging.info(f"  {metric}: {value*100:.2f}%")
    
    logging.info("Recalls for Semantic Only selection:")
    for metric, value in recalls_sem.items():
        logging.info(f"  {metric}: {value*100:.2f}%")
    
    logging.info("Recalls for Combined (0.5 depth + 0.5 semantic) selection:")
    for metric, value in recalls_combined.items():
        logging.info(f"  {metric}: {value*100:.2f}%")

    logging.info("Recalls for Combined proportional selection:")
    for metric, value in recalls_combined_proportional.items():
        logging.info(f"  {metric}: {value*100:.2f}%")
    
    # ========================
    # **Generate LaTeX Table**
    # ========================
    latex_table = r"""
\begin{table}[ht]
    \centering
    \begin{tabular}{lcccc}
        \hline
        \textbf{Method} & \textbf{1m} & \textbf{0.5m} & \textbf{0.1m} & \textbf{1m 30°} \\
        \hline
"""
    # Baseline (always selecting k=0)
    row_baseline = "Baseline (Top 1) & " + " & ".join([
        f"{recalls_baseline['1m']*100:.2f}",
        f"{recalls_baseline['0.5m']*100:.2f}",
        f"{recalls_baseline['0.1m']*100:.2f}",
        f"{recalls_baseline['1m 30 deg']*100:.2f}"
    ]) + r" \\"
    latex_table += "\n    " + row_baseline

    # Depth Only
    row_depth = "Depth Only & " + " & ".join([
        f"{recalls_depth['1m']*100:.2f}",
        f"{recalls_depth['0.5m']*100:.2f}",
        f"{recalls_depth['0.1m']*100:.2f}",
        f"{recalls_depth['1m 30 deg']*100:.2f}"
    ]) + r" \\"
    latex_table += "\n    " + row_depth

    # Semantic Only
    row_sem = "Semantic Only & " + " & ".join([
        f"{recalls_sem['1m']*100:.2f}",
        f"{recalls_sem['0.5m']*100:.2f}",
        f"{recalls_sem['0.1m']*100:.2f}",
        f"{recalls_sem['1m 30 deg']*100:.2f}"
    ]) + r" \\"
    latex_table += "\n    " + row_sem

    # Combined
    row_comb = "Combined 1 & " + " & ".join([
        f"{recalls_combined['1m']*100:.2f}",
        f"{recalls_combined['0.5m']*100:.2f}",
        f"{recalls_combined['0.1m']*100:.2f}",
        f"{recalls_combined['1m 30 deg']*100:.2f}"
    ]) + r" \\"
    latex_table += "\n    " + row_comb

    # Combined
    row_comb = "Combined 2 & " + " & ".join([
        f"{recalls_combined_proportional['1m']*100:.2f}",
        f"{recalls_combined_proportional['0.5m']*100:.2f}",
        f"{recalls_combined_proportional['0.1m']*100:.2f}",
        f"{recalls_combined_proportional['1m 30 deg']*100:.2f}"
    ]) + r" \\"
    latex_table += "\n    " + row_comb

    # Close the table
    latex_table += r"""
        \hline
    \end{tabular}
    \caption{Recall Metrics for Different Candidate Selection Methods}
    \label{tab:recall_metrics}
\end{table}
"""
    # Print the LaTeX table
    print(latex_table)
    
    # Optionally, save the metrics to a file along with the LaTeX table
    metrics_path = os.path.join(default_root_dir, "evaluation_metrics.txt")
    latex_path = os.path.join(default_root_dir, "recall_metrics_table.tex")
    try:
        with open(metrics_path, "w") as f:            
            f.write("Recalls for baseline (always selecting k=0):\n")
            for metric, value in recalls_baseline.items():
                f.write(f"  {metric}: {value*100:.2f}%\n")
            f.write("\nRecalls for Depth Only selection:\n")
            for metric, value in recalls_depth.items():
                f.write(f"  {metric}: {value*100:.2f}%\n")
            f.write("\nRecalls for Semantic Only selection:\n")
            for metric, value in recalls_sem.items():
                f.write(f"  {metric}: {value*100:.2f}%\n")
            f.write("\nRecalls for Combined selection:\n")
            for metric, value in recalls_combined.items():
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
