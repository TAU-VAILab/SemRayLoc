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
from modules.semantic.semantic_net_pl_maskformer_small import semantic_net_pl_maskformer_small
from top_k_dataset import TopKDataset
from tqdm import tqdm

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
        if 'floor' in scene:
            processed.append(scene)
        else:
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
    Set up the TopKDataset and corresponding DataLoader.
    Here we use the candidate dimension from the configuration.
    """
    dataset_dir = os.path.join(config.dataset_path, config.dataset)
    scenes = process_scene_names(scenes)[:NUM_OF_SCENES]
    
    dataset = TopKDataset(
        scene_names=scenes,
        image_base_dir=dataset_dir,
        top_k_dir=config.top_k_results_dir,
        poses_filename=config.poses_filename,
        enforce_fixed_resolution=True,
        target_resolution=(360, 640),
        desired_candidate_dim=(config.num_candidates, 5, 40)
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
        "1m 30 deg": np.sum(np.logical_and(acc_record < 1, acc_orn_record < 30)) / acc_record.shape[0]
                     if acc_orn_record is not None else 0,
    }
    return recalls

def angular_difference(pred_o, gt_o):
    """
    Calculate the minimal angular difference between predictions and ground truth.
    Assumes angles are in degrees.
    """
    # Convert from radians to degrees
    pred_o_deg = torch.rad2deg(pred_o)
    gt_o_deg = torch.rad2deg(gt_o)
    diff = torch.abs(pred_o_deg - gt_o_deg) % 360
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
    parser = argparse.ArgumentParser(
        description="Evaluation script for candidate selection with configurable k and ray similarity-based scoring."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="modules/top_k/config_refine_top_k.yaml",
        help="Path to the YAML config file.",
    )
    args = parser.parse_args()

    # Enable CUDA_LAUNCH_BLOCKING for better error tracing
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Load configuration and dataset split
    config = load_config(args.config)
    split = load_split(config.split_file)

    # Create log directory (if needed)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    default_root_dir = config.ckpt_path
    os.makedirs(default_root_dir, exist_ok=True)
    logging.info(f"Logging evaluation to: {default_root_dir}")

    # Setup dataset and DataLoader for the test split.
    test_dataset, test_loader = setup_dataset_and_loader(config, split.test)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the pretrained networks
    depth_net = depth_net_pl.load_from_checkpoint(
        checkpoint_path=config.log_dir_depth,
        d_min=config.d_min,
        d_max=config.d_max,
        d_hyp=config.d_hyp,
        D=config.D,
    ).to(device)
    depth_net.eval()
    semantic_net = semantic_net_pl_maskformer_small.load_from_checkpoint(
        checkpoint_path=config.log_dir_semantic,
        num_classes=config.num_classes,
    ).to(device)
    semantic_net.eval()

    # Containers for error metrics:
    # - Baseline: always candidate K1 with augmentation 0
    # - Best Augmentation: selection using the best augmentation available per candidate
    acc_records_baseline = []
    acc_orn_records_baseline = []
    acc_records_best_aug = []
    acc_orn_records_best_aug = []

    # Get alpha parameter for weighting erroSrs
    alpha = config.get("alpha", 0)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating batches", total=len(test_loader))):
            # Load batch data
            ref_imgs = batch["ref_img"].to(device)              # (N, 3, 360, 640)
            # depth_vecs = batch["depth_vec"].to(device)            # (N, num_candidates, 40)
            # sem_vecs = batch["sem_vec"].to(device)                # (N, num_candidates, 40)
            semantic_maps = batch["semantic_map"].to(device)      # (N, 1, 300, 300)
            gt_locations = batch["gt_location"].to(device)        # (N, 3)
            k_positions = batch["k_positions"]                    # list of N elements, each a list of num_candidates [x, y, o]
            metadata_paths = batch["metadata_path"]               # list of N metadata file paths

            # Compute network predictions (rays) for depth and semantic features
            pred_depths, _, _ = depth_net.encoder(ref_imgs, None)  # (N, 40)
            _, _, prob = semantic_net.encoder(ref_imgs, None)
            # Adjust probability shape and sample semantic predictions
            prob_squeezed = prob.squeeze(dim=0) if prob.shape[0] == 1 else prob
            sampled_indices = torch.multinomial(
                prob_squeezed.view(-1, prob_squeezed.shape[-1]),
                num_samples=1,
                replacement=True
            )
            sampled_indices = sampled_indices.view(prob_squeezed.shape[:-1])  # (N, 40)
            sampled_indices_np = sampled_indices.cpu().numpy()

            # Convert k_positions into a tensor of shape (num_candidates, batch_size, 3)
            k_positions = [[list(p) for p in pos] for pos in k_positions]
            k_positions_tensor = torch.tensor(k_positions, dtype=torch.float32, device=device)

            batch_size = len(batch["ref_img"])
            gt_x, gt_y, gt_o = gt_locations[:, 0], gt_locations[:, 1], gt_locations[:, 2]

            # ---------- Baseline: Always select candidate K1 with augmentation 0 ----------
            # (Assuming candidate index 0 corresponds to K1.)
            baseline_positions = k_positions_tensor[0, :, :]  # (batch_size, 3)
            baseline_x, baseline_y, baseline_o = baseline_positions[:, 0], baseline_positions[:, 1], baseline_positions[:, 2]
            acc_baseline = torch.sqrt((baseline_x - gt_x) ** 2 + (baseline_y - gt_y) ** 2)  # (batch_size,)
            acc_orn_baseline = angular_difference(baseline_o, gt_o)  # (batch_size,)
            acc_records_baseline.extend(acc_baseline.cpu().numpy().tolist())
            acc_orn_records_baseline.extend(acc_orn_baseline.cpu().numpy().tolist())

            # ---------- Best Augmentation Selection based on Ray Similarity ----------
            # For each sample, load its metadata and search for the best augmentation (over all candidates)
            for i in range(batch_size):
                gt_pos = gt_locations[i]
                gt_x_i = gt_pos[0].item()
                gt_y_i = gt_pos[1].item()
                gt_o_i = gt_pos[2].item()

                metadata = load_metadata(metadata_paths[i])
                if metadata is None:
                    continue  # Skip if metadata cannot be loaded

                # Candidate keys are like "K1", "K2", ...; sort them to match candidate order.
                candidate_keys = sorted(metadata.keys(), key=lambda x: int(x[1:]))
                best_aug_score = float('inf')
                best_aug_candidate_index = None
                best_aug_center_angle = None
                # Loop over each candidate and each augmentation inside it
                for candidate_index, candidate_key in enumerate(candidate_keys[:5]):
                    candidate_info = metadata[candidate_key]
                    # Check if there is an "augmentations" key; if not, skip this candidate.
                    if "augmentations" not in candidate_info:
                        continue
                    for aug_key, aug_data in candidate_info["augmentations"].items():
                        # Use the augmentation rays instead of the top-level candidate rays.
                        candidate_center_angle = aug_data["center_angle"]
                        candidate_depth_rays = np.array([ray["distance_m"] for ray in aug_data["depth_rays"]])
                        candidate_sem_rays = np.array([ray["prediction_class"] for ray in aug_data["semantic_rays"]])
                        candidate_sem_rays[candidate_sem_rays == 2] = -1
                        candidate_sem_rays[candidate_sem_rays == 1] = 2
                        candidate_sem_rays[candidate_sem_rays == -1] = 1

                        pred_depth_i = pred_depths[i].cpu().numpy()
                        pred_sem_i = sampled_indices_np[i]
                        error_depth = np.mean(np.abs(pred_depth_i - candidate_depth_rays))
                        error_sem = np.mean((pred_sem_i != candidate_sem_rays).astype(np.float32))
                        candidate_score = alpha * error_depth + (1 - alpha) * error_sem

                        if candidate_score < best_aug_score:
                            best_aug_score = candidate_score
                            best_aug_candidate_index = candidate_index
                            best_aug_center_angle = candidate_center_angle

                if best_aug_candidate_index is None:
                    continue  # Skip if no augmentation was selected

                candidate_pos_best_aug = k_positions_tensor[best_aug_candidate_index, i, :]  # (3,)
                cand_x_aug = candidate_pos_best_aug[0].item()
                cand_y_aug = candidate_pos_best_aug[1].item()
                cand_o_aug = best_aug_center_angle
                error_dist_best_aug = np.sqrt((cand_x_aug - gt_x_i)**2 + (cand_y_aug - gt_y_i)**2)
                error_orient_best_aug = angular_difference(torch.tensor(cand_o_aug), torch.tensor(gt_o_i)).item()
                acc_records_best_aug.append(error_dist_best_aug)
                acc_orn_records_best_aug.append(error_orient_best_aug)

    # Convert error lists to numpy arrays.
    acc_records_baseline = np.array(acc_records_baseline)
    acc_orn_records_baseline = np.array(acc_orn_records_baseline)
    acc_records_best_aug = np.array(acc_records_best_aug)
    acc_orn_records_best_aug = np.array(acc_orn_records_best_aug)

    # Calculate recall metrics for both methods.
    recalls_baseline = calculate_recalls(acc_records_baseline, acc_orn_records_baseline)
    recalls_best_aug = calculate_recalls(acc_records_best_aug, acc_orn_records_best_aug)

    logging.info("Recalls for Baseline (K1 with Augmentation 0):")
    for metric, value in recalls_baseline.items():
        logging.info(f"  {metric}: {value*100:.2f}%")

    logging.info("Recalls for Best Augmentation Selection (using best augmentation per candidate):")
    for metric, value in recalls_best_aug.items():
        logging.info(f"  {metric}: {value*100:.2f}%")

    # ========================
    # **Generate LaTeX Table**
    # ========================
    latex_table = r"""
\begin{table}[ht]
    \centering
    \begin{tabular}{lcccc}
        \hline
        \textbf{Method} & \textbf{0.1m} & \textbf{0.5m} & \textbf{1m} & \textbf{1m 30°} \\
        \hline
"""
    row_baseline = "Baseline & " + " & ".join([
        f"{recalls_baseline['0.1m']*100:.2f}",
        f"{recalls_baseline['0.5m']*100:.2f}",
        f"{recalls_baseline['1m']*100:.2f}",
        f"{recalls_baseline['1m 30 deg']*100:.2f}"
    ]) + r" \\"
    latex_table += "\n    " + row_baseline

    row_best_aug = "Best Augmentation Selection & " + " & ".join([
        f"{recalls_best_aug['0.1m']*100:.2f}",
        f"{recalls_best_aug['0.5m']*100:.2f}",
        f"{recalls_best_aug['1m']*100:.2f}",
        f"{recalls_best_aug['1m 30 deg']*100:.2f}"
    ]) + r" \\"
    latex_table += "\n    " + row_best_aug

    latex_table += r"""
        \hline
    \end{tabular}
    \caption{Recall Metrics for Baseline and Best Augmentation Selection Methods}
    \label{tab:recall_metrics}
\end{table}
"""
    print(latex_table)

    # Optionally, save the metrics and LaTeX table to files.
    metrics_path = os.path.join(default_root_dir, "evaluation_metrics.txt")
    latex_path = os.path.join(default_root_dir, "recall_metrics_table.tex")
    try:
        with open(metrics_path, "w") as f:
            f.write("Recalls for Baseline (K1 with Augmentation 0):\n")
            for metric, value in recalls_baseline.items():
                f.write(f"  {metric}: {value*100:.2f}%\n")
            f.write("\nRecalls for Best Augmentation Selection:\n")
            for metric, value in recalls_best_aug.items():
                f.write(f"  {metric}: {value*100:.2f}%\n")
        logging.info(f"Evaluation metrics saved to {metrics_path}")
    except Exception as e:
        logging.error(f"Error saving evaluation metrics: {e}")

    try:
        with open(latex_path, "w") as f:
            f.write(latex_table)
        logging.info(f"LaTeX table saved to {latex_path}")
    except Exception as e:
        logging.error(f"Error saving LaTeX table: {e}")

if __name__ == "__main__":
    main()
