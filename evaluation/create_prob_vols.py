import argparse
import os
import numpy as np
import torch
import tqdm
import yaml
from attrdict import AttrDict
import gzip
import logging

# Import your models
from modules.mono.depth_net_pl import depth_net_pl
from modules.semantic.semantic_net_pl import semantic_net_pl
from modules.semantic.semantic_net_pl_maskformer_small import semantic_net_pl_maskformer_small

# Import your dataset
from data_utils.data_utils import GridSeqDataset

# Import helper functions from utils
from utils.localization_utils import (
    get_ray_from_depth,
    get_ray_from_semantics,
    get_ray_from_semantics_v2,
    localize,
)
from utils.data_loader_helper import load_scene_data

# Global variables for models and data
depth_net = None
semantic_net = None
shared_test_set = None
shared_desdfs = None
shared_semantics = None
shared_gt_poses = None
shared_prob_vol_save_dir = None
shared_config = None


def initializer(
    log_dir_depth,
    log_dir_semantic,
    config,
    test_set,
    desdfs,
    semantics,
    gt_poses,
    prob_vol_save_dir,
):
    """
    Initializer function.
    Loads the models and assigns them to global variables.
    Also shares dataset and related data to minimize pickling.
    """
    global depth_net
    global semantic_net
    global shared_test_set
    global shared_desdfs
    global shared_semantics
    global shared_gt_poses
    global shared_prob_vol_save_dir
    global shared_config

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load depth_net
        depth_net = depth_net_pl.load_from_checkpoint(
            checkpoint_path=log_dir_depth,
            d_min=config.d_min,
            d_max=config.d_max,
            d_hyp=config.d_hyp,
            D=config.D,
        ).to(device)
        depth_net.eval()

        # Load semantic_net
        if config.use_mask_former_semantics:            
            semantic_net = semantic_net_pl_maskformer_small.load_from_checkpoint(
                checkpoint_path= config.maskformer_semantics_path,
                num_classes=config.num_classes,
            ).to(device)
            semantic_net.eval()
        else:
            semantic_net = semantic_net_pl.load_from_checkpoint(
                checkpoint_path= log_dir_semantic,
                num_classes=config.num_classes,
            ).to(device)
            semantic_net.eval()

        # Assign shared data
        shared_test_set = test_set
        shared_desdfs = desdfs
        shared_semantics = semantics
        shared_gt_poses = gt_poses
        shared_prob_vol_save_dir = prob_vol_save_dir
        shared_config = config

        logging.info("Models and data loaded successfully.")
    except Exception as e:
        logging.error(f"Error during initialization: {e}")


def save_prob_vol_torch_compressed(prob_vol, save_path, file_name):
    """
    Saves the probability volume as a compressed .pt.gz file.

    Args:
        prob_vol (torch.Tensor): Probability volume tensor.
        save_path (str): Directory where the file will be saved.
        file_name (str): Name of the file without extension.
    """
    try:
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, file_name + ".pt.gz")

        # Save with gzip compression
        with gzip.GzipFile(file_path, "wb") as f:
            torch.save(prob_vol.cpu(), f, _use_new_zipfile_serialization=True)
    except Exception as e:
        logging.error(f"Error saving file {file_name}: {e}")


def process_data(data_idx):
    """
    Processes a single data index and saves probability volumes as compressed .pt.gz files.

    Args:
        data_idx (int): Index of the data point to process.
    """
    global depth_net
    global semantic_net
    global shared_test_set
    global shared_desdfs
    global shared_semantics
    global shared_gt_poses
    global shared_prob_vol_save_dir
    global shared_config

    try:
        # Ensure models and data are loaded
        if depth_net is None or semantic_net is None or shared_test_set is None:
            raise RuntimeError("Models or data are not initialized.")

        # Use the device where the models are loaded
        device = next(depth_net.parameters()).device

        data = shared_test_set[data_idx]
        scene_idx = np.sum(data_idx >= np.array(shared_test_set.scene_start_idx)) - 1
        scene_name = shared_test_set.scene_names[scene_idx]
        scene_number = int(scene_name.split("_")[1])
        scene = f"scene_{scene_number}"
        idx_within_scene = data_idx - shared_test_set.scene_start_idx[scene_idx]

        try:
            desdf = shared_desdfs[scene]
            semantic = shared_semantics[scene]
        except KeyError:
            logging.warning(
                f"Scene data not available for scene: {scene}. Skipping data index {data_idx}."
            )
            return  # Skip if scene data is not available

        # Create save path for the scene
        scene_save_dir = os.path.join(shared_prob_vol_save_dir, scene)
        # Use idx_within_scene as camera identifier
        camera_id = f"camera_{idx_within_scene}"

        # Process predicted depth
        ref_img_torch = (
            torch.as_tensor(data["ref_img"], device=device)
            .unsqueeze(0)
        )
        with torch.no_grad():
            pred_depths, _, _ = depth_net.encoder(ref_img_torch, None)
        pred_depths = pred_depths.squeeze(0).cpu().numpy()
        pred_rays_depth = get_ray_from_depth(pred_depths)

        # Process predicted semantics
        with torch.no_grad():
            _, _, prob = semantic_net.encoder(ref_img_torch, None)
        prob_squeezed = prob.squeeze(dim=0)
        sampled_indices = torch.multinomial(
            prob_squeezed, num_samples=1, replacement=True
        )
        sampled_indices = sampled_indices.squeeze(dim=1)
        sampled_indices_np = sampled_indices.cpu().numpy()
        pred_rays_semantic = get_ray_from_semantics_v2(sampled_indices_np)

        # Generate probability volumes
        # Predicted Depth Probability Volume
        prob_vol_pred_depth, _, _, _ = localize(
            torch.tensor(shared_desdfs[scene]["desdf"]),
            torch.tensor(pred_rays_depth),
            return_np=False,
        )
        # Predicted Semantic Probability Volume
        prob_vol_pred_semantic, _, _, _ = localize(
            torch.tensor(shared_semantics[scene]["desdf"]),
            torch.tensor(pred_rays_semantic),
            return_np=False,
            localize_type="semantic",
        )

        # Ground Truth Depth Probability Volume
        gt_depths = data["ref_depth"]
        gt_rays_depth = get_ray_from_depth(gt_depths)
        prob_vol_gt_depth, _, _, _ = localize(
            torch.tensor(shared_desdfs[scene]["desdf"]),
            torch.tensor(gt_rays_depth),
            return_np=False,
        )

        # Ground Truth Semantic Probability Volume
        gt_semantics = data["ref_semantics"]
        gt_rays_semantic = get_ray_from_semantics_v2(gt_semantics)
        prob_vol_gt_semantic, _, _, _ = localize(
            torch.tensor(shared_semantics[scene]["desdf"]),
            torch.tensor(gt_rays_semantic),
            return_np=False,
            localize_type="semantic",
        )

        # Save the probability volumes
        save_prob_vol_torch_compressed(
            prob_vol_pred_depth, scene_save_dir, f"{camera_id}_pred_depth_prob_vol"
        )
        save_prob_vol_torch_compressed(
            prob_vol_pred_semantic, scene_save_dir, f"{camera_id}_pred_semantic_prob_vol",
        )
        save_prob_vol_torch_compressed(
            prob_vol_gt_depth, scene_save_dir, f"{camera_id}_gt_depth_prob_vol"
        )
        save_prob_vol_torch_compressed(
            prob_vol_gt_semantic, scene_save_dir, f"{camera_id}_gt_semantic_prob_vol"
        )

    except Exception as e:
        logging.error(f"Error processing data index {data_idx}: {e}")
        return


def generate_and_save_prob_volumes(
    desdfs,
    semantics,
    test_set,
    gt_poses,
    prob_vol_save_dir,
    config,
):
    """
    Generates and saves probability volumes sequentially on the GPU.

    Args:
        desdfs (dict): Depth and semantic data frames.
        semantics (dict): Semantic data.
        test_set (GridSeqDataset): Dataset object.
        gt_poses (dict): Ground truth poses.
        prob_vol_save_dir (str): Directory to save probability volumes.
        config (AttrDict): Configuration parameters.
    """
    logging.info("Processing data sequentially on GPU.")

    # Initialize models and shared data
    initializer(
        config.log_dir_depth,
        config.log_dir_semantic,
        config,
        test_set,
        desdfs,
        semantics,
        gt_poses,
        prob_vol_save_dir,
    )

    total = len(test_set)
    for data_idx in tqdm.tqdm(range(total), desc="Processing Data"):
        process_data(data_idx)

    logging.info("Probability volume generation completed.")


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    parser = argparse.ArgumentParser(description="Create probability volumes.")
    parser.add_argument(
        "--config_file",
        type=str,
        default="evaluation/configuration/config_create_prob_vols.yaml",
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    # Load configuration from file
    try:
        with open(args.config_file, "r") as f:
            config_dict = yaml.safe_load(f)
        config = AttrDict(config_dict)
        logging.info(f"Configuration loaded from {args.config_file}.")
    except Exception as e:
        logging.error(f"Failed to load configuration file {args.config_file}: {e}")
        return

    # Get device for main process
    main_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"======= USING DEVICE (Main Process): {main_device} =======")

    # Extract configuration parameters
    dataset_dir = config.dataset_dir
    desdf_path = config.desdf_path
    prob_vol_save_dir = config.prob_vol_save_dir

    # Instantiate dataset
    L = config.L
    start_scene = config.start_scene
    end_scene = config.end_scene

    scene_numbers = range(start_scene, end_scene + 1)
    scene_names = [f"scene_{str(i).zfill(5)}" for i in scene_numbers]
    # scene_names = ['scene_03261','scene_03279','scene_03280','scene_03452']

    try:
        test_set = GridSeqDataset(
            dataset_dir,
            scene_names,
            L=L,
        )
        logging.info(f"Dataset instantiated with {len(test_set)} data points.")
    except Exception as e:
        logging.error(f"Failed to instantiate dataset: {e}")
        return

    # Load desdf, semantics, maps, ground truth poses, and any additional data
    try:
        desdfs, semantics, _, gt_poses, _, _ = load_scene_data(
            test_set, dataset_dir, desdf_path
        )
        logging.info("Scene data loaded successfully.")
    except ValueError as ve:
        logging.error(f"Error unpacking load_scene_data: {ve}")
        raise ve
    except Exception as e:
        logging.error(f"Unexpected error loading scene data: {e}")
        raise e

    # Generate and save probability volumes sequentially
    generate_and_save_prob_volumes(
        desdfs,
        semantics,
        test_set,
        gt_poses,
        prob_vol_save_dir,
        config,
    )


if __name__ == "__main__":
    main()
