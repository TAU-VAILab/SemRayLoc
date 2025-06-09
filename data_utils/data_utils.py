"""
Dataset Module for Grid Sequence Data

This module provides a PyTorch Dataset implementation for handling grid sequence data,
supporting both Structured3D and ZInD datasets. It handles loading and processing of
images, depth maps, semantic labels, poses, and room information.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import logging
from utils.utils import gravity_align
from modules.semantic.semantic_mapper import room_type_to_id, zind_room_type_to_id

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Camera intrinsics for Structured3D dataset
K = np.array([
    [320/np.tan(0.698132), 0,               320],
    [0,                   180/np.tan(0.440992), 180],
    [0,                   0,               1]
], dtype=np.float32)

class LocalizationDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        scene_names: List[str],
        data_dir: Optional[str] = None,
        start_scene: Optional[int] = None,
        end_scene: Optional[int] = None,
    ):
        """Initialize the dataset.
        
        Args:
            dataset_dir: Base directory containing the dataset
            scene_names: List of scene names to include
            data_dir: Optional directory for additional data (defaults to dataset_dir)
            start_scene: Optional index to start from
            end_scene: Optional index to end at
        """
        super().__init__()
        self.dataset_dir = dataset_dir
        self.scene_names = scene_names
        self.data_dir = data_dir or dataset_dir
        self.start_scene = start_scene
        self.end_scene = end_scene
        
        # Initialize data containers
        self.scene_start_idx: List[int] = []
        self.gt_values: List[Dict[str, List]] = []
        self.gt_pose: List[List[np.ndarray]] = []
        self.gt_room_label: List[List[str]] = []
        self.room_polygons: List[Dict[str, List]] = []

        self._load_scene_data()
        logger.info(f"Number of valid scenes: {len(self.scene_names)}")
        
        # Calculate total length
        self.total_len = sum(len(poses) for poses in self.gt_pose)

    def __len__(self) -> int:
        """Get the total number of samples in the dataset."""
        return self.total_len

    def _load_scene_data(self) -> None:
        """Load and process data for all scenes."""
        self.scene_start_idx.append(0)
        start_idx = 0
        valid_scene_names = []

        for scene_idx, scene in enumerate(self.scene_names):
            is_zind = 'floor' in scene
            scene_number = int(scene.split('_')[1]) if not is_zind else None
            scene_name = f"scene_{scene_number}" if not is_zind else scene

            # Load scene data
            try:
                scene_data = self._load_single_scene(scene_name, is_zind)
                if scene_data is None:
                    continue

                # Update indices and store data
                valid_scene_names.append(scene_name)
                start_idx += len(scene_data['depth'])
                self.scene_start_idx.append(start_idx)

                self.gt_values.append({
                    "depth": scene_data['depth'],
                    "semantic": scene_data['semantic'],
                    "pitch": scene_data['pitch'],
                    "roll": scene_data['roll']
                })
                self.gt_pose.append(scene_data['poses'])
                self.gt_room_label.append(scene_data['room_labels'])
                self.room_polygons.append(scene_data['polygons'])

            except Exception as e:
                logger.error(f"Error processing scene {scene}: {e}")
                continue

        self.scene_names = valid_scene_names

    def _load_single_scene(self, scene_name: str, is_zind: bool) -> Optional[Dict]:
        """Load data for a single scene.
        
        Args:
            scene_name: Name of the scene to load
            is_zind: Whether this is a ZInD dataset scene
            
        Returns:
            Dictionary containing scene data or None if loading fails
        """
        scene_folder = os.path.join(self.data_dir, scene_name)
        
        # Define file paths
        file_paths = {
            'depth': os.path.join(scene_folder, "depth.txt"),
            'semantic': os.path.join(scene_folder, "semantic.txt"),
            'pitch': os.path.join(scene_folder, "pitch.txt"),
            'roll': os.path.join(scene_folder, "roll.txt"),
            'poses': os.path.join(scene_folder, "poses.txt"),
            'room_label': os.path.join(scene_folder, scene_name, 
                "room_type_per_image_mapped.txt" if is_zind else "room_type_per_image.txt"),
            'room_rectangles': os.path.join(scene_folder, scene_name,
                "room_types_rectangles_mapped.json" if is_zind else "room_types_rectangles.json")
        }

        try:
            # Load all text files
            data = {}
            for key, path in file_paths.items():
                if key == 'room_rectangles':
                    with open(path, 'r') as f:
                        rectangles_data = json.load(f)
                        data[key] = {entry["room_type"]: entry["polygons"] 
                                   for entry in rectangles_data}
                else:
                    with open(path, 'r') as f:
                        data[key] = [line.strip() for line in f.readlines()]

            # Validate data consistency
            if not all(len(data['depth']) == len(data[key]) 
                      for key in ['semantic', 'room_label']):
                logger.error(f"Inconsistent data lengths in scene {scene_name}")
                return None

            # Process numerical data
            processed_data = {
                'depth': [np.array([float(d) for d in line.split()], dtype=np.float32)
                         for line in data['depth']],
                'semantic': [np.array([float(s) for s in line.split()], dtype=np.float32)
                           for line in data['semantic']],
                'poses': [np.array([float(p) for p in line.split()], dtype=np.float32)
                         for line in data['poses']],
                'pitch': [float(p) for p in data['pitch']],
                'roll': [float(r) for r in data['roll']],
                'room_labels': data['room_label'],
                'polygons': data['room_rectangles']
            }

            return processed_data

        except Exception as e:
            logger.error(f"Error reading data for {scene_name}: {e}")
            return None

    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to get
            
        Returns:
            Dictionary containing the sample data
        """
        if self.start_scene is not None:
            idx += self.scene_start_idx[self.start_scene]

        # Find scene and local index
        scene_idx = np.sum(idx >= np.array(self.scene_start_idx)) - 1
        scene_name = self.scene_names[scene_idx]
        is_zind = 'floor' in scene_name
        idx_within_scene = idx - self.scene_start_idx[scene_idx]

        # Get basic data
        data_dict = {
            "scene_name": scene_name,
            "idx_within_scene": idx_within_scene,
            "ref_depth": self.gt_values[scene_idx]["depth"][idx_within_scene],
            "ref_semantics": self.gt_values[scene_idx]["semantic"][idx_within_scene],
            "ref_pose": self.gt_pose[scene_idx][idx_within_scene],
            "ref_noise": 0,
            "ref_pitch": self.gt_values[scene_idx]["pitch"][idx_within_scene],
            "ref_roll": self.gt_values[scene_idx]["roll"][idx_within_scene]
        }

        # Process room label
        room_label_str = self.gt_room_label[scene_idx][idx_within_scene]
        room_label_id = (zind_room_type_to_id if is_zind else room_type_to_id).get(
            room_label_str, 
            (zind_room_type_to_id if is_zind else room_type_to_id)["undefined"]
        )
        data_dict["room_label"] = torch.tensor(room_label_id, dtype=torch.long)
        
        # Add room polygons
        data_dict["room_polygons"] = self.room_polygons[scene_idx]

        # Load and process image
        image_path = os.path.join(
            self.dataset_dir,
            scene_name,
            "rgb",
            f"{idx_within_scene}.png"
        )
        
        ref_img = self._load_and_process_image(image_path, is_zind, 
                                             data_dict["ref_roll"], 
                                             data_dict["ref_pitch"])
        data_dict.update(ref_img)

        return data_dict

    def _load_and_process_image(
        self, 
        image_path: str, 
        is_zind: bool,
        roll: float,
        pitch: float
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """Load and process an image with gravity alignment if needed.
        
        Args:
            image_path: Path to the image file
            is_zind: Whether this is a ZInD dataset image
            roll: Camera roll angle
            pitch: Camera pitch angle
            
        Returns:
            Dictionary containing processed image and mask
        """
        # Load image
        ref_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if ref_img is None:
            raise FileNotFoundError(f"Image not found or could not be read at path: {image_path}")
        
        # Convert to RGB and normalize
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB) / 255.0

        if not is_zind:
            # Apply gravity alignment for Structured3D
            ref_img = gravity_align(ref_img, r=roll, p=pitch, K=K)
            
            # Create and align mask
            mask = np.ones(ref_img.shape[:2], dtype=np.float32)
            mask = gravity_align(mask, r=roll, p=pitch, K=K)
            mask[mask < 1] = 0
            ref_mask = mask.astype(np.uint8)
        else:
            # ZInD dataset: use full mask
            ref_mask = np.ones(ref_img.shape[:2], dtype=np.uint8)

        # Convert image to channel-first format
        ref_img = np.transpose(ref_img, (2, 0, 1)).astype(np.float32)

        return {
            "ref_img": ref_img,
            "ref_mask": ref_mask
        }
