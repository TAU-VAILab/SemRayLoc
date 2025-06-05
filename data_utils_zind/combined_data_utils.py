import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.utils import gravity_align
import tqdm

K = np.array([[320/np.tan(0.698132), 0, 320],
              [0, 180/np.tan(0.440992), 180],
              [0, 0, 1]], dtype=np.float32)  # Hardcoded intrinsics

class CombinedDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        scene_names,
        L,
        desdf_path,
        data_dir=None,
        start_scene=None,
        end_scene=None,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.scene_names = scene_names
        self.L = L
        self.desdf_path = desdf_path  # Path to desdf maps
        self.data_dir = data_dir or dataset_dir
        self.start_scene = start_scene
        self.end_scene = end_scene
        self.scene_start_idx = []
        self.gt_values = []
        self.gt_pose = []
        self.load_scene_start_idx_and_values_and_poses()

        self.total_len = 0
        for poses in self.gt_pose:
            self.total_len+= len(poses)
            
    def __len__(self):
        return self.total_len        
    
    def load_scene_start_idx_and_values_and_poses(self):
        self.scene_start_idx.append(0)
        start_idx = 0
        valid_scene_names = []

        for scene_idx, scene in enumerate(tqdm.tqdm(self.scene_names)):
            scene_number = int(scene.split('_')[1])
            scene_name = f"scene_{scene_number}"
            scene_folder = os.path.join(self.data_dir, scene_name)

            depth_file = os.path.join(scene_folder, "depth.txt")
            semantic_file = os.path.join(scene_folder, "semantic.txt")
            pitch_file = os.path.join(scene_folder, "pitch.txt")
            roll_file = os.path.join(scene_folder, "roll.txt")
            
            pose_file = os.path.join(self.dataset_dir, scene_name, "poses.txt")
            desdf_file = os.path.join(self.desdf_path, scene_name, "desdf.npy")
            color_file = os.path.join(self.desdf_path, scene_name, "color.npy")

            # Check if all necessary files exist
            necessary_files = [depth_file, semantic_file, pitch_file, roll_file, pose_file, desdf_file]
            missing_files = [f for f in necessary_files if not os.path.exists(f)]
            if missing_files:
                print(f"Missing files for scene {scene_name}: {missing_files}, skipping this scene.")
                continue
            
            # **Step 2: Comprehensive Validation within a Single Try Block**
            try:
                # **File Size Validation**
                desdf_file_size = os.path.getsize(desdf_file)
                if desdf_file_size == 0:
                    print(f"Desdf file is empty for scene {scene_name}, skipping this scene.")
                    continue  # Skip this scene

                color_file_size = os.path.getsize(color_file)
                if color_file_size == 0:
                    print(f"Color file is empty for scene {scene_name}, skipping this scene.")
                    continue  # Skip this scene

                # **Load desdf and color data**
                desdf_data = np.load(desdf_file, allow_pickle=True).item()
                color_data = np.load(color_file, allow_pickle=True).item()

                # **Content Validation: Check for 'desdf' key**
                if "desdf" not in desdf_data:
                    print(f"Missing 'desdf' key in desdf_data for scene {scene_name}, skipping this scene.")
                    continue
                if "desdf" not in color_data:
                    print(f"Missing 'desdf' key in color_data for scene {scene_name}, skipping this scene.")
                    continue

                # **Shape Validation: Ensure arrays are not empty**
                if desdf_data["desdf"].size == 0:
                    print(f"Desdf data array is empty for scene {scene_name}, skipping this scene.")
                    continue

                if color_data["desdf"].size == 0:
                    print(f"Color data array is empty for scene {scene_name}, skipping this scene.")
                    continue

                # **Optional Shape Checks: Verify expected dimensions**
                # For example, ensure desdf has at least 1 dimension
                if len(desdf_data["desdf"].shape) < 1:
                    print(f"Desdf data array has invalid shape for scene {scene_name}, skipping this scene.")
                    continue

                if len(color_data["desdf"].shape) < 1:
                    print(f"Color data array has invalid shape for scene {scene_name}, skipping this scene.")
                    continue

            except Exception as e:
                print(f"Error during validation for scene {scene_name}: {e}. Skipping this scene.")
                continue  # Skip this scene on any validation failure

            try:
                # Load depth
                with open(depth_file, "r") as f:
                    depth_txt = [line.strip() for line in f.readlines()]

                # Load semantics
                with open(semantic_file, "r") as f:
                    semantic_txt = [line.strip() for line in f.readlines()]

                # Load poses
                with open(pose_file, "r") as f:
                    poses_txt = [line.strip() for line in f.readlines()]

                # Load pitch and roll
                with open(pitch_file, "r") as f:
                    pitch_txt = [float(line.strip()) for line in f.readlines()]

                with open(roll_file, "r") as f:
                    roll_txt = [float(line.strip()) for line in f.readlines()]
            except Exception as e:
                print(f"Error loading data for scene {scene_name}: {e}, skipping this scene.")
                continue

            traj_len = len(poses_txt)
            scene_depths = []
            scene_semantics = []
            scene_poses = []
            scene_pitch = []
            scene_roll = []

            for state_id in range(traj_len):
                # Get depth
                depth = depth_txt[state_id].split(" ")
                depth = np.array([float(d) for d in depth], dtype=np.float32)
                scene_depths.append(depth)

                # Get semantic
                semantic = semantic_txt[state_id].split(" ")
                semantic = np.array([float(s) for s in semantic], dtype=np.float32)
                scene_semantics.append(semantic)

                # Get pose
                pose = poses_txt[state_id].split(" ")
                pose = np.array([float(s) for s in pose], dtype=np.float32)
                scene_poses.append(pose)

                # Get pitch and roll
                scene_pitch.append(pitch_txt[state_id])
                scene_roll.append(roll_txt[state_id])

            valid_scene_names.append(self.scene_names[scene_idx])
            start_idx += traj_len // (self.L + 1)
            self.scene_start_idx.append(start_idx)

            # Store ground truth values
            self.gt_values.append({
                "depth": scene_depths,
                "semantic": scene_semantics,
                "pitch": scene_pitch,
                "roll": scene_roll
            })
            self.gt_pose.append(scene_poses)

        self.scene_names = valid_scene_names

    def __getitem__(self, idx):
        if self.start_scene is not None:
            idx += self.scene_start_idx[self.start_scene]

        # Get the scene index according to the idx
        scene_idx = np.sum(idx >= np.array(self.scene_start_idx)) - 1
        scene_name = self.scene_names[scene_idx]

        scene_number = int(scene_name.split('_')[1])
        scene_name = f"scene_{scene_number}"

        # Get idx within scene
        idx_within_scene = idx - self.scene_start_idx[scene_idx]

        # Compute the index for the reference image
        ref_idx = idx_within_scene

        # Get reference depth
        ref_depth = self.gt_values[scene_idx]["depth"][ref_idx]
        data_dict = {"ref_depth": ref_depth}

        # Get reference semantic
        ref_semantics = self.gt_values[scene_idx]["semantic"][ref_idx]
        data_dict["ref_semantics"] = ref_semantics

        # Get reference pose
        ref_pose = self.gt_pose[scene_idx][ref_idx]
        data_dict["ref_pose"] = ref_pose

        # Get pitch and roll
        ref_pitch = self.gt_values[scene_idx]["pitch"][ref_idx]
        ref_roll = self.gt_values[scene_idx]["roll"][ref_idx]
        data_dict["ref_pitch"] = ref_pitch
        data_dict["ref_roll"] = ref_roll
        
        # Get reference image
        image_path = os.path.join(
            self.dataset_dir,
            scene_name,
            "rgb",
            f"{ref_idx}.png",
        )
        ref_img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if ref_img is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB) / 255.0

        # Apply gravity alignment
        r = ref_roll
        p = ref_pitch
        ref_img = gravity_align(ref_img, r=r, p=p, K=K)

        mask = np.ones(list(ref_img.shape[:2]))
        mask = gravity_align(mask, r=r, p=p, K=K)            
        mask[mask < 1] = 0
        ref_mask = mask.astype(np.uint8)
        data_dict["ref_mask"] = ref_mask
        
        # From H,W,C to C,H,W
        ref_img = np.transpose(ref_img, (2, 0, 1)).astype(np.float32)
        data_dict["ref_img"] = ref_img
        
        desdf_file = os.path.join(self.desdf_path, scene_name, "desdf.npy")
        desdf_data = np.load(desdf_file, allow_pickle=True).item()
        data_dict["desdf"] =desdf_data
        
        color_file = os.path.join(self.desdf_path, scene_name, "color.npy")
        color_data = np.load(color_file, allow_pickle=True).item()
        data_dict["color"] =color_data        
    
        return data_dict
