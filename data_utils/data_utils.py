import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import json
import torch
from utils.utils import gravity_align
from modules.semantic.semantic_mapper import room_type_to_id, zind_room_type_to_id
# Hardcoded intrinsics for Structured3D cameras
K = np.array([
    [320/np.tan(0.698132), 0,               320],
    [0,                   180/np.tan(0.440992), 180],
    [0,                   0,               1]
], dtype=np.float32)

class GridSeqDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        scene_names,
        L,
        data_dir=None,
        roll=0,
        pitch=0,
        start_scene=None,
        end_scene=None,
        max_h=3000,
        max_w=3000,
        augment=False,      # <--- NEW: flag for photometric augmentation
        noise_std=0.01,     # <--- NEW: Gaussian noise standard deviation
        room_data_dir='/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/test_data_set_full/structured3d_perspective_full_with_room_types',
        pano_dir=None,
        is_train=True
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.scene_names = scene_names
        self.L = L
        self.data_dir = data_dir or dataset_dir
        self.roll = roll
        self.pitch = pitch
        self.start_scene = start_scene
        self.end_scene = end_scene
        self.max_h = max_h
        self.max_w = max_w
        self.room_data_dir = room_data_dir

        # Photometric augmentation flags
        self.augment = augment
        self.noise_std = noise_std
        self.color_jitter = T.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.05
        )

        self.scene_start_idx = []
        self.gt_values = []   # depth, semantic, pitch, roll
        self.gt_pose = []     # camera poses

        # NEW: Store room labels per image and polygons per room type
        self.gt_room_label = []   # for each scene, list of N labels (one per image)
        self.room_polygons = []   # for each scene, dict { "bedroom": [bbox1, bbox2,...], ... }

        # Load info for all scenes
        self.load_scene_start_idx_and_values_and_poses()
        print("number of valid scenes: ", len(self.scene_names))
        # Calculate total length from all scenes
        self.total_len = 0
        for poses in self.gt_pose:
            self.total_len += len(poses)

    def __len__(self):
        return self.total_len

    def load_scene_start_idx_and_values_and_poses(self):
        self.scene_start_idx.append(0)
        start_idx = 0
        valid_scene_names = []
        is_zind = False
        for scene_idx, scene in enumerate(self.scene_names):
            # If "scene_00000" etc. => get the integer for S3D
            if 'floor' in scene:
                is_zind = True
                # ZInD dataset logic, if relevant
                pass
            else:
                # Structured3D
                scene_number = int(scene.split('_')[1])  # e.g. "scene_0" => 0
                scene = f"scene_{scene_number}"

            # Paths to data for this scene
            scene_folder = os.path.join(self.data_dir, scene)
            # depth_file = os.path.join(scene_folder, "depth_closed_doors.txt") #closed_doors
            depth_file = os.path.join(scene_folder, "depth.txt") #closed_doors
            semantic_file = os.path.join(scene_folder, "semantic.txt")
            pitch_file = os.path.join(scene_folder, "pitch.txt")
            roll_file = os.path.join(scene_folder, "roll.txt")
            pose_file = os.path.join(scene_folder, "poses.txt")

            # Also paths to room label + polygons
            # (in your "room_data_dir" instead of "data_dir")
            room_label_file = os.path.join(self.room_data_dir, scene, "room_type_per_image.txt")
            room_rectangles_file = os.path.join(self.room_data_dir, scene, "room_types_rectangles.json")
            
            if is_zind:
                room_label_file = os.path.join(self.room_data_dir, scene, "room_type_per_image_mapped.txt")
                room_rectangles_file = os.path.join(self.room_data_dir, scene, "room_types_rectangles_mapped.json")
                

            try:
                # 1) Depth / semantic
                with open(depth_file, "r") as f:
                    depth_txt = [line.strip() for line in f.readlines()]
                with open(semantic_file, "r") as f:
                    semantic_txt = [line.strip() for line in f.readlines()]

                # 2) Pose
                with open(pose_file, "r") as f:
                    poses_txt = [line.strip() for line in f.readlines()]

                # 3) Pitch / Roll
                with open(pitch_file, "r") as f:
                    pitch_txt = [float(line.strip()) for line in f.readlines()]
                with open(roll_file, "r") as f:
                    roll_txt = [float(line.strip()) for line in f.readlines()]

                # 4) Room label (one per image line)
                with open(room_label_file, "r") as f:
                    room_label_lines = [line.strip() for line in f.readlines()]

                # 5) Room polygons (JSON structure)
                with open(room_rectangles_file, "r") as f:
                    rectangles_data = json.load(f)
                # Convert to a dict: rtype -> list_of_bboxes
                polygons_dict = {}
                for entry in rectangles_data:
                    rtype = entry["room_type"]
                    polygons_dict[rtype] = entry["polygons"]

                # 6) Check floorplan dimensions
                floorplan_file = os.path.join(scene_folder, "floorplan_semantic.png")
                with Image.open(floorplan_file) as img:
                    width, height = img.size
                    if width > self.max_w or height > self.max_h:
                        print(f"Scene {scene} has floorplan_semantic.png with dimensions "
                              f"{width}x{height}, skipping this scene.")
                        continue

            except Exception as e:
                print(f"Error reading data for {scene}: {e}. Skipping.")
                continue

            traj_len = len(poses_txt)
            # Optional check that everything matches
            if not (len(depth_txt) == len(semantic_txt) == traj_len == len(room_label_lines)):
                print(f"Inconsistent data lengths in scene {scene}. Skipping.")
                continue

            # Build arrays for depth, semantic, pitch, roll, poses
            scene_depths = []
            scene_semantics = []
            scene_poses = []
            scene_pitch = []
            scene_roll = []

            for state_id in range(traj_len):
                # Depth
                depth_vals = depth_txt[state_id].split(" ")
                depth_arr = np.array([float(d) for d in depth_vals], dtype=np.float32)
                scene_depths.append(depth_arr)

                # Semantic
                semantic_vals = semantic_txt[state_id].split(" ")
                semantic_arr = np.array([float(s) for s in semantic_vals], dtype=np.float32)
                scene_semantics.append(semantic_arr)

                # Pose
                pose_vals = poses_txt[state_id].split(" ")
                pose_arr = np.array([float(s) for s in pose_vals], dtype=np.float32)
                scene_poses.append(pose_arr)

                # Pitch / Roll
                scene_pitch.append(pitch_txt[state_id])
                scene_roll.append(roll_txt[state_id])

            # Store them
            if is_zind:
                valid_scene_names.append(scene)
            else:
                valid_scene_names.append(f"scene_{scene_number}")
            start_idx += traj_len
            self.scene_start_idx.append(start_idx)

            self.gt_values.append({
                "depth": scene_depths,
                "semantic": scene_semantics,
                "pitch": scene_pitch,
                "roll": scene_roll
            })
            self.gt_pose.append(scene_poses)

            # Also store the room labels and polygons
            self.gt_room_label.append(room_label_lines)  # list of strings, length=traj_len
            self.room_polygons.append(polygons_dict)     # dict {rtype: [bbox1, bbox2,...], ... }

        self.scene_names = valid_scene_names

    def __getitem__(self, idx):
        # Possibly offset by start_scene
        if self.start_scene is not None:
            idx += self.scene_start_idx[self.start_scene]

        # Which scene does this index belong to?
        scene_idx = np.sum(idx >= np.array(self.scene_start_idx)) - 1
        scene_name = self.scene_names[scene_idx]
        is_zind = ('floor' in scene_name)  # simplistic check

        # Index within the scene
        idx_within_scene = idx - self.scene_start_idx[scene_idx]

        # Retrieve data
        ref_depth = self.gt_values[scene_idx]["depth"][idx_within_scene]
        ref_semantics = self.gt_values[scene_idx]["semantic"][idx_within_scene]
        ref_pose = self.gt_pose[scene_idx][idx_within_scene]
        ref_pitch = self.gt_values[scene_idx]["pitch"][idx_within_scene]
        ref_roll = self.gt_values[scene_idx]["roll"][idx_within_scene]

        data_dict = {
            "scene_name": scene_name,
            "idx_within_scene": idx_within_scene,
            "ref_depth": ref_depth,
            "ref_semantics": ref_semantics,
            "ref_pose": ref_pose,
            "ref_noise": 0,
            "ref_pitch": ref_pitch,
            "ref_roll": ref_roll
        }

        # ------------------------------------------------------------------
        # GET ROOM LABEL AND CORRESPONDING POLYGONS
        # ------------------------------------------------------------------
        room_label_str = self.gt_room_label[scene_idx][idx_within_scene]
        if is_zind:
            room_label_id = zind_room_type_to_id.get(room_label_str, zind_room_type_to_id["undefined"])
        else:
            room_label_id = room_type_to_id.get(room_label_str, room_type_to_id["undefined"])
        data_dict["room_label"] = torch.tensor(room_label_id, dtype=torch.long)
                
        polygons_dict = self.room_polygons[scene_idx]  # e.g. { "bedroom": [...], "living room": [...], ... }
        data_dict["room_polygons"] = polygons_dict

        # ------------------------------------------------------------------
        # LOAD reference image from dataset_dir
        # ------------------------------------------------------------------
        image_path = os.path.join(
            self.dataset_dir,
            scene_name,
            "rgb",
            f"{idx_within_scene}.png"
        )
        ref_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if ref_img is None:
            raise FileNotFoundError(f"Image not found or could not be read at path: {image_path}")
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB) / 255.0

        # Gravity align (S3D only) - if scene_name is not from ZInD
        if not is_zind:
            r = ref_roll
            p = ref_pitch
            ref_img = gravity_align(ref_img, r=r, p=p, K=K)

            # Build mask similarly
            mask = np.ones(list(ref_img.shape[:2]), dtype=np.float32)
            mask = gravity_align(mask, r=r, p=p, K=K)
            mask[mask < 1] = 0
            ref_mask = mask.astype(np.uint8)
            data_dict["ref_mask"] = ref_mask
        else:
            # ZInD dataset: just use full 1.0 mask
            mask = np.ones(list(ref_img.shape[:2]), dtype=np.uint8)
            data_dict["ref_mask"] = mask

        # --------------------------------------------------
        # OPTIONAL Photometric Augmentations
        # --------------------------------------------------
        if self.augment:
            ref_img = self.apply_photometric_augmentations(ref_img)

        # Convert HWC -> CHW, float32
        ref_img = np.transpose(ref_img, (2, 0, 1)).astype(np.float32)
        data_dict["ref_img"] = ref_img

        return data_dict

    def apply_photometric_augmentations(self, image_np):
        """
        image_np: NumPy array of shape (H, W, 3), values in [0,1].
        Applies color jitter and optional Gaussian noise.
        Returns augmented NumPy array (H, W, 3) in [0,1].
        """
        # Convert NumPy [0..1] -> PIL [0..255]
        pil_img = Image.fromarray((image_np * 255).astype(np.uint8))

        # 1) Color jitter
        pil_img = self.color_jitter(pil_img)

        # Convert back to NumPy float in [0,1]
        augmented = np.array(pil_img, dtype=np.float32) / 255.0

        # 2) Gaussian noise
        if self.noise_std > 0:
            noise = np.random.randn(*augmented.shape) * self.noise_std
            augmented = augmented + noise
            augmented = np.clip(augmented, 0.0, 1.0)

        return augmented
