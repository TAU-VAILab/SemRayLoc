import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import json
import torch
from modules.semantic.semantic_mapper import  zind_room_type_to_id
from utils.raycast_utils import ray_cast
import matplotlib.pyplot as plt
from data_utils.zind.zind_utils import *
    
    
# Hardcoded intrinsics for Structured3D cameras
K = np.array([
    [320/np.tan(0.698132), 0,               320],
    [0,                   180/np.tan(0.440992), 180],
    [0,                   0,               1]
], dtype=np.float32)

class LocalizationDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        scene_names,
        start_scene=None,
        end_scene=None,
        random_yaw=True,
        is_train=True,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.scene_names = scene_names
        self.start_scene = start_scene
        self.end_scene = end_scene
        self.random_yaw = random_yaw
        self.scene_start_idx = []
        self.gt_values = []   # depth, semantic, pitch, roll
        self.gt_pose = []     # camera poses
        self.metadata = []     # camera poses
        self.processed_data_dir = os.path.join(self.dataset_dir, "processed")
        self.is_train = is_train
        self.gt_room_label = []
        self.room_polygons = []
        self.gt_original_path = []
        self.pano_dir = os.path.join(self.dataset_dir, "raw_data")

        # Load info for all scenes
        self.load_scene_start_idx_and_values_and_poses()

        # Calculate total length from all scenes
        self.total_len = 0
        for poses in self.gt_pose:
            self.total_len += len(poses)
        if not self.is_train:
            print("random seed")
            np.random.seed(
                123456789
            )

    def __len__(self):
        return self.total_len

    def load_scene_start_idx_and_values_and_poses(self):
        self.scene_start_idx.append(0)
        start_idx = 0
        valid_scene_names = []
        for _, scene in enumerate(self.scene_names):
            # Paths to data for this scene
            scene_folder = os.path.join(self.processed_data_dir, scene)
            depth_file = os.path.join(scene_folder, "depth.txt")
            semantic_file = os.path.join(scene_folder, "semantic.txt")
            pitch_file = os.path.join(scene_folder, "pitch.txt")
            roll_file = os.path.join(scene_folder, "roll.txt")
            pose_file = os.path.join(scene_folder, "poses.txt")
            metadata_file = os.path.join(scene_folder, "metadata.json")
            room_label_file = os.path.join(scene_folder, "room_type_per_image_mapped.txt")
            room_rectangles_file = os.path.join(scene_folder, "room_types_rectangles_mapped.json")

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

                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                original_path = metadata["original_path"]

            except Exception as e:
                print(f"Error reading data for {scene}: {e}. Skipping.")
                continue

            # Optional check that everything matches
            if not (len(depth_txt) == len(semantic_txt) == len(room_label_lines)):
                print(f"Inconsistent data lengths in scene {scene}. Skipping.")
                continue

            # Build arrays for depth, semantic, pitch, roll, poses
            scene_depths = []
            scene_semantics = []
            scene_poses = []
            scene_pitch = []
            scene_roll = []

            for state_id in range(len(poses_txt)):
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
            valid_scene_names.append(scene)
            start_idx += len(poses_txt)

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
            self.gt_original_path.append(original_path)  # Save original_path from metadata


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
    
        ref_pose = self.gt_pose[scene_idx][idx_within_scene]            
        x, y, th = ref_pose
    
        pano_image_name= self.gt_original_path[scene_idx][idx_within_scene]
        parts = scene_name.split('_')
        scene_number = int(parts[1])
        scene_path = os.path.join(self.pano_dir, f"{scene_number:04d}")
        pano_image_path = os.path.join(scene_path, pano_image_name)
        pano_image = cv2.imread(pano_image_path, cv2.IMREAD_COLOR)
        if pano_image is None:
            raise FileNotFoundError(f"Image not found or could not be read at path: {pano_image_path}")
        pano_rot = np.rad2deg(th)
                  
        if self.random_yaw:
            yaw = np.random.rand() * 360
        else:
            yaw = 0           
        
        query_image = pano2persp(
                        pano_image, 80, yaw, 0, 0, (360,640)
                    )    

        pano_rot = pano_rot % 360
        pano_rot -= yaw
        
        ref_img = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB) / 255.0
        # Convert HWC -> CHW, float32
        ref_img = np.transpose(ref_img, (2, 0, 1)).astype(np.float32)
            
        #extract depth
        scene_dir = os.path.join(self.processed_data_dir, scene_name)
        sem_png = os.path.join(scene_dir, "floorplan_semantic.png")
        img_semantic = plt.imread(sem_png)        
        ray_n = 40
        F_W = 1 / np.tan(0.698132) / 2   
        
        center_angs = np.flip(np.arctan2((np.arange(ray_n) - np.arange(ray_n).mean()),ray_n * F_W))
        angs = center_angs + np.deg2rad(pano_rot)
        ref_pose[2] = np.deg2rad(pano_rot)

        depth_ref =[]
        semantic_ref = []
        hit_coords = []
        for _, ang in enumerate(angs):
            dist, pred_class, hit_coord = ray_cast(img_semantic, np.array([x*100, y*100]),
                                                ang, dist_max=15*100, min_dist=5)
            distance_adjusted = dist/100 
            depth_ref.append(distance_adjusted)
            semantic_ref.append(pred_class)                             
            hit_coords.append(hit_coord)                             
            
        data_dict = {
            "scene_name": scene_name,
            "idx_within_scene": idx_within_scene,
            "ref_depth": torch.tensor(depth_ref, dtype=torch.long),
            "ref_semantics": torch.tensor(semantic_ref, dtype=torch.int),
            "ref_pose": ref_pose,
            "ref_noise": 0,
            "ref_pitch": 0,
            "ref_roll": 0
        }
        
        data_dict["ref_img"] = ref_img
        data_dict["original_pano"] = query_image
        mask = np.ones(list(ref_img.shape[:2]), dtype=np.uint8)
        data_dict["ref_mask"] = mask

        room_label_str = self.gt_room_label[scene_idx][idx_within_scene]
        room_label_id = zind_room_type_to_id.get(room_label_str, zind_room_type_to_id["undefined"])

        data_dict["room_label"] = torch.tensor(room_label_id, dtype=torch.long)
                
        polygons_dict = self.room_polygons[scene_idx] 
        data_dict["room_polygons"] = polygons_dict

        return data_dict

   