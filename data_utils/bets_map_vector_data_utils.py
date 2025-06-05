import os
import numpy as np
import torch
from torch.utils.data import Dataset
import tqdm
import gzip
from PIL import Image


K = np.array([[320/np.tan(0.698132), 0, 320],
              [0, 180/np.tan(0.440992), 180],
              [0, 0, 1]], dtype=np.float32)  # Hardcoded intrinsics

class BestMapVectorDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        scene_names,
        L,
        prob_vol_path,
        best_map_path,
        data_dir=None,
        start_scene=None,
        end_scene=None,
        narrow_gt_map = False,
        max_h= 3000,
        max_w= 3000
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.scene_names = scene_names
        self.L = L
        self.prob_vol_path = prob_vol_path 
        self.best_map_path = best_map_path 
        self.data_dir = data_dir or dataset_dir
        self.start_scene = start_scene
        self.end_scene = end_scene
        self.narrow_gt_map = narrow_gt_map
        self.scene_start_idx = []
        self.gt_values = []
        self.gt_pose = []
        self.max_h= max_h
        self.max_w= max_w
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

            pose_file = os.path.join(self.dataset_dir, scene_name, "poses.txt")
            floorplan_file = os.path.join(scene_folder, "floorplan_semantic.png")

            # Check if all necessary files exist
            necessary_files = [pose_file]
            missing_files = [f for f in necessary_files if not os.path.exists(f)]
            if missing_files:
                print(f"Missing files for scene {scene_name}: {missing_files}, skipping this scene.")
                continue
                    
            try:
                with Image.open(floorplan_file) as img:
                    width, height = img.size
                    if width > self.max_w or height > self.max_h:
                        print(f"Scene {scene_name} has floorplan_semantic.png with dimensions {width}x{height}, skipping this scene.")
                        continue
            except Exception as e:
                print(f"Error opening floorplan_semantic.png for scene {scene_name}: {e}, skipping this scene.")
                continue
            
            # Validate all images have corresponding pred_depth and pred_semantics
            missing_prediction_files = False
            invalid_dimensions = False
            image_path = os.path.join(self.dataset_dir, scene_name, "rgb")
            
            for ref_idx in range(len(os.listdir(image_path))):
                pred_depth_file = os.path.join(self.prob_vol_path, scene_name, f"camera_{ref_idx}_pred_depth_prob_vol.pt.gz")
                pred_semantics_file = os.path.join(self.prob_vol_path, scene_name, f"camera_{ref_idx}_pred_semantic_prob_vol.pt.gz")
                best_map_vector = os.path.join(self.best_map_path, scene_name, f"camera_{ref_idx}_pred_best.pt")

                # Check if prediction files exist
                if not (os.path.exists(pred_depth_file) and os.path.exists(pred_semantics_file) and os.path.exists(best_map_vector)):
                    print(f"Missing prediction files for scene {scene_name}, image {ref_idx}. Skipping this scene.")
                    missing_prediction_files = True
                    break

            if missing_prediction_files or invalid_dimensions:
                continue

            try:
                # Load poses
                with open(pose_file, "r") as f:
                    poses_txt = [line.strip() for line in f.readlines()]
            except Exception as e:
                print(f"Error loading data for scene {scene_name}: {e}, skipping this scene.")
                continue

            traj_len = len(poses_txt)
            scene_poses = []

            for state_id in range(traj_len):
                # Get pose
                pose = poses_txt[state_id].split(" ")
                pose = np.array([float(s) for s in pose], dtype=np.float32)
                scene_poses.append(pose)

            valid_scene_names.append(self.scene_names[scene_idx])
            start_idx += traj_len // (self.L + 1)
            self.scene_start_idx.append(start_idx)

            self.gt_pose.append(scene_poses)

        self.scene_names = valid_scene_names
        print(f"Number of scenes after filtering: {len(self.scene_names)}")

    
    def __getitem__(self, idx):        
        if self.start_scene is not None:
            idx += self.scene_start_idx[self.start_scene]
        
        data_dict = {}
        # Get the scene index according to the idx
        scene_idx = np.sum(idx >= np.array(self.scene_start_idx)) - 1
        scene_name = self.scene_names[scene_idx]

        scene_number = int(scene_name.split('_')[1])
        scene_name = f"scene_{scene_number}"
        data_dict['scene_name'] = scene_name
        
        # Get idx within scene
        idx_within_scene = idx - self.scene_start_idx[scene_idx]
        # Compute the index for the reference image
        ref_idx = idx_within_scene
        data_dict['ref_idx'] = ref_idx


        # # Get reference pose
        ref_pose = self.gt_pose[scene_idx][ref_idx]
        data_dict["ref_pose"] = ref_pose
        
        # Paths for compressed files
        pred_depth_gz_path = os.path.join(self.prob_vol_path, scene_name, f"camera_{ref_idx}_pred_depth_prob_vol.pt.gz")
        pred_semantics_gz_path = os.path.join(self.prob_vol_path, scene_name, f"camera_{ref_idx}_pred_semantic_prob_vol.pt.gz")
        pred_depth_gt_gz_path = os.path.join(self.prob_vol_path, scene_name, f"camera_{ref_idx}_gt_depth_prob_vol.pt.gz")
        pred_semantics_gt_gz_path = os.path.join(self.prob_vol_path, scene_name, f"camera_{ref_idx}_gt_semantic_prob_vol.pt.gz")
        
        with gzip.open(pred_depth_gz_path, 'rb') as f_in:
            prob_vol_depth = torch.load(f_in, weights_only=True)
            prob_vol_depth = prob_vol_depth.float()
            data_dict["prob_vol_depth"] = prob_vol_depth
        with gzip.open(pred_semantics_gz_path, 'rb') as f_in:
            prob_vol_semantic = torch.load(f_in, weights_only=True)
            prob_vol_semantic = prob_vol_semantic.float()
            data_dict["prob_vol_semantic"] = prob_vol_semantic
        with gzip.open(pred_depth_gt_gz_path, 'rb') as f_in:
            prob_vol_depth_gt = torch.load(f_in, weights_only=True)
            prob_vol_depth_gt = prob_vol_depth_gt.float()
            data_dict["prob_vol_depth_gt"] = prob_vol_depth_gt
        with gzip.open(pred_semantics_gt_gz_path, 'rb') as f_in:
            prob_vol_semantic_gt = torch.load(f_in, weights_only=True)
            prob_vol_semantic_gt = prob_vol_semantic_gt.float()
            data_dict["prob_vol_semantic_gt"] = prob_vol_semantic_gt
            
        best_map_vector_path = os.path.join(self.best_map_path, scene_name, f"camera_{ref_idx}_pred_best.pt")
        best_map_vector_path_gt = os.path.join(self.best_map_path, scene_name, f"camera_{ref_idx}_gt_best.pt")
        
        best_pred_map_vector = torch.load(best_map_vector_path, weights_only=True)
        best_gt_map_vector = torch.load(best_map_vector_path_gt, weights_only=True)
        
        data_dict["best_pred_map_vector"] = best_pred_map_vector
        data_dict["best_gt_map_vector"] = best_gt_map_vector
        return data_dict
    
    