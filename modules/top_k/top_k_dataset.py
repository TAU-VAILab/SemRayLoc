import os
import cv2
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F

class TopKDataset(Dataset):
    def __init__(self, 
                 scene_names,
                 image_base_dir="/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/test_data_set_full/structured3d_perspective_full",
                 top_k_dir="/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/modules/top_k/results",
                 poses_filename="poses.txt",
                 enforce_fixed_resolution=True,
                 target_resolution=(360, 640),  # (height, width)
                 desired_candidate_dim=(5, 40)):
    
        super().__init__()
        self.scene_names = scene_names
        self.image_base_dir = image_base_dir
        self.results_base_dir = top_k_dir
        self.poses_filename = poses_filename
        self.enforce_fixed_resolution = enforce_fixed_resolution
        self.target_resolution = target_resolution  # (height, width)
        self.desired_candidate_dim = desired_candidate_dim
        
        # Initialize dictionary to store semantic maps
        self.semantic_maps = {}  # dictionary of scene_name -> semantic_map (tensor)
        
        # Create a list of valid samples as tuples: (scene, image_index)
        # Also load the ground truth poses for each scene.
        self.samples = []   # list of (scene_name, image_index)
        self.gt_poses = {}  # dictionary of scene_name -> list of poses (each as a numpy array)

        # Initialize counter for skipped scenes
        self.skipped_scenes_due_to_size = 0

        self._prepare_dataset()
        
    def _prepare_dataset(self):
        for scene in self.scene_names:
            scene_image_dir = os.path.join(self.image_base_dir, scene, "rgb")
            poses_path = os.path.join(self.image_base_dir, scene, self.poses_filename)
            semantic_map_path = os.path.join(self.image_base_dir, scene, "floorplan_semantic.png")
            
            # Check existence of essential directories/files
            if not os.path.exists(scene_image_dir):
                print(f"Skipping scene {scene}: Missing image directory.")
                continue
            if not os.path.exists(poses_path):
                print(f"Skipping scene {scene}: Missing poses file.")
                continue
            if not os.path.exists(semantic_map_path):
                print(f"Warning: Semantic map not found for scene {scene} at {semantic_map_path}. Skipping scene.")
                self.skipped_scenes_due_to_size += 1
                continue

            # Load the semantic map
            sem_map = cv2.imread(semantic_map_path, cv2.IMREAD_GRAYSCALE)
            if sem_map is None:
                print(f"Warning: Unable to read semantic map for scene {scene}. Skipping scene.")
                self.skipped_scenes_due_to_size += 1
                continue

            original_height, original_width = sem_map.shape[:2]

            if original_height > 3500 or original_width > 3500:
                print(f"Skipping scene {scene}: Semantic map size {original_width}x{original_height} exceeds 3000x3000.")
                self.skipped_scenes_due_to_size += 1
                continue

            pad_height = 3500 - original_height
            pad_width = 3500 - original_width
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left

            sem_map_padded = cv2.copyMakeBorder(
                sem_map,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                borderType=cv2.BORDER_CONSTANT,
                value=255  # White padding
            )

            sem_map_resized = cv2.resize(
                sem_map_padded,
                (300, 300),  # (width, height)
                interpolation=cv2.INTER_NEAREST  # Use nearest for categorical data
            )

            # Normalize the semantic map if necessary (e.g., scaling to [0, 1])
            sem_map_normalized = sem_map_resized.astype(np.float32) / 255.0  # Adjust based on your specific requirements

            # Store the semantic map as a tensor with shape (1, 300, 300)
            self.semantic_maps[scene] = torch.from_numpy(sem_map_normalized).unsqueeze(0)  # shape: (1, H, W)

            # Load ground truth poses
            try:
                with open(poses_path, "r") as f:
                    lines = f.readlines()
            except Exception as e:
                print(f"Skipping scene {scene}: Failed to read poses file with error: {e}")
                continue

            poses = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                try:
                    pose_xyo = np.array([float(parts[0]), float(parts[1]),float(parts[2])], dtype=np.float32)
                except ValueError:
                    continue
                poses.append(pose_xyo)
            if not poses:
                print(f"Skipping scene {scene}: No valid poses found.")
                continue
            self.gt_poses[scene] = poses

            # List all image files
            img_files = sorted([f for f in os.listdir(scene_image_dir) if f.endswith('.png')])
            if not img_files:
                print(f"Skipping scene {scene}: No image files found.")
                continue

            for idx, img_file in enumerate(img_files):
                image_path = os.path.join(scene_image_dir, f"{idx}.png")
                # depth_path = os.path.join(self.results_base_dir, scene, f"image_{idx}", "depth.pt")
                # sem_path = os.path.join(self.results_base_dir, scene, f"image_{idx}", "semantic.pt")
                metadata_path = os.path.join(self.results_base_dir, scene, f"image_{idx}", "metadata.json")
                
                valid = True
                if not os.path.exists(image_path):
                    print(f"Warning: Image file not found at: {image_path}. Skipping sample.")
                    valid = False
                # if not os.path.exists(depth_path):
                #     print(f"Warning: Depth file not found at: {depth_path}. Skipping sample.")
                #     valid = False
                # if not os.path.exists(sem_path):
                #     print(f"Warning: Semantic candidate file not found at: {sem_path}. Skipping sample.")
                #     valid = False
                if not os.path.exists(metadata_path):
                    print(f"Warning: Metadata file not found at: {metadata_path}. Skipping sample.")
                    valid = False

                # Validate that the image has the correct dimensions.
                if valid and self.enforce_fixed_resolution:
                    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    if img is None:
                        print(f"Skipping sample {scene} index {idx}: Unable to read image.")
                        valid = False
                    else:
                        h, w, c = img.shape
                        if c != 3:
                            print(f"Skipping sample {scene} index {idx}: Expected 3 channels but got {c}.")
                            valid = False
                        else:
                            if self.target_resolution is None:
                                # Set the target resolution from the first valid image.
                                self.target_resolution = (h, w)
                            else:
                                expected_h, expected_w = self.target_resolution
                                if (h, w) != (expected_h, expected_w):
                                    print(f"Skipping sample {scene} index {idx}: Image resolution mismatch: got {(h, w)}, expected {(expected_h, expected_w)}.")
                                    valid = False

                # Validate the depth candidate tensor.
                # if valid:
                #     try:
                #         depth_data = torch.load(depth_path, weights_only=True, map_location='cpu')
                #         # Assuming 'weights_only' corresponds to extracting 'weights' if it's a dict
                #         if isinstance(depth_data, dict) and 'weights' in depth_data:
                #             depth_vec = depth_data['weights']
                #         else:
                #             depth_vec = depth_data
                #         depth_vec = depth_vec.float()
                #         # If the tensor is 2D, check directly.
                #         if depth_vec.dim() == 2:
                #             if depth_vec.shape != self.desired_candidate_dim:
                #                 print(f"Skipping sample {scene} index {idx}: Depth candidate shape mismatch: got {depth_vec.shape}, expected {self.desired_candidate_dim}")
                #                 valid = False
                #         # If the tensor is 3D (e.g., if an extra channel dimension is present)
                #         elif depth_vec.dim() == 3:
                #             # Here we assume the first dimension should match the expected first dimension.
                #             # The remaining two dimensions should multiply to the expected total if not exactly equal.
                #             if depth_vec.shape[0] != self.desired_candidate_dim[0]:
                #                 print(f"Skipping sample {scene} index {idx}: Depth candidate channel mismatch: got {depth_vec.shape[0]}, expected {self.desired_candidate_dim[0]}")
                #                 valid = False
                #             # elif depth_vec.shape[1] * depth_vec.shape[2] != self.desired_candidate_dim[0] * self.desired_candidate_dim[1]:
                #             #     print(f"Skipping sample {scene} index {idx}: Depth candidate spatial dimension mismatch: got {(depth_vec.shape[1], depth_vec.shape[2])}, expected {self.desired_candidate_dim[1]} columns in total.")
                #             #     valid = False
                #         else:
                #             print(f"Skipping sample {scene} index {idx}: Depth candidate dimension {depth_vec.dim()} not supported.")
                #             valid = False
                #     except Exception as e:
                #         print(f"Skipping sample {scene} index {idx} due to depth load error: {e}")
                #         valid = False
                                        
                # # Validate the semantic candidate tensor.
                # if valid:
                #     try:
                #         sem_data = torch.load(sem_path, weights_only=True, map_location='cpu')
                #         if isinstance(sem_data, dict) and 'weights' in sem_data:
                #             sem_vec = sem_data['weights']
                #         else:
                #             sem_vec = sem_data
                #         sem_vec = sem_vec.float()
                #         if sem_vec.dim() == 2:
                #             if sem_vec.shape != self.desired_candidate_dim:
                #                 print(f"Skipping sample {scene} index {idx}: Semantic candidate shape mismatch: got {sem_vec.shape}, expected {self.desired_candidate_dim}")
                #                 valid = False
                #         elif sem_vec.dim() == 3:
                #             if sem_vec.shape[0] != self.desired_candidate_dim[0]:
                #                 print(f"Skipping sample {scene} index {idx}: Semantic candidate channel mismatch: got {sem_vec.shape[0]}, expected {self.desired_candidate_dim[0]}")
                #                 valid = False
                #             # elif sem_vec.shape[1] * sem_vec.shape[2] != self.desired_candidate_dim[0] * self.desired_candidate_dim[1]:
                #             #     print(f"Skipping sample {scene} index {idx}: Semantic candidate spatial dimension mismatch: got {(sem_vec.shape[1], sem_vec.shape[2])}, expected total {self.desired_candidate_dim[1]} columns.")
                #             #     valid = False
                #         else:
                #             print(f"Skipping sample {scene} index {idx}: Semantic candidate dimension {sem_vec.dim()} not supported.")
                #             valid = False
                #     except Exception as e:
                #         print(f"Skipping sample {scene} index {idx} due to semantic load error: {e}")
                #         valid = False

                if valid:
                    self.samples.append((scene, idx))
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        scene, img_idx = self.samples[index]
        
        # 1. Load the RGB image.
        image_path = os.path.join(self.image_base_dir, scene, "rgb", f"{img_idx}.png")
        ref_img = cv2.imread(image_path, cv2.IMREAD_COLOR)    
        if ref_img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)    
        ref_img = ref_img / 255.0
        ref_img = np.transpose(ref_img, (2, 0, 1)).astype(np.float32)
        
        # # 2. Load the depth candidate tensor.
        # depth_path = os.path.join(self.results_base_dir, scene, f"image_{img_idx}", "depth.pt")
        # try:
        #     depth_data = torch.load(depth_path, weights_only=True, map_location='cpu')
        #     if isinstance(depth_data, dict) and 'weights' in depth_data:
        #         depth_vec = depth_data['weights']
        #     else:
        #         depth_vec = depth_data
        #     depth_vec = depth_vec.float()
        # except Exception as e:
        #     raise RuntimeError(f"Failed to load depth tensor from {depth_path}: {e}")
            
        # # 3. Load the semantic candidate tensor.
        # sem_path = os.path.join(self.results_base_dir, scene, f"image_{img_idx}", "semantic.pt")
        # try:
        #     sem_data = torch.load(sem_path, weights_only=True, map_location='cpu')
        #     if isinstance(sem_data, dict) and 'weights' in sem_data:
        #         sem_vec = sem_data['weights']
        #     else:
        #         sem_vec = sem_data
        #     sem_vec = sem_vec.float()
        # except Exception as e:
        #     raise RuntimeError(f"Failed to load semantic tensor from {sem_path}: {e}")
            
        # 4. Load metadata and extract candidate K positions.
        metadata_path = os.path.join(self.results_base_dir, scene, f"image_{img_idx}", "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load metadata from {metadata_path}: {e}")
        
        k_positions = []
        k_scores = []
        # Sort candidate keys based on numerical order after 'K'
        candidate_keys = sorted([k for k in metadata.keys() if k.startswith("K")], key=lambda x: int(x[1:]))
        for key in candidate_keys:
            candidate = metadata[key]
            pos = np.array([candidate["x"], candidate["y"], candidate["o"]], dtype=np.float32)
            score = candidate["score"]
            k_scores.append(score)
            k_positions.append(pos)
        
        gt_location = self.gt_poses[scene][img_idx]
        
        # Determine the best candidate index (1-indexed) based on Euclidean distance.
        min_dist = np.inf
        best_index = -1
        for i, candidate in enumerate(k_positions):
            candidate_xy = candidate[:2]
            dist = np.linalg.norm(candidate_xy - gt_location[:2])
            if dist < min_dist:
                min_dist = dist
                best_index = i + 1
        
        # 5. Retrieve the semantic map for the scene
        semantic_map = self.semantic_maps[scene]  # tensor of shape (1, 300, 300)
        
        sample = {
            "ref_img": ref_img,             # numpy array (C x H x W)
            # "depth_vec": depth_vec,         # candidate depth tensor (expected shape: (5, 40))
            # "sem_vec": sem_vec,             # candidate semantic tensor (expected shape: (5, 40))
            "k_positions": k_positions,     # list of candidate positions (each as [x, y, o])
            "k_scores": k_scores,     # list of candidate scores 
            "gt_location": gt_location,     # numpy array of shape (2,)
            "best_index": best_index,       # integer between 1 and the number of candidates (1-indexed)
            "semantic_map": semantic_map,    # tensor (1 x 300 x 300)
            "metadata_path": metadata_path   
        }
        
        return sample

# -----------------------
# Usage example:
if __name__ == '__main__':
    scenes = ["scene_0", "scene_1"]  # Add your actual scene names here
    # Enforce a fixed image resolution: images must have the size (360, 640) in (height, width).
    # And candidate tensors are now expected to have the shape (5, 40)
    dataset = TopKDataset(
        scene_names=scenes, 
        enforce_fixed_resolution=True, 
        target_resolution=(360, 640), 
        desired_candidate_dim=(5, 40)
    )
    print(f"Dataset has {len(dataset)} valid samples.")
    print(f"Number of skipped scenes due to oversized semantic maps: {dataset.skipped_scenes_due_to_size}")
    
    if len(dataset) > 0:
        try:
            sample = dataset[0]
            print("Sample keys:", sample.keys())
            print("Image shape:", sample["ref_img"].shape)
            # If the candidate tensors are 2D, you can print their shapes directly.
            print("Depth candidate shape:", sample["depth_vec"].shape)
            print("Semantic candidate shape:", sample["sem_vec"].shape)
            print("K positions:", sample["k_positions"])
            print("GT location:", sample["gt_location"])
            print("Best Index:", sample["best_index"])
            print("Semantic Map shape:", sample["semantic_map"].shape)
        except IndexError as e:
            print("Sample skipped:", e)
        except Exception as e:
            print(f"Error accessing sample: {e}")