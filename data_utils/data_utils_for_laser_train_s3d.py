import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import json
import torch
from modules.semantic.semantic_mapper import room_type_to_id, zind_room_type_to_id
from utils.raycast_utils import ray_cast
import matplotlib.pyplot as plt
from laser.dataset.zind_utils import *
from modules.semantic.semantic_mapper import ObjectType, object_to_color
from matplotlib.patches import Circle


def plot_camera_positions_and_rays(img, hits, semantic, ref_pose, output_path,image,file_name, pano_rot, yaw):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
            
    # Figure 1: Camera positions and rays
    fig1, ax1 = plt.subplots(figsize=(img.shape[1]/100, img.shape[0]/100), dpi=100)
    ax1.imshow(img)
    x,y = ref_pose[:2]
   
    ax1.plot(x, y, 'bo', markersize=5)
   
    for ray in zip(hits,semantic):
        (end_x,end_y),semantic_pred = ray
        object_type = ObjectType(semantic_pred)
        color = object_to_color.get(object_type, 'black')
        ax1.plot([x, end_x], [y, end_y], color=color, lw=0.5)
    
    # plot green ray from x,y in the angle of pano_rot at distacne of 1
    angle = np.deg2rad(pano_rot)
    end_x = x + np.cos(angle) * 200
    end_y = y - np.sin(angle) * 200
    ax1.plot([x, end_x], [y, end_y], color='green', lw=1.5)

    # plot red ray from x,y in the angle of yaw at distacne of 1
    angle = np.deg2rad(yaw)
    end_x = x + np.cos(angle) * 200
    end_y = y - np.sin(angle) * 200
    ax1.plot([x, end_x], [y, end_y], color='red', lw=1.5)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax1.axis('equal')
    ax1.axis('off')
    out1 = os.path.join(output_path, f'{file_name}_camera_positions_with_rays.png')
    fig1.savefig(out1, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig1)  

      
    
    #save the image to the dir as well
    out2 = os.path.join(output_path, f'{file_name}_input_image.png')
    plt.imsave(out2, image)    
    
    

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
        room_data_dir='',
        is_train = True, 
        pano_dir ="",
        find_interesting_fov=False,
        random_yaw=True,
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
        self.is_train = is_train
        self.pano_dir=pano_dir
        self.find_interesting_fov = find_interesting_fov
        # Photometric augmentation flags
        self.augment = augment
        self.noise_std = noise_std
        self.color_jitter = T.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.05
        )
        self.random_yaw = random_yaw
        self.scene_start_idx = []
        self.gt_values = []   # depth, semantic, pitch, roll
        self.gt_pose = []     # camera poses
        self.metadata = []     # camera poses

        # NEW: Store room labels per image and polygons per room type
        self.gt_room_label = []   # for each scene, list of N labels (one per image)
        self.room_polygons = []   # for each scene, dict { "bedroom": [bbox1, bbox2,...], ... }
        self.gt_original_path = []

        # Load info for all scenes
        self.load_scene_start_idx_and_values_and_poses()

        # Calculate total length from all scenes
        self.total_len = 0
        for poses in self.gt_pose:
            self.total_len += len(poses)
        print("*********************************************************")
        print(f"Total length: {self.total_len}")
        print("number of scenes: ", len(self.scene_names))  
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
        is_zind = False
        for scene_idx, scene in enumerate(self.scene_names):
            # If "scene_00000" etc. => get the integer for S3D
            if 'floor' in scene:
                is_zind = True
                pass
            else:
                # Structured3D
                scene_number = int(scene.split('_')[1])  # e.g. "scene_0" => 0
                scene = f"scene_{scene_number}"

            # Paths to data for this scene
            scene_folder = os.path.join(self.data_dir, scene)
            # depth_file = os.path.join(scene_folder, "depth.txt")
            # semantic_file = os.path.join(scene_folder, "semantic.txt")
            # pitch_file = os.path.join(scene_folder, "pitch.txt")
            # roll_file = os.path.join(scene_folder, "roll.txt")
            pose_file = os.path.join(scene_folder, "poses.txt")
            # metadata_file = os.path.join(scene_folder, "metadata.json")

            # Also paths to room label + polygons
            # (in your "room_data_dir" instead of "data_dir")
            room_label_file = os.path.join(self.room_data_dir, scene, "room_type_per_image.txt")
            room_rectangles_file = os.path.join(self.room_data_dir, scene, "room_types_rectangles.json")

            try:
                # # 1) Depth / semantic
                # with open(depth_file, "r") as f:
                #     depth_txt = [line.strip() for line in f.readlines()]
                # with open(semantic_file, "r") as f:
                #     semantic_txt = [line.strip() for line in f.readlines()]

                # 2) Pose
                with open(pose_file, "r") as f:
                    poses_txt = [line.strip() for line in f.readlines()]

                # # 3) Pitch / Roll
                # with open(pitch_file, "r") as f:
                #     pitch_txt = [float(line.strip()) for line in f.readlines()]
                # with open(roll_file, "r") as f:
                #     roll_txt = [float(line.strip()) for line in f.readlines()]

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
                    
                # with open(metadata_file, "r") as f:
                #     metadata = json.load(f)
                # original_path = metadata["original_path"]

            except Exception as e:
                print(f"Error reading data for {scene}: {e}. Skipping.")
                continue

            traj_len = len(room_label_lines)
            # Optional check that everything matches
            # if not (len(depth_txt) == len(semantic_txt) == traj_len == len(room_label_lines)):
            #     print(f"Inconsistent data lengths in scene {scene}. Skipping.")
            #     continue

            # Build arrays for depth, semantic, pitch, roll, poses
            scene_depths = []
            scene_semantics = []
            scene_poses = []
            scene_pitch = []
            scene_roll = []

            for state_id in range(traj_len):
            #     # Depth
            #     depth_vals = depth_txt[state_id].split(" ")
            #     depth_arr = np.array([float(d) for d in depth_vals], dtype=np.float32)
            #     scene_depths.append(depth_arr)

            #     # Semantic
            #     semantic_vals = semantic_txt[state_id].split(" ")
            #     semantic_arr = np.array([float(s) for s in semantic_vals], dtype=np.float32)
            #     scene_semantics.append(semantic_arr)

            # Pose
                pose_vals = poses_txt[state_id].split(" ")
                pose_arr = np.array([float(s) for s in pose_vals], dtype=np.float32)
                scene_poses.append(pose_arr)

            #     # Pitch / Roll
            #     scene_pitch.append(pitch_txt[state_id])
            #     scene_roll.append(roll_txt[state_id])

            # Store them
            if is_zind:
                valid_scene_names.append(scene)
            else:
                valid_scene_names.append(f"scene_{scene_number}")
            start_idx += traj_len
            self.scene_start_idx.append(start_idx)

            # self.gt_values.append({
            #     "depth": scene_depths,
            #     "semantic": scene_semantics,
            #     "pitch": scene_pitch,
            #     "roll": scene_roll
            # })
            self.gt_pose.append(scene_poses)

            # # Also store the room labels and polygons
            self.gt_room_label.append(room_label_lines)  # list of strings, length=traj_len
            self.room_polygons.append(polygons_dict)     # dict {rtype: [bbox1, bbox2,...], ... }
            # self.gt_original_path.append(original_path)  # Save original_path from metadata
        
        
        #self N should be the number of images in the dataset i.e number of total posese for example     
        #sum of all lens of gt_pose
        self.fetch_another_count = 0
        self.total_len = sum([len(poses) for poses in self.gt_pose])
        self.scene_names = valid_scene_names

    def fetch_another(self):
        self.fetch_another_count += 1
        print(f"fetch another {self.fetch_another_count}")
        return self.__getitem__(np.random.randint(self.total_len))
    
    
        
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
        # ref_depth = self.gt_values[scene_idx]["depth"][idx_within_scene]
        # ref_semantics = self.gt_values[scene_idx]["semantic"][idx_within_scene]
        ref_pose = self.gt_pose[scene_idx][idx_within_scene]            
        x, y, th = ref_pose
        x = x / 100
        y = y / 100

        pano_image_path = os.path.join(
            self.dataset_dir,
            scene_name,
            "rgb",
            f"{idx_within_scene}.png"
        )
        pano_image = cv2.imread(pano_image_path, cv2.IMREAD_COLOR)    

        if pano_image is None:
            return self.fetch_another()            
        pano_rot = np.rad2deg(th) - 90
        
        # rnd_rot = np.random.rand() * 360
        # pano_image = rot_pano(pano_image, rnd_rot)
        # pano_rot += rnd_rot
        yaw = 0            
        if self.random_yaw:
            yaw = np.random.rand() * 360
        else:
            yaw = 0           
        
        query_image = pano2persp(
                        pano_image, 80, yaw, 0, 0, (512,512)
                    )    

        pano_rot = pano_rot % 360
        pano_rot -= yaw
        
        ref_img = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB) / 255.0
        # Convert HWC -> CHW, float32
        ref_img = np.transpose(ref_img, (2, 0, 1)).astype(np.float32)
        
        
        #extract depth
        scene_dir = os.path.join(self.dataset_dir, scene_name)
        sem_png = os.path.join(scene_dir, "floorplan_semantic.png")
        img_semantic = plt.imread(sem_png)        
        ray_n = 40
        F_W = 1 / np.tan(0.698132) / 2  # ~ 1/(2*tan(40 deg))    
        
        # print(f"th: {np.rad2deg(th)}")
        center_angs = np.flip(np.arctan2((np.arange(ray_n) - np.arange(ray_n).mean()),ray_n * F_W))
        angs = center_angs + np.deg2rad(pano_rot)
        ref_pose[2] = np.deg2rad(pano_rot)
        depth_ref =[]
        semantic_ref = []
        hit_coords = []
        for _, ang in enumerate(angs):
            dist, pred_class, hit_coord, _ = ray_cast(img_semantic, np.array([x*100, y*100]),
                                                ang, dist_max=15*100, min_dist=5,  cast_type = 2)
            distance_adjusted = dist/100 
            depth_ref.append(distance_adjusted)
            semantic_ref.append(pred_class)                             
            hit_coords.append(hit_coord)                             
    

        # plot_camera_positions_and_rays(img_semantic,hit_coords,semantic_ref,np.array([x*100, y*100]),"/home/yuvalg/projects/Semantic_Floor_plan_localization/temp", query_image,file_name=f"{scene_name}_{idx_within_scene}_yaw_{yaw}", pano_rot= pano_rot, yaw = center_ang_yaw)        
        
        data_dict = {
            "scene_name": scene_name,
            "idx_within_scene": idx_within_scene,
            "ref_depth": torch.tensor(depth_ref, dtype=torch.torch.float32),
            "ref_semantics": torch.tensor(semantic_ref, dtype=torch.int),
            "ref_pose": ref_pose,
            "ref_noise": 0,
            "ref_pitch": 0,
            "ref_roll": 0,
        }
        
        data_dict["ref_img"] = ref_img
        data_dict["original_pano"] = query_image
        mask = np.ones(list(ref_img.shape[:2]), dtype=np.uint8)
        data_dict["ref_mask"] = mask
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
        # data_dict["room_polygons"] = polygons_dict

        return data_dict

   