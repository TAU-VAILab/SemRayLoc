# utils/data_loader_helper.py

import os
import cv2
import numpy as np
import tqdm

def load_scene_data(test_set, dataset_dir, df_path,use_walls=True):
    depth_df = {}
    semantic_df = {}
    maps = {}
    walls = {}
    gt_poses = {}
    valid_scene_names = []  # To keep track of valid scenes

    for scene in tqdm.tqdm(test_set.scene_names):
        print("scene:", scene)
        try:
            if 'floor' in scene: # zind
                pass
            else:
                scene_number = int(scene.split('_')[1])
                scene = f"scene_{scene_number}"
            
            print(f"Loading depth_df of: {scene}")
            depth_df_npy = np.load(os.path.join(df_path, scene, "depth_df.npy"), allow_pickle=True) 


            semantic_df_npy = np.load(os.path.join(df_path, scene, "semantic_df.npy"), allow_pickle=True)
            occ_sem = cv2.imread(os.path.join(dataset_dir, scene, "floorplan_semantic.png"))
            occ_sem_rgb = cv2.cvtColor(occ_sem, cv2.COLOR_BGR2RGB)      
            occ_walls = None      

            depth_df[scene] = depth_df_npy
            semantic_df[scene] = semantic_df_npy

            maps[scene] = occ_sem_rgb
            walls[scene] = occ_walls


            with open(os.path.join(dataset_dir, scene, "poses.txt"), "r") as f:
                poses_txt = [line.strip() for line in f.readlines()]
                traj_len = len(poses_txt)
                poses = np.zeros([traj_len, 3], dtype=np.float32)
                for state_id in range(traj_len):
                    pose = poses_txt[state_id].split(" ")
                    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
                    poses[state_id, :] = np.array((x, y, th), dtype=np.float32)
                gt_poses[scene] = poses
            
            valid_scene_names.append(scene)
        except Exception as e:
            print(f"Error in loading  df of: {scene}")
            print(e)
            continue

    return depth_df, semantic_df, maps, gt_poses, valid_scene_names, walls
