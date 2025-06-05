import os
import cv2
import numpy as np
import tqdm
import yaml
from raycast_utils import ray_cast, get_color_name
from modules.semantic.semantic_mapper import ObjectType, object_to_color
import matplotlib.pyplot as plt


def raycast_depth(
    occ_walls, orn_slice=36, max_dist=1500, original_resolution=0.01, output_resolution=0.1
):
    """
    Get desdf from walls-only occupancy grid and color from semantic grid through brute force raycast.
    Input:
        occ_walls: the walls-only map as occupancy.
        orn_slice: number of equiangular orientations.
        max_dist: maximum raycast distance, [m].
        original_resolution: the resolution of occ input [m/pixel].
        resolution: output resolution of the desdf [m/pixel].
    Output:
        desdf: the directional esdf of the occ input in meters.
    """
    ratio = output_resolution / original_resolution
    desdf = np.zeros(list((np.array(occ_walls.shape[:2]) // ratio).astype(int)) + [orn_slice])

    # Perform raycasting for each orientation slice
    for o in tqdm.tqdm(range(orn_slice)):
        theta = o / orn_slice * np.pi * 2
        for y in range(desdf.shape[0]):
            for x in range(desdf.shape[1]):
                pos = np.array([x, y]) * ratio
                dist, _, _ , _ = ray_cast(occ_walls, pos, theta)
                desdf[y, x, o] = dist / 100  # ray_cast returns in mm/10 --> Store the distance in M    
    
    return desdf

def raycast_semantic(
    occ_semantic, orn_slice=36, max_dist=1500, original_resolution=0.01, output_resolution=0.1
):
    """
    Get desdf from walls-only occupancy grid and color from semantic grid through brute force raycast.
    Input:
        occ_semantic: the semantic map for colors.
        orn_slice: number of equiangular orientations.
        max_dist: maximum raycast distance, [m].
        original_resolution: the resolution of occ input [m/pixel].
        resolution: output resolution of the desdf [m/pixel].
    Output:
        desdf: the directional esdf of the occ input in meters.
    """
    ratio = output_resolution / original_resolution
    colors = np.zeros(list((np.array(occ_semantic.shape[:2]) // ratio).astype(int)) + [orn_slice])

    # Perform raycasting for each orientation slice
    for o in tqdm.tqdm(range(orn_slice)):
        theta = o / orn_slice * np.pi * 2
        for col in range(colors.shape[0]):
            for row in range(colors.shape[1]):
                pos = np.array([row, col]) * ratio
                _, color_val , _ , _ = ray_cast(occ_semantic, pos, theta)
                colors[col, row, o] = color_val                                      
    return colors

def load_yaml(filepath):
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)

def process_test_scenes(yaml_path, base_dir, desdf_dir):
    # Load the YAML file
    split_data = load_yaml(yaml_path)
    # scenes = split_data.get('val',  []) # For training of the mapper
    scenes = split_data.get('test', [])
    scenes = ['scene_03261','scene_03279','scene_03280','scene_03452']
    scenes.sort(key=lambda x: int(x.split('_')[-1]))
    # scenes = scenes[1596:1650]#1
    # scenes = scenes[1650:1700]#1
    # scenes = scenes[1700:1750]#1
    # scenes = scenes[1750:1800]#1
    #FOR TRAIN
    # scenes = scenes[0:200]#1
    # scenes = scenes[200:400]#2
    # scenes = scenes[400:600]#3
    # scenes = scenes[600:800]#4
    # scenes = scenes[800:1000]#5
    # scenes = scenes[1000:1200]#6
    # scenes = scenes[1200:1400]#7
    # scenes = scenes[1400:1600]#8
    # scenes = scenes[1600:1800]#9
    # scenes = scenes[1800:2000]#10
    #FOR TESTS
    # scenes = scenes[0:40]    #1
    # scenes = scenes[40:80]   #2
    # scenes = scenes[80:120]  #3
    # scenes = scenes[120:160] #4
    # scenes = scenes[160:200] #5
    # scenes = scenes[200:240] #6
    # scenes = scenes[240:280] #7
    # scenes = scenes[280:300] #8
    total_scenes = len(scenes)
    print(f"Total scenes to process: {total_scenes}\n")

    # Loop through each test scene and process
    for scene in tqdm.tqdm(scenes):
        if 'floor' in scene: #zind
            pass
        else:
            scene_number = int(scene.split('_')[1])
            scene = f"scene_{scene_number}"
            
        semantic_map_path = os.path.join(base_dir, scene, 'floorplan_semantic.png')
        walls_only_map_path = os.path.join(base_dir, scene, 'floorplan_walls_only.png')
        print(f"Processing scene: {scene}")
        
        # Load the maps
        occ_walls = plt.imread(walls_only_map_path)
        occ_semantic = plt.imread(semantic_map_path)
        
        
        desdf = {}
        color = {}
        
        # Compute DESDF and color
        desdf["desdf"] = raycast_depth(occ_walls)
        color["desdf"] = raycast_semantic(occ_semantic)
        
        # Save the results
        scene_dir = os.path.join(desdf_dir, scene)
        if not os.path.exists(scene_dir):
            os.mkdir(scene_dir)
        
        np.save(os.path.join(scene_dir, "desdf.npy"), desdf)
        np.save(os.path.join(scene_dir, "color.npy"), color)

if __name__ == "__main__":
    #S3D
    yaml_path = '/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/test_data_set_full/structured3d_perspective_full/split.yaml'
    base_dir = '/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/test_data_set_full/structured3d_perspective_full'
    desdf_dir = '/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/test_data_set_full/desdf'
    #------
    # yaml_path = '/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/test_data_set/structured3d_perspective_empty/split.yaml'
    # base_dir = '/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/test_data_set/structured3d_perspective_empty'
    # desdf_dir = '/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/test_data_set/desdf'
    
    #Zind
    # yaml_path = '/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/zind/zind_data_set/split.yaml'
    # base_dir = '/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/zind/zind_data_set'
    # desdf_dir = '/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/zind/desdf_10'
    

    process_test_scenes(yaml_path, base_dir, desdf_dir)
#