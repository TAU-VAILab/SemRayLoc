"""Generate grid raycasts for floor plan maps using multi-threading."""

import os
import numpy as np
import tqdm
import yaml
from utils.raycast_utils import ray_cast
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
from pathlib import Path

# Get the directory of this file
FILE_DIR = Path(__file__).parent.parent

# Raycast parameters
RAYCAST_PARAMS = {
    'orn_slice': 36,          # Number of orientation slices
    'max_dist': 1500,         # Maximum raycast distance in mm
    'min_dist': 5,            # Minimum raycast distance in mm
    'original_resolution': 0.01,  # Input map resolution in m/pixel
    'output_resolution': 0.1,     # Output map resolution in m/pixel
    'dist_max': 15 * 100,     # Maximum distance for semantic raycast (15m in mm)
}

# Processing parameters
PROCESSING_PARAMS = {
    'num_processes': 15,      # Number of parallel processes
    'scene_padding': 5        # Number of digits for scene number padding
}
def raycast_semantic(occ_semantic, **kwargs):
    """Get semantic colors and distances through raycast."""
    params = {**RAYCAST_PARAMS, **kwargs}
    ratio = params['output_resolution'] / params['original_resolution']
    semantic_df = np.zeros(list((np.array(occ_semantic.shape[:2]) // ratio).astype(int)) + [params['orn_slice']])
    depth_df = np.zeros(list((np.array(occ_semantic.shape[:2]) // ratio).astype(int)) + [params['orn_slice']])

    for o in tqdm.tqdm(range(params['orn_slice'])):
        theta = o / params['orn_slice'] * np.pi * 2
        for col in range(semantic_df.shape[0]):
            for row in range(semantic_df.shape[1]):
                pos = np.array([row, col]) * ratio
                depth_val_m, prediction_class, _ = ray_cast(
                    occ_semantic, pos, theta, 
                    dist_max=params['dist_max'],
                    min_dist=params['min_dist'], 
                )
                semantic_df[col, row, o] = prediction_class
                depth_df[col, row, o] = depth_val_m / 100  # Convert mm to meters
                
    return semantic_df, depth_df

def load_yaml(filepath):
    """Load YAML configuration file."""
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)

def process_scene(scene, base_dir, df_dir):
    """Process a single scene to generate raycast maps."""
    try:
        # Handle scene path
        if 'floor' in scene:  # ZInD dataset
            scene_path = scene
        else:  # S3D dataset
            scene_number = int(scene.split('_')[1])
            scene_path = f"scene_{scene_number}"
        
        # Load semantic map
        semantic_map_path = os.path.join(base_dir, scene_path, 'floorplan_semantic.png')
        print(f"Processing scene: {scene_path}")
        occ_semantic = plt.imread(semantic_map_path)
        
        # Generate raycast maps
        depth_df = {}
        semantic_df = {}
        semantic_df["semantic"], depth_df["depth"] = raycast_semantic(occ_semantic)
        
        # Save results
        scene_dir = os.path.join(df_dir, scene_path)
        os.makedirs(scene_dir, exist_ok=True)
        
        np.save(os.path.join(scene_dir, "depth_df.npy"), depth_df["depth"])
        np.save(os.path.join(scene_dir, "semantic_df.npy"), semantic_df["semantic"])
        
    except Exception as e:
        print(f"Failed processing scene {scene}: {str(e)}")

def process_test_scenes(yaml_path, base_dir, df_dir):
    """Process all test scenes in parallel."""
    # Load scenes from YAML
    split_data = load_yaml(yaml_path)
    scenes = split_data.get('test', [])
    scenes.sort(key=lambda x: int(x.split('_')[-1]))
    
    total_scenes = len(scenes)
    print(f"Total scenes to process: {total_scenes}\n")

    # Process scenes in parallel
    num_processes = min(PROCESSING_PARAMS['num_processes'], os.cpu_count() or 1)
    process_scene_partial = partial(process_scene, base_dir=base_dir, df_dir=df_dir)

    with Pool(processes=num_processes) as pool:
        list(tqdm.tqdm(pool.imap(process_scene_partial, scenes), total=total_scenes))

def main():
    """Main function to process scenes."""
    # Dataset paths relative to project root
    # S3D
    # yaml_path = FILE_DIR / "Data/S3D/processed/split.yaml"
    # base_dir = FILE_DIR / "Data/S3D/processed"
    # df_dir = FILE_DIR / "Data/S3D/df"
    #ZInD
    yaml_path = FILE_DIR / "Data/zind/processed/split.yaml"
    base_dir = FILE_DIR / "Data/zind/processed"
    df_dir = FILE_DIR / "Data/zind/df"
    
    # Ensure directories exist
    os.makedirs(df_dir, exist_ok=True)
    
    # Process scenes
    process_test_scenes(str(yaml_path), str(base_dir), str(df_dir))

if __name__ == "__main__":
    main()
