import os
from create_casting_files import process_scene as process_casting_scene
from tqdm import tqdm
import logging

# Set the root logger to warning
def process_all_scenes(base_path, output_base_path, resolution=0.01, dpi=100, fov_segments=40, depth = 15):
    logging.basicConfig(level=logging.WARNING)
    # Override logging levels for specific libraries:
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    
    scenes_to_process = [f'{str(i).zfill(4)}' for i in range(0,1600)]
    # scenes_to_process = [f'{str(i).zfill(4)}' for i in range(0,400)]
    # scenes_to_process = [f'{str(i).zfill(4)}' for i in range(400,800)]
    # scenes_to_process = [f'{str(i).zfill(4)}' for i in range(800,1200)]
    # scenes_to_process = [f'{str(i).zfill(4)}' for i in range(1200,1600)]
    scenes = [d for d in scenes_to_process if os.path.isdir(os.path.join(base_path, d))]
    # scenes = ['1466']
    total_scenes = len(scenes)
    print(f"Total scenes to process: {total_scenes}\n")
    
    failed_scenes = []
    
    for idx, scene_name in tqdm(enumerate(scenes, start=1)):
        try: 
            print(f"Processing Scene {scene_name} ({idx}/{total_scenes})...")
            
            process_casting_scene(scene_name, base_path, output_base_path, resolution, dpi, fov_segments, depth)                       
            
            print(f"Finished processing Scene {scene_name}.\n")
        except Exception as e:
            print(f"Failed to process Scene {scene_name}. Error: {e}\n")
            failed_scenes.append(scene_name)
    
    if failed_scenes:
        print("The following scenes failed to process:")
        for scene in failed_scenes:
            print(f"Scene {scene}")


def main():
    base_path = "/home/yuvalg/projects/Semantic_Floor_plan_localization/data/zind_data"
    # output_base_path = "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/zind/zind_data_set"
    output_base_path = "/home/yuvalg/projects/Semantic_Floor_plan_localization/data/zind_perspective"

    resolution = 0.01  # Resolution in meters per pixel
    dpi = 100          # DPI for output images
    fov_segments = 40  # Number of segments to divide the FOV into
    depth = 15
    process_all_scenes(base_path, output_base_path, resolution, dpi, fov_segments, depth)

if __name__ == "__main__":
    main()
