import os
import shutil

# Define source and destination base directories.
source_base = "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/zind/zind_data_set_80_fov"
destination_base = "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/zind/fixed_fp_data_set"

# Iterate over all items in the source directory.
for item in os.listdir(source_base):
    scene_path = os.path.join(source_base, item)
    
    # Check if the item is a directory and its name starts with "scene_"
    if os.path.isdir(scene_path) and item.startswith("scene_"):
        # Construct the full path to the "rgb" folder in this scene.
        rgb_folder = os.path.join(scene_path, "rgb")
        
        # Proceed only if the "rgb" folder exists.
        if os.path.exists(rgb_folder) and os.path.isdir(rgb_folder):
            # Construct the corresponding destination scene folder.
            destination_scene = os.path.join(destination_base, item)
            os.makedirs(destination_scene, exist_ok=True)
            
            # Define the destination path for the rgb folder.
            destination_rgb_folder = os.path.join(destination_scene, "rgb")
            
            print(f"Moving {rgb_folder} to {destination_rgb_folder}")
            shutil.move(rgb_folder, destination_rgb_folder)
