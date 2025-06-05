import os
import shutil
import yaml

# Load the YAML file
yaml_file = '/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/test_data_set/structured3d_perspective_empty/split.yaml'

with open(yaml_file, 'r') as file:
    data = yaml.safe_load(file)

scenes = data['test']

# Define the base directories
source_base_dir = '/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/test_data_set/structured3d_perspective_empty'
destination_base_dir = '/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/test_data_set/desdf'

# Ensure the destination directory exists
os.makedirs(destination_base_dir, exist_ok=True)

# Iterate over each scene
for scene in scenes:
    source_dir = os.path.join(source_base_dir, scene, 'desdf')
    destination_dir = os.path.join(destination_base_dir, scene)
    
    # Ensure the destination scene directory exists
    os.makedirs(destination_dir, exist_ok=True)
    
    # Move each file from the source to the destination
    for file_name in os.listdir(source_dir):
        source_file = os.path.join(source_dir, file_name)
        destination_file = os.path.join(destination_dir, file_name)
        
        # Move the file
        shutil.move(source_file, destination_file)

    print(f"Moved contents of {source_dir} to {destination_dir}")
