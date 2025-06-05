import os

# Paths
base_dir = '/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/test_data_set/structured3d_perspective_empty'
scenes_dir = base_dir

# Iterate over all scenes
for scene_name in os.listdir(scenes_dir):
    scene_path = os.path.join(scenes_dir, scene_name)
    rgb_path = os.path.join(scene_path, 'rgb')
    poses_file_path = os.path.join(scene_path, 'poses.txt')
    
    # Check if paths exist
    if not os.path.exists(rgb_path) or not os.path.exists(poses_file_path):
        print(f"Skipping {scene_name} as either rgb folder or poses.txt is missing")
        continue

    # Count the number of images in the rgb folder
    num_images = len([f for f in os.listdir(rgb_path) if os.path.isfile(os.path.join(rgb_path, f))])

    # Count the number of rows in the poses.txt file
    with open(poses_file_path, 'r') as poses_file:
        num_rows = len(poses_file.readlines())

    # Compare and print if they do not match
    if num_images != num_rows:
        print(f"Mismatch in {scene_name}: {num_images} images vs {num_rows} rows in poses.txt")

print("Check completed.")
