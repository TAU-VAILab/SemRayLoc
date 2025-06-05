import os

# Define the directory path and the file prefixes to keep
dir_path = r"/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/structured3d_perspective_full"
prefixes_to_keep = ('annotation_3d', 'camera_xyz', 'rgb_rawlight','camera_pose')

# Traverse the directory tree
for root, dirs, files in os.walk(dir_path):
    for file in files:
        # Check if the file does not start with one of the prefixes to keep
        if not file.startswith(prefixes_to_keep):
            file_path = os.path.join(root, file)
            os.remove(file_path)
            print(f"Removed: {file_path}")

print("Cleanup complete.")
