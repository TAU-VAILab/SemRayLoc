import os
import json
import yaml

# Set paths
scene_dir = '/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/zind/fixed_fp_data_set'
partition_file = '/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/laser/dataset/zind_partition.json'
output_file = '/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/laser/dataset/zind_partition_new.yml'

# Load the original partition file (train, val, test)
with open(partition_file, 'r') as f:
    partition = json.load(f)

# List all scene directories in scene_dir that match the expected pattern.
all_scene_dirs = [
    d for d in os.listdir(scene_dir)
    if os.path.isdir(os.path.join(scene_dir, d)) and d.startswith('scene_') and '_floor_' in d
]

# Build a mapping from scene number (as int) to a list of directory names.
scene_mapping = {}
for d in all_scene_dirs:
    parts = d.split('_')
    # Expected format: scene_<scene_number>_floor_<floor_number>
    if len(parts) >= 4:
        try:
            scene_num_int = int(parts[1])
        except ValueError:
            print(f"Could not convert scene number from directory {d}")
            continue
        scene_mapping.setdefault(scene_num_int, []).append(d)
    else:
        print(f"Warning: Directory name {d} does not match the expected pattern.")

# For each scene, sort the directories by floor number (extracted from the last part)
for scene_num in scene_mapping:
    scene_mapping[scene_num] = sorted(scene_mapping[scene_num],
                                      key=lambda d: int(d.split('_')[-1]))

# Create the new partition using full directory names.
new_partition = {}
missing_scenes = []  # Will print these if a scene from the partition is not found

for split in partition:
    new_partition[split] = []
    for scene_str in partition[split]:
        try:
            scene_int = int(scene_str)
        except ValueError:
            print(f"Skipping invalid scene number {scene_str}")
            continue
        if scene_int in scene_mapping:
            new_partition[split].extend(scene_mapping[scene_int])
        else:
            missing_scenes.append(scene_str)
    # Sort the directories for each split by scene number then floor number.
    new_partition[split] = sorted(new_partition[split],
                                  key=lambda d: (int(d.split('_')[1]), int(d.split('_')[-1])))

# Save the new partition as a YAML file.
with open(output_file, 'w') as f:
    yaml.dump(new_partition, f, default_flow_style=False, sort_keys=False)

print("New partition YAML file saved to:", output_file)
if missing_scenes:
    print("Missing scene numbers (not added to YAML):", sorted(missing_scenes, key=lambda s: int(s)))
