#!/usr/bin/env python3
import os
import json
import glob
import yaml

def load_partition(partition_json_path):
    """Load the partition JSON file."""
    with open(partition_json_path, "r") as f:
        partition = json.load(f)
    return partition

def collect_scene_floors(scene_id, base_dir):
    """
    Given a scene id from the partition JSON, pad it to 4 digits and look for matching floor folders.
    
    For example, if scene_id is "101", zfill will convert it to "0101", and the script will search for directories
    matching "scene_0101_floor_*" in the base directory.
    
    Returns a sorted list of unique floor folder names.
    """
    # Pad the scene id to 4 digits
    padded_scene_id = scene_id.zfill(4)
    
    # Build the pattern based on the padded scene id.
    pattern = os.path.join(base_dir, f"scene_{padded_scene_id}_floor_*")
    
    matched_paths = glob.glob(pattern)
    
    # Use a set to avoid duplicate folder names in case of overlapping matches.
    floors = set()
    for path in matched_paths:
        if os.path.isdir(path):
            floors.add(os.path.basename(path))
    
    floors = sorted(list(floors))
    return floors

def build_split_dict(partition, base_dir):
    """
    For each split (train, val, test), collect the unique floor folder names.
    Returns a dictionary mapping each split to a list of folder names.
    """
    split = {}
    for phase in ['train', 'val', 'test']:
        scene_ids = partition.get(phase, [])
        floors_list = []
        for scene_id in scene_ids:
            scene_floors = collect_scene_floors(scene_id, base_dir)
            if scene_floors:
                floors_list.extend(scene_floors)
            else:
                print(f"Warning: No floor folders found for scene id {scene_id} (padded: {scene_id.zfill(4)})")
        # Remove duplicates overall and sort
        unique_floors = sorted(set(floors_list))
        split[phase] = unique_floors
    return split

def write_yaml(split_dict, output_yaml_path):
    """Write the split dictionary to a YAML file."""
    with open(output_yaml_path, "w") as f:
        # Use block style lists (default_flow_style=False) and preserve the key order.
        yaml.dump(split_dict, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote YAML file to {output_yaml_path}")

def main():
    # Path to partition.json. Adjust if needed.
    partition_json_path = (
        "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/zind/zind_raw/partition.json"
    )
    
    # Base directory that holds the floor directories.
    base_dir = (
        "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/zind/zind_data_set_80_fov"
    )
    
    # Output YAML file path (as requested).
    output_yaml_path = (
        "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data_utils_zind/split.yaml"
    )
    
    # Load the partition file.
    partition = load_partition(partition_json_path)
    
    # Build the split dictionary by crawling the base directory.
    split_dict = build_split_dict(partition, base_dir)
    
    # Write the YAML file.
    write_yaml(split_dict, output_yaml_path)

if __name__ == "__main__":
    main()
