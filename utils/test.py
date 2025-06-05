#!/usr/bin/env python3
import os
from collections import Counter

def aggregate_distribution(base_dir):
    """
    Walks through the base directory recursively and aggregates the frequency
    of each room type found in files named "converted_room_types.txt".
    """
    counter = Counter()
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == "converted_room_types.txt":
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    # Each line is assumed to contain one room type.
                    lines = f.read().splitlines()
                    counter.update(lines)
    return counter

def main():
    base_dir = "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/zind/zind_data_set_80_fov_with_room_types"
    distribution = aggregate_distribution(base_dir)
    print("Distribution of camera room types:")
    for room_type, count in distribution.most_common():
        print(f"{room_type}: {count}")

if __name__ == "__main__":
    main()
