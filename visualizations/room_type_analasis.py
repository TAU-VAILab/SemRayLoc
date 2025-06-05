import os
import json
import glob
import numpy as np

def rectangle_area(rect):
    """Compute the area of a rectangle given its min and max coordinates."""
    return (rect["max_x"] - rect["min_x"]) * (rect["max_y"] - rect["min_y"])

# Base directory containing scene folders (adjust this path as needed)
base_dir = "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/test_data_set_full/structured3d_perspective_full_with_room_types"

# Find all scene directories (e.g., scene_0, scene_16, etc.)
scene_dirs = glob.glob(os.path.join(base_dir, "scene_*"))

# List to store the per-image (or per-view) proportions
per_image_proportions = []

for scene_dir in scene_dirs:
    # Path to JSON file with room polygons
    json_path = os.path.join(scene_dir, "room_types_rectangles.json")
    # Path to per-image label file; each line corresponds to the image’s label (the room type)
    txt_path = os.path.join(scene_dir, "room_type_per_image.txt")
    
    # Skip scene if JSON file does not exist
    if not os.path.exists(json_path):
        continue

    # Load JSON data
    with open(json_path, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error reading {json_path}")
            continue

    # Compute total area of all room polygons and also the area per room type
    scene_total_area = 0.0
    room_areas = {}
    
    for room in data:
        room_type = room.get("room_type")
        for poly in room.get("polygons", []):
            area = rectangle_area(poly)
            room_areas[room_type] = room_areas.get(room_type, 0) + area
            scene_total_area += area

    # If there is no area, skip this scene
    if scene_total_area <= 0:
        continue

    # If the per-image label file exists, process each image’s label
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            # Each non-empty line is assumed to be the label (room type) for that image
            labels = [line.strip() for line in f if line.strip()]
        for label in labels:
            # Get the area corresponding to the label (if not present, use 0)
            label_area = room_areas.get(label, 0)
            # Compute the proportion of the labeled room area out of the total area
            proportion = label_area / scene_total_area
            per_image_proportions.append(proportion)
    else:
        # If no per-image label file exists, you could decide to skip or use an alternative method.
        continue

# Compute the overall average proportion over all images
if per_image_proportions:
    overall_average = np.mean(per_image_proportions)
    print("Overall average proportion (labeled room area / total floor plan area):")
    print(f"{overall_average:.4f}")
else:
    print("No image label data available to compute proportions.")
