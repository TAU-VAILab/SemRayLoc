# import os
# import json
# from collections import Counter

# # Path to the dataset directory
# dataset_path = "/home/yuvalg/projects/Semantic_Floor_plan_localization/data/test_data_set_full"

# labels = []

# # Walk through all directories and files in the dataset path
# for root, dirs, files in os.walk(dataset_path):
#     for file in files:
#         file_path = os.path.join(root, file)
#         # Process JSON files that contain room type information
#         if file.endswith("room_types_rectangles.json"):
#             try:
#                 with open(file_path, "r") as f:
#                     data = json.load(f)
#                 # For each room object, extract the room_type
#                 for item in data:
#                     room_type = item.get("room_type")
#                     if room_type:
#                         labels.append(room_type)
#             except json.JSONDecodeError as e:
#                 print(f"Error decoding JSON in {file_path}: {e}")
#         # Process text files with room types listed line by line
#         elif file.endswith("room_type_per_image_mapped.txt"):
#             with open(file_path, "r") as f:
#                 for line in f:
#                     label = line.strip()
#                     if label:
#                         labels.append(label)

# # Compute the count distribution of labels
# label_counts = Counter(labels)

# # Sort the label counts by count in descending order
# sorted_label_counts = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)

# # Print the sorted results
# print("Count distribution of labels (sorted by count):")
# print("total number of labels:", len(labels))
# print("total count of labels:", sum(label_counts.values()))
# for label, count in sorted_label_counts:
#     print(f"{label}: {count}")


from pathlib import Path
from collections import Counter

# Set the base directory to your test data set folder
base_dir = Path("/home/yuvalg/projects/Semantic_Floor_plan_localization/data/test_data_set_full")

# Get a list of scene directories (directories starting with "scene_")
scenes = [scene for scene in base_dir.iterdir() if scene.is_dir() and scene.name.startswith("scene_")]
num_scenes = len(scenes)

total_rgb_pngs = 0
labels_counter = Counter()
scenes_missing_label_file = []

# Process each scene directory
for scene in scenes:
    # Count PNG files in the rgb subfolder if it exists
    rgb_dir = scene / "rgb"
    if rgb_dir.exists():
        png_files = list(rgb_dir.glob("*.png"))
        total_rgb_pngs += len(png_files)
    
    # Path to the room type file
    label_file = scene / "room_type_per_image.txt"
    if label_file.exists():
        with label_file.open("r") as f:
            # Each line in the file represents one room type label
            for line in f:
                label = line.strip()
                if label:  # Ignore empty lines
                    labels_counter[label] += 1
    else:
        scenes_missing_label_file.append(scene.name)

# Print out summary information
print(f"Number of scenes: {num_scenes}")
print(f"Total number of rgb PNGs: {total_rgb_pngs}")

print("\nRoom Type Labels Summary:")
for label, count in labels_counter.items():
    print(f"{label}: {count}")
print(f"Total number of room type labels: {sum(labels_counter.values())}")

# Print scenes that are missing the room type file
if scenes_missing_label_file:
    print("\nScenes missing 'room_type_per_image.txt':")
    for scene_name in scenes_missing_label_file:
        print(scene_name)
else:
    print("\nAll scenes have the 'room_type_per_image.txt' file.")
