import os
target_filename = "rgb_rawlight.png"

def count_rgb_rawlight_files(dataset_dir):
    count = 0
    for root, dirs, files in os.walk(dataset_dir):
        count += files.count(target_filename)
    return count

if __name__ == "__main__":
    dataset_dir = "/home/yuvalg/projects/Semantic_Floor_plan_localization/data/structured3d/structured3d_panorama"  # Replace with your dataset directory path
    file_count = count_rgb_rawlight_files(dataset_dir)
    print(f"Found {file_count} '{target_filename}' files in the dataset.")


# import os

# # Path to the base directory containing scene folders
# base_path = '/home/yuvalg/projects/Semantic_Floor_plan_localization/data/structured3d/structured3d_panorama'

# # List to store missing scene folder names
# missing_scenes = []

# # Iterate over the range of scene numbers from 0 to 3500 inclusive
# for i in range(3501):
#     scene_name = f"scene_{i:05d}"
#     scene_path = os.path.join(base_path, scene_name)
#     if not os.path.isdir(scene_path):
#         missing_scenes.append(scene_name)

# print("Missing scene folders:")
# for scene in missing_scenes:
#     print(scene)

# print(f"Total missing scenes: {len(missing_scenes)}")
