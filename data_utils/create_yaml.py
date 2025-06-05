# Define the original scene ranges
original_total_scenes = 3500  # 0 to 3499

# Define the split points based on the new fixed ranges
train_split_end = 3000
val_split_end = 3250
test_split_end = 3500

# Generate the scenes based on the split points
train_scenes = [f"scene_{i:05d}" for i in range(0, train_split_end)]
val_scenes = [f"scene_{i:05d}" for i in range(train_split_end, val_split_end)]
test_scenes = [f"scene_{i:05d}" for i in range(val_split_end, test_split_end)]

# Create the dictionary structure for the YAML
data = {
    'train': train_scenes,
    'val': val_scenes,
    'test': test_scenes
}

# Function to manually format the YAML output
def format_yaml_list(lst):
    formatted_list = "[\n"
    formatted_list += ",\n".join(f'  "{item}"' for item in lst)
    formatted_list += ",\n]"
    return formatted_list

# Manually format the YAML content
yaml_content = "train:\n" + format_yaml_list(data['train']) + "\n\n"
yaml_content += "val:\n" + format_yaml_list(data['val']) + "\n\n"
yaml_content += "test:\n" + format_yaml_list(data['test']) + "\n"

# Write to the YAML file
with open('/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/test_data_set_full/structured3d_perspective_full/split.yaml', 'w') as file:
    file.write(yaml_content)

print("YAML file 'split.yaml' created successfully.")
