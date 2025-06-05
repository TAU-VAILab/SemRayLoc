#!/usr/bin/env python3
from modules.semantic.semantic_mapper import zind_room_type_to_id
import os

# Explicit mapping rules for labels that need to be converted exactly.
# Note: "family room" is mapped to "living room" (as the closest match).
explicit_mapping = {
    "family room": "living room",
    "stair landing": "stairs",
    "breakfast nook": "dining room",
    "storage": "closet",
    "pantry": "closet",
    "primary bedroom": "bedroom",
    "primary bathroom": "bathroom"
}

def convert_label(label):
    """
    Convert a label string to one of the standardized room types.
    
    Rules:
      1. If the label exactly matches an entry in explicit_mapping, use that.
      2. If the label exactly exists in zind_room_type_to_id, return it.
      3. If the label contains any of the keywords (kitchen, bathroom,
         hallway, closet, bedroom), return that keyword.
      4. Otherwise, return "undefined".
    """
    label = label.strip().lower()
    
    # 1. Check for explicit mapping rules.
    if label in explicit_mapping:
        return explicit_mapping[label]
    
    # 2. If the label exactly matches one of the target keys, return it.
    if label in zind_room_type_to_id:
        return label
    
    # 3. If the label contains any of the keywords, return the keyword.
    keywords = ["kitchen", "bathroom", "hallway", "closet", "bedroom", "laundry"]
    for keyword in keywords:
        if keyword in label:
            return keyword
    
    # 4. All other cases become "undefined"
    return "undefined"

def process_file(input_file):
    """Reads an input file, converts labels, and writes them to converted_room_types.txt."""
    print(f"Processing file: {input_file}")
    # Read the file
    with open(input_file, "r") as f:
        lines = f.readlines()
    
    converted_labels = []
    for line in lines:
        original_label = line.strip()
        new_label = convert_label(original_label)
        # Optionally, lookup the numeric id (if needed):
        label_id = zind_room_type_to_id.get(new_label, zind_room_type_to_id["undefined"])
        converted_labels.append(new_label)
        print(f"  Original: '{original_label}' -> Converted: '{new_label}', ID: {label_id}")
    
    # Save the converted labels in the same folder
    output_file = os.path.join(os.path.dirname(input_file), "room_type_per_image.txt")
    with open(output_file, "w") as f_out:
        for label in converted_labels:
            f_out.write(label + "\n")
    print(f"Converted labels saved to {output_file}\n")

def main():
    # Base directory containing the dataset.
    base_dir = "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/zind/zind_data_set_80_fov_with_room_types"
    
    # Walk through all subdirectories of base_dir
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            # Process only files named "room_type_per_image.txt"
            if file == "room_type_per_image.txt":
                input_file = os.path.join(root, file)
                process_file(input_file)

if __name__ == "__main__":
    main()
