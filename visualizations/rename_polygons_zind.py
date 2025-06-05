#!/usr/bin/env python3
import os
import json
#!/usr/bin/env python3
from modules.semantic.semantic_mapper import zind_room_type_to_id
import os

# Explicit mapping rules for room types.
explicit_mapping = {
    "family room": "living room",   # family room -> living room
    "stair landing": "stairs",        # stair landing -> stairs
    "breakfast nook": "dining room",  # breakfast nook -> dining room
    "storage": "closet",              # storage -> closet
    "pantry": "closet",               # pantry -> closet
    "bonus room": "undefined",        # bonus room -> undefined
    "doorway": "undefined",           # doorway -> undefined
    "other": "undefined",             # other -> undefined
    "primary bedroom": "bedroom",     # primary bedroom -> bedroom
    "primary bathroom": "bathroom"    # primary bathroom -> bathroom
}

def convert_label(label):
    """
    Convert a room_type string using the following logic:
      1. If the label exactly matches an entry in explicit_mapping, use that.
      2. If the label exactly exists in zind_room_type_to_id, return it.
      3. If the label contains any of the keywords (kitchen, bathroom,
         hallway, closet, bedroom), return that keyword.
      4. Otherwise, return "undefined".
    """
    label = label.strip().lower()
    if label in explicit_mapping:
        return explicit_mapping[label]
    if label in zind_room_type_to_id:
        return label
    keywords = ["kitchen", "bathroom", "hallway", "closet", "bedroom"]
    for keyword in keywords:
        if keyword in label:
            return keyword
    return "undefined"

def process_json_file(input_file):
    """
    Reads a room_types_rectangles.json file, converts the room_type field for each entry,
    merges entries with the same room_type (by concatenating their polygons lists),
    and overwrites the original file with the updated data.
    """
    print(f"Processing JSON file: {input_file}")
    with open(input_file, "r") as f:
        data = json.load(f)
    
    merged = {}  # key: converted room_type, value: dict with room_type and merged polygons

    for entry in data:
        original_room_type = entry.get("room_type", "")
        new_room_type = convert_label(original_room_type)
        entry["room_type"] = new_room_type

        # Merge polygons if room_type already exists.
        if new_room_type not in merged:
            merged[new_room_type] = {
                "room_type": new_room_type,
                "polygons": entry.get("polygons", [])
            }
        else:
            merged[new_room_type]["polygons"].extend(entry.get("polygons", []))
    
    merged_list = list(merged.values())
    
    # Overwrite the original file with the merged data.
    with open(input_file, "w") as f_out:
        json.dump(merged_list, f_out, indent=4)
    
    print(f"File overwritten with converted data: {input_file}\n")

def main():
    # Base directory containing your dataset.
    base_dir = "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/zind/zind_data_set_80_fov_with_room_types"
    
    # Walk through the directory tree to process all matching JSON files.
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == "room_types_rectangles.json":
                input_file = os.path.join(root, file)
                process_json_file(input_file)

if __name__ == "__main__":
    main()
