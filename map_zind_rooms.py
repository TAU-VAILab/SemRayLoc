import os
import json

# Allowed mapping: only these keys are acceptable in the final output.
zind_room_type_to_id = {
    "bedroom": 0,
    "closet": 1,
    "hallway": 2,
    "living room": 3,
    "kitchen": 4,
    "bathroom": 5,
    "basement": 6,
    "dining room": 7,
    "loft": 8,
    "garage": 9,
    "laundry": 10,
    "office": 11,
    "stairs": 12,
    "undefined": 13
}

# Allowed keys for default substring matching.
# Note: 'utility' is purposefully omitted since it will be mapped to 'closet'
allowed_keys = [
    "bedroom", "closet", "hallway", "living room",
    "kitchen", "bathroom", "basement", "dining room",
    "loft", "garage", "laundry", "office", "stairs"
]

def map_label(label):
    """
    Map a given label to one of the allowed keys according to these rules:
    
      Special overrides:
      - If the label contains "breakfast nook", map to "dining room".
      - If the label contains "family room", map to "living room".
      - If the label contains any of "attic", "pantry", or "utility", map to "closet".
      - If the label contains "fireplace", map to "living room".
      - If the label contains any stairs-related substring (e.g., "stair"), map to "stairs".
    
      Otherwise:
      - If the label contains one of the allowed keys, return that key.
      - Else, return "undefined".
    """
    normalized = label.lower()
    
    # Special overrides:
    if "breakfast nook" in normalized:
        return "dining room"
    if "family room" in normalized:
        return "living room"
    if any(x in normalized for x in ["attic", "pantry", "utility"]):
        return "closet"
    if "fireplace" in normalized:
        return "living room"
    if "stair" in normalized:  # catches "stair landing", "stairs", etc.
        return "stairs"
    
    # Default: check if any allowed key is present.
    for key in allowed_keys:
        if key in normalized:
            return key
    
    return "undefined"

# Base directory for your dataset.
dataset_path = "/home/yuvalg/projects/Semantic_Floor_plan_localization/data/zind_perspective"

# Process all files recursively.
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        file_path = os.path.join(root, file)
        
        # Process JSON files containing room types.
        if file.endswith("room_types_rectangles.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                
                # First, map each room_type using our rules.
                # Then, merge entries with the same room_type.
                merged_entries = {}
                for entry in data:
                    original_label = entry.get("room_type", "")
                    mapped_label = map_label(original_label)
                    polygons = entry.get("polygons", [])
                    
                    # If already present, extend the polygons list.
                    if mapped_label in merged_entries:
                        merged_entries[mapped_label]["polygons"].extend(polygons)
                    else:
                        merged_entries[mapped_label] = {
                            "room_type": mapped_label,
                            "polygons": polygons.copy()
                        }
                
                # Convert merged_entries to a list.
                merged_data = list(merged_entries.values())
                
                # Save to a new file with a _mapped suffix.
                new_file_path = file_path.replace("room_types_rectangles.json", "room_types_rectangles_mapped.json")
                with open(new_file_path, "w") as f:
                    json.dump(merged_data, f, indent=4)
                print(f"Processed and saved (with merged coordinates): {new_file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Process text files with one room label per line.
        elif file.endswith("room_type_per_image.txt"):
            try:
                with open(file_path, "r") as f:
                    lines = f.readlines()
                new_lines = []
                for line in lines:
                    original_label = line.strip()
                    new_label = map_label(original_label)
                    new_lines.append(new_label + "\n")
                # Save to a new file with a _mapped suffix.
                new_file_path = file_path.replace("room_type_per_image.txt", "room_type_per_image_mapped.txt")
                with open(new_file_path, "w") as f:
                    f.writelines(new_lines)
                print(f"Processed and saved: {new_file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
