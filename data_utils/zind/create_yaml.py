#!/usr/bin/env python3
"""
YAML configuration generator for ZInD dataset splits.
"""
import os
import json
import glob
import yaml
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration settings."""
    partition_json_path: str = os.path.join("Data", "zind", "zind_raw", "partition.json")
    base_dir: str = os.path.join("Data", "zind", "raw_perspective")
    output_yaml_path: str = os.path.join("Data", "zind", "raw_perspective", "split.yaml")

def load_partition(partition_json_path: str) -> Dict[str, List[str]]:
    """Load partition JSON file."""
    try:
        with open(partition_json_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load partition file: {e}")
        raise

def collect_scene_floors(scene_id: str, base_dir: str) -> List[str]:
    """Collect floor folders for a scene ID."""
    padded_scene_id = scene_id.zfill(4)
    pattern = os.path.join(base_dir, f"scene_{padded_scene_id}_floor_*")
    
    floors: Set[str] = set()
    for path in glob.glob(pattern):
        if os.path.isdir(path):
            floors.add(os.path.basename(path))
    
    return sorted(list(floors))

def build_split_dict(partition: Dict[str, List[str]], base_dir: str) -> Dict[str, List[str]]:
    """Build split dictionary from partition data."""
    split: Dict[str, List[str]] = {}
    for phase in ['train', 'val', 'test']:
        scene_ids = partition.get(phase, [])
        floors_list: List[str] = []
        
        for scene_id in scene_ids:
            scene_floors = collect_scene_floors(scene_id, base_dir)
            if scene_floors:
                floors_list.extend(scene_floors)
            else:
                logger.warning(f"No floor folders found for scene {scene_id.zfill(4)}")
        
        split[phase] = sorted(set(floors_list))
    
    return split

def write_yaml(split_dict: Dict[str, List[str]], output_yaml_path: str) -> None:
    """Write split dictionary to YAML file."""
    try:
        os.makedirs(os.path.dirname(output_yaml_path), exist_ok=True)
        with open(output_yaml_path, "w") as f:
            yaml.dump(split_dict, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Wrote YAML file to {output_yaml_path}")
    except Exception as e:
        logger.error(f"Failed to write YAML file: {e}")
        raise

def main() -> None:
    """Generate YAML configuration for dataset splits."""
    try:
        config = Config()
        
        # Load partition and build splits
        partition = load_partition(config.partition_json_path)
        split_dict = build_split_dict(partition, config.base_dir)
        
        # Write configuration
        write_yaml(split_dict, config.output_yaml_path)
        
        # Log statistics
        for phase, floors in split_dict.items():
            logger.info(f"{phase.capitalize()} split: {len(floors)} floors")
            
    except Exception as e:
        logger.error(f"Failed to generate YAML configuration: {e}")
        raise

if __name__ == "__main__":
    main()
