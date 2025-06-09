"""YAML configuration generator for dataset splits."""

import os
from typing import List, Dict
import logging
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration settings."""
    TOTAL_SCENES: int = 3500
    TRAIN_SPLIT_END: int = 3000
    VAL_SPLIT_END: int = 3250
    TEST_SPLIT_END: int = 3500
    
    BASE_DIR: str = os.path.join("Data", "raw_S3D_perspective")
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "structured3d_perspective")
    YAML_FILENAME: str = "split.yaml"
    
    def validate(self) -> None:
        """Validate configuration settings."""
        if not (0 < self.TRAIN_SPLIT_END < self.VAL_SPLIT_END < self.TEST_SPLIT_END <= self.TOTAL_SCENES):
            raise ValueError("Invalid split configuration")
        if not self.YAML_FILENAME.endswith('.yaml'):
            raise ValueError("YAML_FILENAME must end with .yaml")

def generate_scene_ids(start: int, end: int) -> List[str]:
    """Generate scene IDs for a given range."""
    if start < 0 or end <= start:
        raise ValueError(f"Invalid range: start={start}, end={end}")
    return [f"scene_{i:05d}" for i in range(start, end)]

def create_split_configuration(config: Config) -> Dict[str, List[str]]:
    """Create dataset split configuration."""
    config.validate()
    return {
        'train': generate_scene_ids(0, config.TRAIN_SPLIT_END),
        'val': generate_scene_ids(config.TRAIN_SPLIT_END, config.VAL_SPLIT_END),
        'test': generate_scene_ids(config.VAL_SPLIT_END, config.TEST_SPLIT_END)
    }

def format_yaml_list(lst: List[str]) -> str:
    """Format list of strings for YAML output."""
    if not lst:
        return "[]"
    formatted_list = "[\n"
    formatted_list += ",\n".join(f'  "{item}"' for item in lst)
    formatted_list += ",\n]"
    return formatted_list

def ensure_directory_exists(directory: str) -> None:
    """Ensure directory exists."""
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directory {directory}: {e}")
        raise

def write_yaml_file(data: Dict[str, List[str]], output_path: str) -> None:
    """Write split configuration to YAML file."""
    if not data:
        raise ValueError("Empty data dictionary provided")
    
    ensure_directory_exists(os.path.dirname(output_path))
    
    yaml_content = ""
    for split_name, scenes in data.items():
        if not scenes:
            logger.warning(f"Empty scene list for {split_name} split")
        yaml_content += f"{split_name}:\n{format_yaml_list(scenes)}\n\n"
    
    try:
        with open(output_path, 'w') as file:
            file.write(yaml_content)
        logger.info(f"Created YAML file: {output_path}")
    except Exception as e:
        logger.error(f"Failed to write YAML file: {e}")
        raise

def main() -> None:
    """Generate and write split configuration."""
    try:
        config = Config()
        config.validate()
        
        split_config = create_split_configuration(config)
        output_path = os.path.join(config.OUTPUT_DIR, config.YAML_FILENAME)
        
        write_yaml_file(split_config, output_path)
        
        total_scenes = sum(len(scenes) for scenes in split_config.values())
        logger.info(f"Total scenes: {total_scenes}")
        for split_name, scenes in split_config.items():
            logger.info(f"{split_name.capitalize()}: {len(scenes)} scenes")
            
    except Exception as e:
        logger.error(f"Failed to generate YAML: {e}")
        raise

if __name__ == "__main__":
    main()
