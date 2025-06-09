"""
ZInD Dataset Processing Module

This module handles the processing of ZInD dataset scenes, converting raw data into
a structured format suitable for training. It processes multiple scenes in sequence,
handling depth maps, semantic labels, and room information.
"""

# Standard library imports
import os
import logging
from typing import List

# Third-party imports
from tqdm import tqdm

# Local imports
from data_utils.zind.create_casting_files_zind import process_scene as process_casting_scene

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress noisy library logs
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

class Config:
    """Configuration settings for ZInD dataset processing."""
    # Directory configuration
    BASE_DIR = os.path.join("Data", "zind")
    RAW_DIR = os.path.join(BASE_DIR, "raw_perspective")
    PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
    
    # Scene configuration
    TOTAL_SCENES = 1600
    SCENE_ID_FORMAT = "{:04d}"

def get_scene_list(base_path: str) -> List[str]:
    """Get list of valid scene directories.
    
    Args:
        base_path: Base directory containing scene folders
        
    Returns:
        List of valid scene directory names
    """
    scenes_to_process = [
        Config.SCENE_ID_FORMAT.format(i) 
        for i in range(Config.TOTAL_SCENES)
    ]
    return [
        d for d in scenes_to_process 
        if os.path.isdir(os.path.join(base_path, d))
    ]

def process_all_scenes(
    base_path: str,
    output_base_path: str,
) -> None:
    """Process all scenes in the dataset.
    
    Args:
        base_path: Directory containing raw scene data
        output_base_path: Directory to save processed data
    """
    # Get list of scenes to process
    scenes = get_scene_list(base_path)
    total_scenes = len(scenes)
    logger.info(f"Total scenes to process: {total_scenes}")
    
    # Track failed scenes
    failed_scenes = []
    
    # Process each scene
    for idx, scene_name in tqdm(enumerate(scenes, start=1), total=total_scenes):
        try:
            logger.info(f"Processing Scene {scene_name} ({idx}/{total_scenes})...")
            
            process_casting_scene(
                scene_name,
                base_path,
                output_base_path
            )
            
            logger.info(f"Finished processing Scene {scene_name}")
            
        except Exception as e:
            logger.error(f"Failed to process Scene {scene_name}: {e}")
            failed_scenes.append(scene_name)
    
    # Report failed scenes
    if failed_scenes:
        logger.warning("The following scenes failed to process:")
        for scene in failed_scenes:
            logger.warning(f"Scene {scene}")
    else:
        logger.info("All scenes processed successfully")

def main():
    """Main function to process the ZInD dataset."""
    # Create output directory if it doesn't exist
    os.makedirs(Config.PROCESSED_DIR, exist_ok=True)
    
    # Process all scenes
    process_all_scenes(
        base_path=Config.RAW_DIR,
        output_base_path=Config.PROCESSED_DIR
    )

if __name__ == "__main__":
    main()
