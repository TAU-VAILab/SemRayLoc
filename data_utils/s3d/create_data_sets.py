"""
Data Set Creation Module for Structured3D Dataset

This module provides functionality to process and create datasets from the Structured3D dataset.
It handles the conversion of perspective images and creates organized datasets for training and testing.

The module processes scene directories containing perspective images and generates corresponding
casting files and organized datasets with specified resolution and parameters.
"""

import os
from create_casting_files import process_scene as process_casting_scene #For perspective images

def process_all_scenes(base_path, output_base_path, resolution=0.01, dpi=100, fov_segments=40):
    """
    Process all scenes in the base directory and create corresponding datasets.

    Args:
        base_path (str): Path to the directory containing scene folders
        output_base_path (str): Path where the processed datasets will be saved
        resolution (float, optional): Resolution in meters per pixel. Defaults to 0.01
        dpi (int, optional): DPI for output images. Defaults to 100
        fov_segments (int, optional): Number of segments to divide the Field of View into. Defaults to 40

    Returns:
        None: The function processes scenes and saves outputs to the specified directory
    """
    scenes = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d.startswith('scene_')] 
    scenes.sort(key=lambda x: int(x.split('_')[-1]))
    total_scenes = len(scenes)
    print(f"Total scenes to process: {total_scenes}\n")
    failed_scenes = []
    
    for idx, scene_dir in enumerate(scenes, start=1):
        scene_id = str(int(scene_dir.split('_')[-1]))
        try:
            print(f"Processing Scene {scene_id} ({idx}/{total_scenes})...")
            process_casting_scene(scene_id, base_path, output_base_path, resolution, dpi, fov_segments)
            print(f"Finished processing Scene {scene_id}.\n")
        except Exception as e:
            print(f"Failed to process Scene {scene_id}. Error: {e}\n")
            failed_scenes.append(scene_id)
    
    if failed_scenes:
        print("The following scenes failed to process:")
        for scene in failed_scenes:
            print(f"Scene {scene}")


def main():
    """
    Main function to demonstrate the usage of the dataset creation process.
    Sets up default parameters and processes the Structured3D dataset.
    """
    base_path = os.path.join("Data", "raw_S3D_perspective")
    output_base_path = os.path.join("Data", "processed_S3D")
    resolution = 0.01  # Resolution in meters per pixel
    dpi = 100          # DPI for output images
    fov_segments = 40  # Number of segments to divide the FOV into

    process_all_scenes(base_path, output_base_path, resolution, dpi, fov_segments)

if __name__ == "__main__":
    main()
