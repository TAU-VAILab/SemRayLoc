# """CLI script to visualize & validate data for the public-facing Zillow Indoor Dataset (ZInD).
#
# Validation includes:
#  (1) required JSON fields are presented
#  (2) verify non self-intersection of room floor_plan_layouts
#  (3) verify that windows/doors/openings lie on the room layout geometry
#  (4) verify that windows/doors/openings are defined by two points (left/right boundaries)
#  (5) verify that panos_layouts are RGB images with valid FoV ratio (2:1)
#
# Visualization includes:
#  (1) render the top-down floor map projection: merged room floor_plan_layouts,WDO and camera centers
#  (2) render the room floor_plan_layouts and windows/doors/openings on the pano
#
# Example usage (1): Render all layouts on primary and secondary panos.
#  python visualize_zind_cli.py -i <input_folder> -o <output_folder> --visualize-layout --visualize-floor-plan \
#  --raw --complete --visible --primary --secondary
#
# Example usage (2): Render all vector layouts using merger (based on raw or complete) and the final redraw layouts.
#  python visualize_zind_cli.py -i <input_folder> -o <output_folder> --visualize-floor-plan --redraw --complete --raw
#  python visualize_zind_cli_org.py -i /datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/zind/sample_tour -o /datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/zind/sample_tour/results_2 --visualize-floor-plan --redraw --complete --raw
# Example usage (3): Render the raster to vector alignments using merger (based on raw or complete) and final redraw.
#  python visualize_zind_cli.py -i <input_folder> -o <output_folder> --visualize-raster --redraw --complete --raw
#

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, Any
from zind_utils import Polygon, PolygonType

from floor_plan import FloorPlan
from render_org import (
    render_jpg_image,
)
from tqdm import tqdm

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
LOG = logging.getLogger(__name__)

RENDER_FOLDER = "render_data"


def validate_and_render(
    zillow_floor_plan: "FloorPlan",
    *,
    output_folder: str,
):
    """Validate and render various ZInD elements, e.g.
    1. Primary/secondary layout and WDO
    2. Raw/complete/visible layouts
    3. Top-down merger results (draft floor-plan)
    4. Top-down redraw results (final floor-plan)
    5. Raster to vector alignment results.

    :param zillow_floor_plan: ZInD floor plan object.
    :param input_folder: Input folder of the current tour.
    :param output_folder: Folder where the renderings will be saved.
    :param args: Input arguments to the script.

    :return: None
    """
    # Get the types of floor_plan_layouts that we should render.
    geometry_to_visualize = []
    geometry_to_visualize.append("complete")
        
    # Render the top-down draft floor plan, result of the merger stage.
    output_folder_floor_plan = os.path.join(output_folder, "floor_plan")
    os.makedirs(output_folder_floor_plan, exist_ok=True)

    for geometry_type in geometry_to_visualize:

        zind_dict = zillow_floor_plan.floor_plan_layouts[geometry_type]

        for i,(_, zind_poly_list) in enumerate(zind_dict.items()):
            render_jpg_image(
                polygon_list=zind_poly_list, jpg_file_name=os.path.join(output_folder_floor_plan,f"floor_{i}_{geometry_type}_wall_only.png"), rendering_type = "wall_only", output_path= output_folder_floor_plan
            )
            render_jpg_image(
                polygon_list=zind_poly_list, jpg_file_name=os.path.join(output_folder_floor_plan,f"floor_{i}_{geometry_type}_semantic.png"), rendering_type = "semantic", output_path= output_folder_floor_plan
            )


def main():
    input = "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/zind/zind_raw"
    output_path = "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/zind/temp"

    # Collect all the feasible input JSON files.
    input_files_list = [input]
    if Path(input).is_dir():
        input_files_list = sorted(Path(input).glob("**/zind_data.json"))

    num_failed = 0
    num_success = 0
    failed_tours = []
    for input_file in tqdm(input_files_list, desc="Validating ZInD data"):
        # Try loading and validating the file.
        try:
            zillow_floor_plan = FloorPlan(input_file)

            current_output_folder = os.path.join(output_path, RENDER_FOLDER, str(Path(input_file).parent.stem))
            os.makedirs(current_output_folder, exist_ok=True)

            validate_and_render(
                zillow_floor_plan,
                output_folder=current_output_folder,
            )
            num_success += 1
            
        except Exception as ex:
            failed_tours.append(str(Path(input_file).parent.stem))
            num_failed += 1
            track = traceback.format_exc()
            print("Error validating {}: {}".format(input_file, str(ex)))
            continue

    if num_failed > 0:
        print("Failed to validate: {}".format(num_failed))    
    else:
        print("All ZInD validated successfully")


if __name__ == "__main__":
    main()
