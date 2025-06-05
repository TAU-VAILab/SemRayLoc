"""
This module contains some common rendering routines for ZInD floor plans.
"""

import itertools
import logging
import sys
from typing import List, Tuple
import cv2
import numpy as np
from zind_utils import Polygon, PolygonType
import os

# Configure logging to display debug information
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
LOG = logging.getLogger(__name__)

# Default parameters when drawing ZInD floor plans.
DEFAULT_LINE_THICKNESS = 4
DEFAULT_RENDER_RESOLUTION = 2048

# Polygon colors for wall-only and semantic rendering.
WALL_ONLY_POLYGON_COLOR = {
    PolygonType.ROOM: (0, 0, 0),            # Black for walls
    PolygonType.WINDOW: (0, 0, 0),          # Black (treated like walls)
    PolygonType.DOOR: (255, 255, 255),      # White for doors
    PolygonType.OPENING: (0, 0, 0),
    PolygonType.PRIMARY_CAMERA: (255, 0, 0),
    PolygonType.SECONDARY_CAMERA: (255, 0, 0),
    PolygonType.PIN_LABEL: (255, 0, 0),
}

SEMANTIC_POLYGON_COLOR = {
    PolygonType.ROOM: (0, 0, 0),            # Black for walls
    PolygonType.WINDOW: (0, 0, 255),        # Blue for windows
    PolygonType.DOOR: (255, 0, 0),          # Red for doors
    PolygonType.OPENING: (0, 0, 0),
    PolygonType.PRIMARY_CAMERA: (255, 0, 0),
    PolygonType.SECONDARY_CAMERA: (255, 0, 0),
    PolygonType.PIN_LABEL: (255, 0, 0),
}
def save_camera_poses(
    polygon_list: List[Polygon],
    polygon_list_points: List[List[Tuple[float, float]]],
    output_path: str = ""
):
    # Prepare to collect camera poses
    camera_poses = []

    # Collect camera poses (centroids of the polygons)
    for polygon, points in zip(polygon_list, polygon_list_points):
        if polygon.type in [PolygonType.PRIMARY_CAMERA, PolygonType.SECONDARY_CAMERA]:
            # Compute the centroid of the polygon (assuming camera polygon points are small areas)
            centroid_x = np.mean([p[0] for p in points])
            centroid_y = np.mean([p[1] for p in points])
            camera_poses.append((centroid_x, centroid_y))

    # Write the camera poses to a file
    save_path = os.path.join(output_path, "posses.txt")
    with open(save_path, "w") as f:
        for pose in camera_poses:
            # Write x, y values for each camera pose
            f.write(f"{pose[0]} {pose[1]}\n")
            
def render_jpg_image(
    polygon_list: List[Polygon],
    *,
    jpg_file_name: str = None,
    thickness: int = DEFAULT_LINE_THICKNESS,
    output_width: int = DEFAULT_RENDER_RESOLUTION,
    rendering_type: str = "wall_only",  # Specify the rendering type
    output_path: str = ""
):
    """
    Render a set of ZInD polygon objects to an image that can be saved to the file system.

    :param polygon_list: List of Polygon objects.
    :param jpg_file_name: File name to save the image to (if None we won't save).
    :param thickness: The line thickness when drawing the polygons.
    :param output_width: The default output resolution.
    :param rendering_type: Either 'wall_only' or 'semantic' to choose the rendering style.

    :return: An OpenCV image object.
    """
    # Set the appropriate polygon color map based on rendering type
    if rendering_type == "semantic":
        polygon_color_map = SEMANTIC_POLYGON_COLOR
    else:
        polygon_color_map = WALL_ONLY_POLYGON_COLOR

    if not polygon_list:
        return np.ones([output_width, output_width, 3], dtype=np.uint8) * 255

    # Determine the bounds of the polygons
    min_x = min(point[0] for polygon in polygon_list for point in polygon.points)
    min_y = min(point[1] for polygon in polygon_list for point in polygon.points)

    # Normalize based on the upper-left corner
    polygon_list_points = []
    for polygon in polygon_list:
        polygon_modified = [(point[0] - min_x, point[1] - min_y) for point in polygon.points]
        polygon_list_points.append(polygon_modified)

    # Calculate the max bounds after normalizing
    max_x = max(point[0] for polygon in polygon_list_points for point in polygon)
    max_y = max(point[1] for polygon in polygon_list_points for point in polygon)

    resize_ratio = output_width / max(max_x, max_y) if max(max_x, max_y) != 0 else 1
    max_x *= resize_ratio
    max_y *= resize_ratio

    # Resize polygons based on the calculated ratio
    polygon_list_points_modified = [
        [(point[0] * resize_ratio, point[1] * resize_ratio) for point in polygon] 
        for polygon in polygon_list_points
    ]
    polygon_list_points = polygon_list_points_modified

    # Prepare the output image (white background)
    img_floor_map = np.ones([int(max_y) + 1, int(max_x) + 1, 3], dtype=np.uint8) * 255
    save_camera_poses(polygon_list, polygon_list_points, output_path)

    # Draw the walls and openings first
    for polygon, points in zip(polygon_list, polygon_list_points):
        if polygon.type in [PolygonType.ROOM, PolygonType.OPENING]:
            try:
                wall_thickness = thickness  # Walls and openings have default thickness
                color = polygon_color_map.get(polygon.type, (255, 255, 255))  # Default to white if type is not found

                cv2.polylines(
                    img_floor_map,
                    [np.int32([points])],
                    isClosed=True,
                    color=color,
                    thickness=wall_thickness,
                    lineType=cv2.LINE_8,
                )
                LOG.debug(f"Drew {'ROOM' if polygon.type == PolygonType.ROOM else 'OPENING'} polygon with points: {points}")
            except Exception as ex:
                LOG.debug(f"Error drawing wall or opening {jpg_file_name}: {polygon} {str(ex)}")
                continue

    # Draw other polygons except for walls and openings
    for polygon, points in zip(polygon_list, polygon_list_points):
        if polygon.type in [PolygonType.ROOM, PolygonType.OPENING]:
            continue  # Skip walls and openings
        try:
            if polygon.type in [PolygonType.PRIMARY_CAMERA, PolygonType.SECONDARY_CAMERA, PolygonType.PIN_LABEL]:
                continue  # Skip camera and pin label types

            if rendering_type == "wall_only" and polygon.type == PolygonType.WINDOW:
                current_thickness = thickness                
            else:
                current_thickness = thickness if polygon.type == PolygonType.ROOM else 2 * thickness

            color = polygon_color_map.get(polygon.type, (255, 255, 255))  # Default to white if type is not found

            cv2.polylines(
                img_floor_map,
                [np.int32([points])],
                isClosed=True,
                color=color,
                thickness=current_thickness,
                lineType=cv2.LINE_8,
            )
            LOG.debug(f"Drew polygon of type {polygon.type} with points: {points}")
        except Exception as ex:
            LOG.debug(f"Error drawing polygon {jpg_file_name}: {polygon} {str(ex)}")
            continue
    if rendering_type == "wall_only":
        all_points = []
        # Iterate over polygon_list and polygon_list_points simultaneously
        for polygon, polygon_points in zip(polygon_list, polygon_list_points):
            # Skip camera types
            if polygon.type in [PolygonType.DOOR, PolygonType.ROOM]:    
                all_points.extend(polygon_points)  # Flatten all polygon points into a single list

        # Now iterate through all points and compare them
        for i, point1 in enumerate(all_points):
            for j, point2 in enumerate(all_points):
                if i != j:  # Ensure we're not comparing the same point
                    # Calculate the Euclidean distance between the two points
                    distance = np.linalg.norm(np.array(point1) - np.array(point2))
                    if 1<= distance <= 0:  # Distance threshold
                        # Draw a line between the points if they are within the proximity threshold
                        cv2.line(
                            img_floor_map, 
                            tuple(np.int_(point1)), 
                            tuple(np.int_(point2)), 
                            (0, 0, 0),  # Red color for the lines between close points
                            thickness=thickness,  # Line thickness
                            lineType=cv2.LINE_AA
                        )
        
                            
    # Save the image if a filename is provided
    if jpg_file_name:
        success = cv2.imwrite(jpg_file_name, img_floor_map)
        if success:
            LOG.info(f"Image saved successfully at {jpg_file_name}")
        else:
            LOG.error(f"Failed to save image at {jpg_file_name}")

    return img_floor_map
