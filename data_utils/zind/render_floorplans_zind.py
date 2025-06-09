"""
ZInD Floor Plan Rendering Module

This module provides rendering routines for ZInD floor plans using a fixed scale
of 100 pixels per meter. It supports both wall-only and semantic rendering modes.
"""

# Standard library imports
import logging
import sys
from typing import List, Tuple, Dict, Optional

# Third-party imports
import cv2
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import MultiPolygon

# Local imports
from zind_utils import Polygon, PolygonType

# Configure logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Rendering constants
DEFAULT_LINE_THICKNESS = 5
DEFAULT_RENDER_RESOLUTION = 2048
PIXELS_PER_METER = 100.0

# Color mappings for different rendering types
WALL_ONLY_COLORS: Dict[PolygonType, Tuple[int, int, int]] = {
    PolygonType.ROOM: (0, 0, 0),            # Black for walls
    PolygonType.WINDOW: (0, 0, 0),          # Black (treated like walls)
    PolygonType.DOOR: (255, 255, 255),      # White for doors
    PolygonType.OPENING: (0, 0, 0),
    PolygonType.PRIMARY_CAMERA: (255, 0, 0),
    PolygonType.SECONDARY_CAMERA: (255, 0, 0),
    PolygonType.PIN_LABEL: (255, 0, 0),
}

SEMANTIC_COLORS: Dict[PolygonType, Tuple[int, int, int]] = {
    PolygonType.ROOM: (0, 0, 0),           # Black for walls
    PolygonType.WINDOW: (255, 0, 0),       # Red in BGR
    PolygonType.DOOR: (0, 0, 255),         # Blue in BGR
    PolygonType.OPENING: (0, 0, 0),
    PolygonType.PRIMARY_CAMERA: (0, 0, 255),
    PolygonType.SECONDARY_CAMERA: (0, 0, 255),
    PolygonType.PIN_LABEL: (0, 0, 255),
}

def shapely_poly_to_cv2_pts(shapely_poly: ShapelyPolygon) -> np.ndarray:
    """Convert Shapely polygon coordinates to OpenCV format.
    
    Args:
        shapely_poly: Shapely polygon object
        
    Returns:
        NumPy array of points in OpenCV format
    """
    exterior_coords = np.array(shapely_poly.exterior.coords, dtype=np.int32)
    return exterior_coords.reshape((-1, 1, 2))

def render_jpg_image(
    polygon_list: List[Polygon],
    *,
    polygon_list_points: List[List[Tuple[float, float]]],
    jpg_file_name: Optional[str] = None,
    thickness: int = DEFAULT_LINE_THICKNESS,
    rendering_type: str = "wall_only",
    output_path: str = "",
    floor_scale: float = 1.0,
    px_per_meter: float = PIXELS_PER_METER
) -> np.ndarray:
    """Render ZInD polygons to an image with fixed scale.
    
    Args:
        polygon_list: List of Polygon objects (ROOM, DOOR, etc.)
        polygon_list_points: List of polygon coordinates in meters
        jpg_file_name: Path to save the image file, or None to skip saving
        thickness: Line thickness for drawing polygons
        rendering_type: 'wall_only' or 'semantic'
        output_path: Output directory path
        floor_scale: Scale factor for floor dimensions
        px_per_meter: Pixels per meter scaling factor
        
    Returns:
        Rendered OpenCV image as numpy array
    """
    # Select color map based on rendering type
    polygon_color_map = SEMANTIC_COLORS if rendering_type == "semantic" else WALL_ONLY_COLORS

    # Scale polygon points to pixel coordinates
    scaled_polygons = [
        [(p[0] * px_per_meter, p[1] * px_per_meter) for p in polygon_points]
        for polygon_points in polygon_list_points
    ]

    # Calculate image dimensions
    max_x = max(p[0] for poly in scaled_polygons for p in poly)
    max_y = max(p[1] for poly in scaled_polygons for p in poly)
    height = int(np.ceil(max_y)) + 1
    width = int(np.ceil(max_x)) + 1

    # Create blank white image
    img_floor_map = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Draw room and opening polygons
    for polygon, scaled_points in zip(polygon_list, scaled_polygons):
        if polygon.type in [PolygonType.ROOM, PolygonType.OPENING]:
            pts = np.array(scaled_points, dtype=np.int32).reshape((-1, 1, 2))
            color = polygon_color_map.get(polygon.type, (255, 255, 255))
            
            cv2.polylines(
                img_floor_map,
                [pts],
                isClosed=True,
                color=color,
                thickness=8,
                lineType=cv2.LINE_8
            )

    # Draw other polygons (doors, windows, etc.)
    for polygon, scaled_points in zip(polygon_list, scaled_polygons):
        if polygon.type in [PolygonType.ROOM, PolygonType.OPENING, PolygonType.PARTIAL_ROOM]:
            continue
        
        try:
            # Skip camera and pin polygons
            if polygon.type in [
                PolygonType.PRIMARY_CAMERA,
                PolygonType.SECONDARY_CAMERA,
                PolygonType.PIN_LABEL
            ]:
                continue

            # Set thickness based on polygon type
            current_thickness = 10  # Default thickness
            color = polygon_color_map.get(polygon.type, (255, 255, 255))

            # Handle special case for windows (3-point segments)
            if len(scaled_points) % 3 == 0 and len(scaled_points) > 2:
                for i in range(0, len(scaled_points), 3):
                    segment = scaled_points[i:i + 2]
                    if len(segment) == 2:
                        cv2.polylines(
                            img_floor_map,
                            [np.int32([segment])],
                            isClosed=True,
                            color=color,
                            thickness=current_thickness,
                            lineType=cv2.LINE_8
                        )
            else:
                cv2.polylines(
                    img_floor_map,
                    [np.int32([scaled_points])],
                    isClosed=True,
                    color=color,
                    thickness=current_thickness,
                    lineType=cv2.LINE_8
                )
        except Exception as ex:
            logging.warning(f"Failed to draw polygon: {ex}")
            continue

    # Save image if filename provided
    if jpg_file_name:
        success = cv2.imwrite(jpg_file_name, img_floor_map)
        if not success:
            logging.error(f"Failed to save image at {jpg_file_name}")

    return img_floor_map
