"""
This module contains some common rendering routines for ZInD floor plans,
modified to use a fixed 100 pixels per meter scaling.
"""

import logging
import sys
from typing import List, Tuple
import cv2
import numpy as np
from zind_utils import Polygon, PolygonType
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import MultiPolygon


# Configure logging to display debug information
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Default parameters when drawing ZInD floor plans.
DEFAULT_LINE_THICKNESS = 5

# We no longer need DEFAULT_RENDER_RESOLUTION if we are forcing 100 px/m
# But you can keep it if desired:
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
    PolygonType.ROOM: (0, 0, 0),           # Black for walls
    PolygonType.WINDOW: (255, 0, 0),         # Blue in RGB, but to get blue in OpenCV (BGR) it must be (255, 0, 0)
    PolygonType.DOOR: (0, 0, 255),           # Red in RGB, but in OpenCV (BGR) it's (0, 0, 255)
    PolygonType.OPENING: (0, 0, 0),
    PolygonType.PRIMARY_CAMERA: (0, 0, 255),
    PolygonType.SECONDARY_CAMERA: (0, 0, 255),
    PolygonType.PIN_LABEL: (0, 0, 255),
}
def shapely_poly_to_cv2_pts(shapely_poly: ShapelyPolygon) -> np.ndarray:
    """
    Convert a Shapely polygon's exterior coordinates to a NumPy array formatted
    for OpenCV.
    """
    exterior_coords = np.array(shapely_poly.exterior.coords, dtype=np.int32)
    return exterior_coords.reshape((-1, 1, 2))

def render_jpg_image(
    polygon_list: List[Polygon],
    *,
    polygon_list_points: List[List[Tuple[float, float]]],
    jpg_file_name: str = None,
    thickness: int = DEFAULT_LINE_THICKNESS,
    rendering_type: str = "wall_only",  # "semantic" or "wall_only"
    output_path: str = "",
    floor_scale: float = 1.0,
    px_per_meter: float = 100.0  # <--- Force 100 px per meter
):
    """
    Render a set of ZInD polygon objects to an image, using a fixed scale 
    of `px_per_meter` pixels per meter (default=100).

    :param polygon_list:         List of Polygon objects (ROOM, DOOR, etc.).
    :param polygon_list_points:  Matching list of polygon coordinates (in meters).
    :param jpg_file_name:        Path to save the image file, or None to skip saving.
    :param thickness:            The line thickness when drawing polygons.
    :param rendering_type:       'wall_only' or 'semantic'.
    :param px_per_meter:         Scaling factor (pixels per meter).
    :return:                     The rendered OpenCV image (numpy array).
    """

    # Pick color map
    if rendering_type == "semantic":
        polygon_color_map = SEMANTIC_POLYGON_COLOR
    else:
        polygon_color_map = WALL_ONLY_POLYGON_COLOR

    # 1) First, scale polygon_list_points by 100 px/m (or px_per_meter).
    scaled_polygons = []
    for polygon_points in polygon_list_points:
        scaled_points = [(p[0] * px_per_meter, p[1] * px_per_meter) 
                         for p in polygon_points]
        scaled_polygons.append(scaled_points)

    # 2) Figure out bounding box: find max_x, max_y
    max_x = 0.0
    max_y = 0.0
    for poly in scaled_polygons:
        for (x, y) in poly:
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y

    # 3) Build a blank (white) image of size (ceil(max_y)+1, ceil(max_x)+1)
    H = int(np.ceil(max_y)) + 1
    W = int(np.ceil(max_x)) + 1
    img_floor_map = np.ones((H, W, 3), dtype=np.uint8) * 255

    # 4) Draw polygons
    #        (A) ROOM and OPENING polygons first with extruded (outward offset) boundary.
    # for polygon, scaled_points in zip(polygon_list, scaled_polygons):
        # if polygon.type in [PolygonType.ROOM, PolygonType.OPENING]:
        #     try:
        #         # Set the extrusion (offset) distance.
        #         wall_thickness = 4  # This is the extra width to be added outward.
        #         color = polygon_color_map.get(polygon.type, (255, 255, 255))

        #         # Create a Shapely polygon from the scaled points.
        #         original_poly = ShapelyPolygon(scaled_points)
        #         if not original_poly.is_valid or original_poly.is_empty:
        #             continue

        #         # Create an outward offset polygon by buffering.
        #         offset_poly = original_poly.buffer(6)

        #         # Compute only the extruded "ring" (difference between the offset and original).
        #         extruded_ring = offset_poly.difference(original_poly)
        #         if extruded_ring.is_empty:
        #             continue

        #         # The difference may return a MultiPolygon. Iterate over parts.
        #         if isinstance(extruded_ring, MultiPolygon):
        #             for part in extruded_ring.geoms:
        #                 contour = shapely_poly_to_cv2_pts(part)
        #                 cv2.polylines(
        #                     img_floor_map,
        #                     [contour],
        #                     isClosed=True,
        #                     color=color,
        #                     thickness=4,  # border thickness is already encoded in the geometry
        #                     lineType=cv2.LINE_8,
        #                 )
        #         else:
        #             contour = shapely_poly_to_cv2_pts(extruded_ring)
        #             cv2.polylines(
        #                 img_floor_map,
        #                 [contour],
        #                 isClosed=True,
        #                 color=color,
        #                 thickness=4,
        #                 lineType=cv2.LINE_8,
        #             )

        #     except Exception as ex:
        #         logging.exception("Error drawing extruded room/opening polygon:")
        #         continue
    
    for polygon, scaled_points in zip(polygon_list, scaled_polygons):
        if polygon.type in [PolygonType.ROOM, PolygonType.OPENING]:
            # Convert the scaled points to a NumPy array of integers for cv2.
            pts = np.array(scaled_points, dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Get a color for the polygon, defaulting to white if not specified.
            color = polygon_color_map.get(polygon.type, (255, 255, 255))
            
            # Draw the polygon by connecting all vertices.
            cv2.polylines(
                img_floor_map,
                [pts],
                isClosed=True,
                color=color,
                thickness=8,         # Adjust thickness as needed.
                lineType=cv2.LINE_8,
            )


    #    (B) Other polygons (DOOR, WINDOW, etc.)
    for polygon, scaled_points in zip(polygon_list, scaled_polygons):
        if polygon.type in [PolygonType.ROOM, PolygonType.OPENING, PolygonType.PARTIAL_ROOM]:
            continue
        
        try:
            # Skip cameras or pins if you don't want them drawn
            if polygon.type in [
                PolygonType.PRIMARY_CAMERA,
                PolygonType.SECONDARY_CAMERA,
                PolygonType.PIN_LABEL
            ]:
                continue

            # Window in wall-only => thickness = thickness
            # Door => 2 * thickness, etc. (custom logic)
            if  polygon.type == PolygonType.WINDOW:
                current_thickness = 10
            else:
                current_thickness = 10

            color = polygon_color_map.get(polygon.type, (255, 255, 255))

            # -- NEW LOGIC: chunk the points in groups of 3, then use only the first 2 for each segment --
            # e.g. if length = 3, we get one segment [0,1]
            # if length = 6, we get segments [0,1] and [3,4], etc.
            if len(scaled_points) % 3 == 0 and len(scaled_points) > 2:
                for i in range(0, len(scaled_points), 3):
                    segment = scaled_points[i : i + 2]  # keep only the first 2 points
                    if len(segment) == 2:
                        cv2.polylines(
                            img_floor_map,
                            [np.int32([segment])],
                            isClosed=True,
                            color=color,
                            thickness=current_thickness,
                            lineType=cv2.LINE_8,
                        )
            else:
                # if it's not a multiple of 3 or fewer than 3 points,
                # just draw them as-is (or handle differently if you prefer)
                cv2.polylines(
                    img_floor_map,
                    [np.int32([scaled_points])],
                    isClosed=True,
                    color=color,
                    thickness=current_thickness,
                    lineType=cv2.LINE_8,
                )
        except Exception as ex:
            continue

    # 6) Save if requested
    if jpg_file_name:
        success = cv2.imwrite(jpg_file_name, img_floor_map)
        if not success:
            print(f"Failed to save image at {jpg_file_name}")


    return img_floor_map
