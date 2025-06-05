import os
import json
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box, Point
import argparse
from PIL import Image, ImageOps
from modules.semantic.semantic_mapper import room_type_to_id

# -----------------------------------
# Utility to convert lines to vertices
# -----------------------------------
def convert_lines_to_vertices(lines):
    """
    lines : List[List[int]] or np.ndarray of shape (N, 2)
      Each row is [start_junction, end_junction].

    Returns: List of polygons, each polygon is a list of junction indices,
      e.g. [[0,1,2,3,0], ... ] if it closes.

    This version won't crash if the lines don't form a perfect loop. If we
    fail to find a next line that connects, we finalize the polygon as-is
    and move on.
    """
    import numpy as np
    polygons = []
    lines = np.array(lines, dtype=int)  # Ensure array of shape (N,2).

    polygon = None
    while len(lines) != 0:
        # If we are NOT currently building a polygon, start one from the first line
        if polygon is None:
            polygon = lines[0].tolist()  # e.g. [start_junc, end_junc]
            lines = np.delete(lines, 0, axis=0)
            continue

        # Try to find a line in 'lines' that starts (or ends) where the polygon's
        # last vertex is.  i.e. lines == polygon[-1].
        last_vertex = polygon[-1]
        lineID, juncID = np.where(lines == last_vertex)

        if len(lineID) == 0:
            # No line continuing from 'last_vertex'. We'll finalize the current polygon:
            polygons.append(polygon)
            polygon = None
            continue

        # Otherwise, pick the first match
        chosen_line_idx = lineID[0]
        chosen_line_col = juncID[0]  # 0 or 1
        # The other column is 1 - chosen_line_col
        vertex = lines[chosen_line_idx, 1 - chosen_line_col]

        # Remove this line from 'lines'
        lines = np.delete(lines, chosen_line_idx, axis=0)

        # Check if we closed the polygon
        if vertex in polygon:
            # We have looped back to an existing vertex, so finalize
            polygons.append(polygon)
            polygon = None
        else:
            # Just append and keep going
            polygon.append(vertex)

    # If we exit the while loop but 'polygon' is not None, we have a leftover
    if polygon is not None:
        polygons.append(polygon)

    return polygons


# -----------------------------------
# Plot the floorplan with semantic colors
# -----------------------------------
def plot_floorplan_semantics(annos, polygons, scene_number, output_dir,
                             resolution=0.005, dpi=1000):
    """
    Creates an (oversized) figure, plots outwalls in black and
    other polygons in color (walls, windows, doors), then saves as
    floorplan_semantic.png.
    """
    resolution_mm = resolution * 1000  # Convert to mm/pixel
    
    # Hardcoded figure size: 80m x 80m in mm
    fig_width_mm = 80000
    fig_height_mm = 80000
    
    width_pixels = int(fig_width_mm / resolution_mm)
    height_pixels = int(fig_height_mm / resolution_mm)
    
    fig_width_in = width_pixels / dpi
    fig_height_in = height_pixels / dpi
    
    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=dpi)
    ax.set_xlim(-fig_width_mm / 2, fig_width_mm / 2)
    ax.set_ylim(-fig_height_mm / 2, fig_height_mm / 2)
    ax.set_aspect('equal')
    junctions = np.array([junc['coordinate'][:2] for junc in annos['junctions']])
    
    # 1) Plot outwall polygons in black
    for (polygon_indices, poly_type) in polygons:
        if poly_type != 'outwall':
            continue
        if len(polygon_indices) == 0:
            continue
        try:
            polygon_indices = np.array(polygon_indices, dtype=int)
            polygon_coords = junctions[polygon_indices]
            polygon_shape = Polygon(polygon_coords)
            if polygon_shape.is_empty:
                continue
            # Fill with black
            color = 'black'
            x, y = polygon_shape.exterior.xy
            ax.fill(x, y, color=color, alpha=1.0)
        except Exception as e:
            print(f"Error plotting polygon for {poly_type}: {e}")
            continue
    
    # 2) Plot everything else except door/window/outwall
    for (polygon_indices, poly_type) in polygons:
        if poly_type in ['door', 'window', 'outwall']:
            continue
        if len(polygon_indices) == 0:
            continue
        try:
            polygon_indices = np.array(polygon_indices, dtype=int)
            polygon_coords = junctions[polygon_indices]
            polygon_shape = Polygon(polygon_coords)
            if polygon_shape.is_empty:
                continue
            # Fill with white, outline black
            color = 'white'
            x, y = polygon_shape.exterior.xy
            ax.fill(x, y, color=color, alpha=1.0)
            ax.plot(x, y, color='black', alpha=1.0)
        except Exception as e:
            print(f"Error plotting polygon for {poly_type}: {e}")
            continue

    # 3) Plot doors/windows in special color
    for (polygon_indices, poly_type) in polygons:
        if poly_type not in ['door', 'window']:
            continue
        if len(polygon_indices) == 0:
            continue
        try:
            polygon_indices = np.array(polygon_indices, dtype=int)
            polygon_coords = junctions[polygon_indices]
            polygon_shape = Polygon(polygon_coords)
            if polygon_shape.is_empty:
                continue
            color = 'blue' if poly_type == 'window' else 'red'
            x, y = polygon_shape.exterior.xy
            if poly_type == 'door':
                ax.fill(x, y, color=color, alpha=1)
                ax.plot(x, y, color="red", alpha=1, linewidth=3)
            else:
                ax.fill(x, y, color=color, alpha=1)
                ax.plot(x, y, color=color, alpha=1)
        except Exception as e:
            print(f"Error plotting polygon for {poly_type}: {e}")
            continue

    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    output_path = os.path.join(output_dir, 'floorplan_semantic.png')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()


# -----------------------------------
# Plot the floorplan with walls only
# -----------------------------------
def plot_floorplan_walls_only(annos, polygons, scene_number, output_dir,
                              resolution=0.01, dpi=100):
    """
    Similar approach but we only keep walls in black, everything else in white.
    """
    resolution_mm = resolution * 1000
    fig_width_mm = 80000
    fig_height_mm = 80000
    
    width_pixels = int(fig_width_mm / resolution_mm)
    height_pixels = int(fig_height_mm / resolution_mm)
    fig_width_in = width_pixels / dpi
    fig_height_in = height_pixels / dpi
    
    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=dpi)
    ax.set_xlim(-fig_width_mm / 2, fig_height_mm / 2)
    ax.set_ylim(-fig_height_mm / 2, fig_height_mm / 2)
    ax.set_aspect('equal')
    junctions = np.array([junc['coordinate'][:2] for junc in annos['junctions']])

    # 1) Fill outwall in black
    for (polygon_indices, poly_type) in polygons:
        if poly_type != 'outwall':
            continue
        if len(polygon_indices) == 0:
            continue
        try:
            polygon_indices = np.array(polygon_indices, dtype=int)
            polygon_coords = junctions[polygon_indices]
            polygon_shape = Polygon(polygon_coords)
            if polygon_shape.is_empty:
                continue
            color = 'black'
            x, y = polygon_shape.exterior.xy
            ax.fill(x, y, color=color, alpha=1.0)
        except Exception as e:
            print(f"Error plotting polygon for {poly_type}: {e}")
            continue
    
    # 2) Plot interior walls in black, skip door/window for now
    for (polygon_indices, poly_type) in polygons:
        if poly_type in ['door', 'window', 'outwall']:
            continue
        if len(polygon_indices) == 0:
            continue
        try:
            polygon_indices = np.array(polygon_indices, dtype=int)
            polygon_coords = junctions[polygon_indices]
            polygon_shape = Polygon(polygon_coords)
            if polygon_shape.is_empty:
                continue
            # Outline in black
            x, y = polygon_shape.exterior.xy
            ax.plot(x, y, color='black', alpha=1.0)
        except Exception as e:
            print(f"Error plotting polygon for {poly_type}: {e}")
            continue

    # 3) Fill interior in white
    for (polygon_indices, poly_type) in polygons:
        if poly_type in ['door', 'window', 'outwall']:
            continue
        if len(polygon_indices) == 0:
            continue
        try:
            polygon_indices = np.array(polygon_indices, dtype=int)
            polygon_coords = junctions[polygon_indices]
            polygon_shape = Polygon(polygon_coords)
            if polygon_shape.is_empty:
                continue
            x, y = polygon_shape.exterior.xy
            ax.fill(x, y, color='white', alpha=1.0)
        except Exception as e:
            print(f"Error plotting polygon for {poly_type}: {e}")
            continue

    # 4) Doors in white so they become "passable" 
    for (polygon_indices, poly_type) in polygons:
        if poly_type != 'door':
            continue
        if len(polygon_indices) == 0:
            continue
        try:
            polygon_indices = np.array(polygon_indices, dtype=int)
            polygon_coords = junctions[polygon_indices]
            door_polygon = Polygon(polygon_coords)
            if door_polygon.is_empty:
                continue
            x, y = door_polygon.exterior.xy
            ax.fill(x, y, color='white', alpha=1)
            ax.plot(x, y, color='white', alpha=1, linewidth=5)
        except Exception as e:
            print(f"Error plotting polygon for {poly_type}: {e}")
            continue

    # 5) Outline outwall again 
    for (polygon_indices, poly_type) in polygons:
        if poly_type != 'outwall':
            continue
        if len(polygon_indices) == 0:
            continue
        try:
            polygon_indices = np.array(polygon_indices, dtype=int)
            polygon_coords = junctions[polygon_indices]
            polygon_shape = Polygon(polygon_coords)
            if polygon_shape.is_empty:
                continue
            color = 'black'
            x, y = polygon_shape.exterior.xy
            ax.plot(x, y, color=color, alpha=1.0)
        except Exception as e:
            print(f"Error plotting polygon for {poly_type}: {e}")
            continue

    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    image_path = os.path.join(output_dir, 'floorplan_walls_only.png')
    plt.savefig(image_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()


# -----------------------------------
# Crop the saved floorplan images
# -----------------------------------
def crop_floorplan(image_path):
    with Image.open(image_path) as img:
        # Convert image to grayscale and invert
        img_gray = img.convert("L")
        img_inverted = ImageOps.invert(img_gray)
        original_width, original_height = img.size
        
        # Get bounding box of non-white areas
        bbox = img_inverted.getbbox()
        if bbox:
            img_cropped = img.crop(bbox)
            img_cropped.save(image_path)
            # Calculate how far we've moved from the center
            original_center_x = original_width / 2
            original_center_y = original_height / 2
            bbox_left, bbox_top, _, _ = bbox
            delta_x = bbox_left - original_center_x
            delta_y = bbox_top - original_center_y
            return (delta_x, delta_y)
        else:
            print("No non-white pixels found. Image was not cropped.")
            return (0, 0)

# ----------------------------------------------------------------------
# Main visualization entry: returns offsets + a DICTIONARY for room polygons
# ----------------------------------------------------------------------
def visualize_floorplan(annos, scene_number, base_output_dir,
                        resolution=0.01, dpi=100):
    """
    Generates floorplan_semantic.png and floorplan_walls_only.png,
    crops them, and returns bounding-box offsets plus a dictionary of
    room_type -> [list of shapely Polygons].
    """
    output_dir = os.path.join(base_output_dir, f'scene_{scene_number}')
    os.makedirs(output_dir, exist_ok=True)

    # Collect floor planes
    planes = []
    for semantic in annos['semantics']:
        for planeID in semantic['planeID']:
            if annos['planes'][planeID]['type'] == 'floor':
                planes.append({'planeID': planeID, 'type': semantic['type']})

    # Identify outwall planes
    outwall_planes = []
    for semantic in annos['semantics']:
        if semantic['type'] == 'outwall':
            outwall_planes.extend(semantic['planeID'])

    # Lines that correspond to windows or doors
    lines_holes = []
    for semantic in annos['semantics']:
        if semantic['type'] in ['window', 'door']:
            for planeID in semantic['planeID']:
                lines_holes.extend(
                    np.where(np.array(annos['planeLineMatrix'][planeID]))[0].tolist()
                )
    lines_holes = np.unique(lines_holes)

    junctions = np.array([junc['coordinate'] for junc in annos['junctions']])
    junction_floor = np.where(np.isclose(junctions[:, -1], 0))[0]

    # Build polygons from the floor planes
    polygons = []
    for plane in planes:
        lineIDs = np.where(np.array(annos['planeLineMatrix'][plane['planeID']]))[0].tolist()
        junction_pairs = [
            np.where(np.array(annos['lineJunctionMatrix'][lineID]))[0].tolist()
            for lineID in lineIDs
        ]
        polygon = convert_lines_to_vertices(junction_pairs)
        if polygon:
            polygons.append([polygon[0], plane['type']])

    # Construct outwall polygons
    outerwall_floor = []
    for planeID in outwall_planes:
        lineIDs = np.where(np.array(annos['planeLineMatrix'][planeID]))[0].tolist()
        lineIDs = np.setdiff1d(lineIDs, lines_holes)
        junction_pairs = [
            np.where(np.array(annos['lineJunctionMatrix'][lid]))[0].tolist()
            for lid in lineIDs
        ]
        for start, end in junction_pairs:
            if start in junction_floor and end in junction_floor:
                outerwall_floor.append([start, end])
    outerwall_polygon = convert_lines_to_vertices(outerwall_floor)
    if outerwall_polygon:
        polygons.append([outerwall_polygon[0], 'outwall'])

    # 1) Render the semantic floorplan
    plot_floorplan_semantics(annos, polygons, scene_number, output_dir,
                             resolution, dpi)

    # 2) Render the walls-only floorplan
    # plot_floorplan_walls_only(annos, polygons, scene_number, output_dir,
    #                           resolution, dpi)
    
    # 3) Crop the images, get bounding-box offsets
    semantic_image_path = os.path.join(output_dir, 'floorplan_semantic.png')
    # walls_only_image_path = os.path.join(output_dir, 'floorplan_walls_only.png')
    semantic_bbox = crop_floorplan(semantic_image_path)
    # walls_only_bbox = crop_floorplan(walls_only_image_path)

    # ---------------------------------------------------------------
    # NEW (Room Types) - gather polygons that correspond to room types
    # ---------------------------------------------------------------
    room_type_polygons_dict = {}  # rtype -> list[Polygon]
    for sem in annos['semantics']:
        rtype = sem['type']
        if rtype in room_type_to_id:
            plane_list = sem['planeID']
            # Process each plane separately.
            for pid in plane_list:
                # Get the line IDs for this plane.
                line_ids = np.where(np.array(annos['planeLineMatrix'][pid]))[0].tolist()
                if not line_ids:
                    continue

                # Collect all junctions from these lines.
                all_junctions = set()
                for lid in line_ids:
                    junc_pair = np.where(np.array(annos['lineJunctionMatrix'][lid]))[0].tolist()
                    all_junctions.update(junc_pair)

                # Need at least 3 junctions to form a polygon.
                if len(all_junctions) < 3:
                    continue

                # Convert the set to a list and extract the junction coordinates.
                vertices = list(all_junctions)
                polygon_coords = junctions[np.array(vertices, dtype=int), :2]

                # Order the points by computing the centroid and sorting by angle.
                center = polygon_coords.mean(axis=0)
                angles = np.arctan2(polygon_coords[:, 1] - center[1], polygon_coords[:, 0] - center[0])
                order = np.argsort(angles)
                ordered_coords = polygon_coords[order]

                p = Polygon(ordered_coords)
                if p.is_empty:
                    continue

                if rtype not in room_type_polygons_dict:
                    room_type_polygons_dict[rtype] = []
                room_type_polygons_dict[rtype].append(p)

    return semantic_bbox, semantic_bbox, room_type_polygons_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Structured3D 3D Visualization")
    parser.add_argument("--path", required=False, help="dataset path", metavar="DIR")
    parser.add_argument("--scene", required=False, help="scene id", type=int)
    parser.add_argument("--resolution", default=0.01, type=float, help="Resolution (m/px)")
    parser.add_argument("--dpi", default=100, type=float, help="DPI")
    return parser.parse_args()

def main():
    args = parse_args()
    # Example usage:
    if args.path is None or args.scene is None:
        print("Specify --path and --scene to run directly.")
        return
    anno_path = os.path.join(args.path, f"scene_{args.scene:05d}", "annotation_3d.json")
    with open(anno_path, 'r') as f:
        annos = json.load(f)
    visualize_floorplan(annos, args.scene, 'floorplans', args.resolution, args.dpi)

if __name__ == "__main__":
    main()
