import os
import json
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from PIL import Image, ImageOps
from modules.semantic.semantic_mapper import room_type_to_id

# Constants for floorplan visualization
DEFAULT_RESOLUTION = 0.01  # meters per pixel
DEFAULT_DPI = 100
DEFAULT_FIGURE_SIZE = 80000  # mm
DEFAULT_SEMANTIC_RESOLUTION = 0.005
DEFAULT_SEMANTIC_DPI = 1000

# Color mappings for different room elements
COLOR_MAPPINGS = {
    'outwall': 'black',
    'door': 'red',
    'window': 'blue',
    'interior': 'white'
}

def convert_lines_to_vertices(lines):
    """Convert line segments into polygon vertices.
    
    Args:
        lines: List of [start_junction, end_junction] pairs
        
    Returns:
        List of polygons, each polygon is a list of junction indices
    """
    import numpy as np
    polygons = []
    lines = np.array(lines, dtype=int)

    polygon = None
    while len(lines) != 0:
        if polygon is None:
            polygon = lines[0].tolist()
            lines = np.delete(lines, 0, axis=0)
            continue

        last_vertex = polygon[-1]
        lineID, juncID = np.where(lines == last_vertex)

        if len(lineID) == 0:
            polygons.append(polygon)
            polygon = None
            continue

        chosen_line_idx = lineID[0]
        chosen_line_col = juncID[0]
        vertex = lines[chosen_line_idx, 1 - chosen_line_col]
        lines = np.delete(lines, chosen_line_idx, axis=0)

        if vertex in polygon:
            polygons.append(polygon)
            polygon = None
        else:
            polygon.append(vertex)

    if polygon is not None:
        polygons.append(polygon)

    return polygons


def setup_plot_figure(resolution, dpi, fig_size=DEFAULT_FIGURE_SIZE):
    """Setup matplotlib figure with specified resolution and size.
    
    Args:
        resolution: Resolution in meters per pixel
        dpi: DPI for output images
        fig_size: Figure size in millimeters
        
    Returns:
        Tuple of (figure, axes, resolution_mm)
    """
    resolution_mm = resolution * 1000
    width_pixels = int(fig_size / resolution_mm)
    height_pixels = int(fig_size / resolution_mm)
    fig_width_in = width_pixels / dpi
    fig_height_in = height_pixels / dpi
    
    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=dpi)
    ax.set_xlim(-fig_size / 2, fig_size / 2)
    ax.set_ylim(-fig_size / 2, fig_size / 2)
    ax.set_aspect('equal')
    
    return fig, ax, resolution_mm


def plot_floorplan_semantics(annos, polygons, scene_number, output_dir, 
                           resolution=DEFAULT_SEMANTIC_RESOLUTION, 
                           dpi=DEFAULT_SEMANTIC_DPI):
    """Plot floorplan with semantic colors for different room types and features."""
    fig, ax, _ = setup_plot_figure(resolution, dpi)
    junctions = np.array([junc['coordinate'][:2] for junc in annos['junctions']])
    
    # Plot outwall polygons
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
            x, y = polygon_shape.exterior.xy
            ax.fill(x, y, color=COLOR_MAPPINGS['outwall'], alpha=1.0)
        except Exception as e:
            print(f"Error plotting polygon for {poly_type}: {e}")
            continue
    
    # Plot interior elements
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
            ax.fill(x, y, color=COLOR_MAPPINGS['interior'], alpha=1.0)
            ax.plot(x, y, color='black', alpha=1.0)
        except Exception as e:
            print(f"Error plotting polygon for {poly_type}: {e}")
            continue

    # Plot doors and windows
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
            x, y = polygon_shape.exterior.xy
            color = COLOR_MAPPINGS[poly_type]
            if poly_type == 'door':
                ax.fill(x, y, color=color, alpha=1)
                ax.plot(x, y, color=color, alpha=1, linewidth=3)
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


def plot_floorplan_walls_only(annos, polygons, scene_number, output_dir, 
                            resolution=DEFAULT_RESOLUTION, 
                            dpi=DEFAULT_DPI):
    """Plot floorplan with only walls visible, everything else in white."""
    fig, ax, _ = setup_plot_figure(resolution, dpi)
    junctions = np.array([junc['coordinate'][:2] for junc in annos['junctions']])

    # Plot outwall
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
            x, y = polygon_shape.exterior.xy
            ax.fill(x, y, color=COLOR_MAPPINGS['outwall'], alpha=1.0)
        except Exception as e:
            print(f"Error plotting polygon for {poly_type}: {e}")
            continue
    
    # Plot interior walls
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
            ax.plot(x, y, color='black', alpha=1.0)
        except Exception as e:
            print(f"Error plotting polygon for {poly_type}: {e}")
            continue

    # Fill interior spaces
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
            ax.fill(x, y, color=COLOR_MAPPINGS['interior'], alpha=1.0)
        except Exception as e:
            print(f"Error plotting polygon for {poly_type}: {e}")
            continue

    # Plot doors
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
            ax.fill(x, y, color=COLOR_MAPPINGS['interior'], alpha=1)
            ax.plot(x, y, color=COLOR_MAPPINGS['interior'], alpha=1, linewidth=5)
        except Exception as e:
            print(f"Error plotting polygon for {poly_type}: {e}")
            continue

    # Outline outwall
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
            x, y = polygon_shape.exterior.xy
            ax.plot(x, y, color=COLOR_MAPPINGS['outwall'], alpha=1.0)
        except Exception as e:
            print(f"Error plotting polygon for {poly_type}: {e}")
            continue

    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    image_path = os.path.join(output_dir, 'floorplan_walls_only.png')
    plt.savefig(image_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()


def crop_floorplan(image_path):
    """Crop floorplan image to remove unnecessary white space."""
    with Image.open(image_path) as img:
        img_gray = img.convert("L")
        img_inverted = ImageOps.invert(img_gray)
        original_width, original_height = img.size
        
        bbox = img_inverted.getbbox()
        if bbox:
            img_cropped = img.crop(bbox)
            img_cropped.save(image_path)
            original_center_x = original_width / 2
            original_center_y = original_height / 2
            bbox_left, bbox_top, _, _ = bbox
            delta_x = bbox_left - original_center_x
            delta_y = bbox_top - original_center_y
            return (delta_x, delta_y)
        else:
            print("No non-white pixels found. Image was not cropped.")
            return (0, 0)


def visualize_floorplan(annos, scene_number, base_output_dir, 
                       resolution=DEFAULT_RESOLUTION, 
                       dpi=DEFAULT_DPI):
    """Generate and save floorplan visualizations.
    
    Args:
        annos: Annotations containing floorplan data
        scene_number: Scene identifier
        base_output_dir: Directory to save outputs
        resolution: Resolution in meters per pixel
        dpi: DPI for output images
        
    Returns:
        Tuple of (semantic_bbox, semantic_bbox, room_type_polygons_dict)
    """
    output_dir = os.path.join(base_output_dir, f'scene_{scene_number}')
    os.makedirs(output_dir, exist_ok=True)

    # Process floor planes
    planes = []
    for semantic in annos['semantics']:
        for planeID in semantic['planeID']:
            if annos['planes'][planeID]['type'] == 'floor':
                planes.append({'planeID': planeID, 'type': semantic['type']})

    # Process outwall planes
    outwall_planes = []
    for semantic in annos['semantics']:
        if semantic['type'] == 'outwall':
            outwall_planes.extend(semantic['planeID'])

    # Process window and door lines
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

    # Build polygons from floor planes
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

    # Build outwall polygons
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

    # Generate visualizations
    plot_floorplan_semantics(annos, polygons, scene_number, output_dir, resolution, dpi)
    semantic_image_path = os.path.join(output_dir, 'floorplan_semantic.png')
    semantic_bbox = crop_floorplan(semantic_image_path)

    # Process room type polygons
    room_type_polygons_dict = {}
    for sem in annos['semantics']:
        rtype = sem['type']
        if rtype in room_type_to_id:
            plane_list = sem['planeID']
            for pid in plane_list:
                line_ids = np.where(np.array(annos['planeLineMatrix'][pid]))[0].tolist()
                if not line_ids:
                    continue

                all_junctions = set()
                for lid in line_ids:
                    junc_pair = np.where(np.array(annos['lineJunctionMatrix'][lid]))[0].tolist()
                    all_junctions.update(junc_pair)

                if len(all_junctions) < 3:
                    continue

                vertices = list(all_junctions)
                polygon_coords = junctions[np.array(vertices, dtype=int), :2]

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
