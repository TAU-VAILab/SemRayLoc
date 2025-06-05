import os
import json
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import argparse
from PIL import Image, ImageOps
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import numpy as np
import os

def convert_lines_to_vertices(lines):
    polygons = []
    lines = np.array(lines)

    polygon = None
    while len(lines) != 0:
        if polygon is None:
            polygon = lines[0].tolist()
            lines = np.delete(lines, 0, 0)

        lineID, juncID = np.where(lines == polygon[-1])
        vertex = lines[lineID[0], 1 - juncID[0]]
        lines = np.delete(lines, lineID, 0)

        if vertex in polygon:
            polygons.append(polygon)
            polygon = None
        else:
            polygon.append(vertex)

    return polygons

def plot_floorplan_semantics(annos, polygons, scene_number, output_dir, resolution=0.005, dpi=1000):
    resolution_mm = resolution * 1000  # Convert to mm/pixel
    
    # Hardcoded figure size: 50x50 meters
    fig_width_mm = 80000  # 50 meters in millimeters
    fig_height_mm = 80000  # 50 meters in millimeters 
    
    # Calculate figure size in pixels
    width_pixels = int(fig_width_mm / resolution_mm)
    height_pixels = int(fig_height_mm / resolution_mm)
    
    # Convert to inches for matplotlib
    fig_width_in = width_pixels / dpi
    fig_height_in = height_pixels / dpi
    
    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=dpi)
    
    # Set the plot limits to ensure the correct scaling
    ax.set_xlim(-fig_width_mm / 2, fig_width_mm / 2)
    ax.set_ylim(-fig_height_mm / 2, fig_height_mm / 2)
    
    # Ensure aspect ratio is equal
    ax.set_aspect('equal')
    
    junctions = np.array([junc['coordinate'][:2] for junc in annos['junctions']])
    # First pass: Fill the "outwall" polygons
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
                            
            color = 'white'
            x, y = polygon_shape.exterior.xy
            ax.fill(x, y, color=color, alpha=1.0)
            
            color = 'black'
            x, y = polygon_shape.exterior.xy
            ax.plot(x, y, color=color, alpha=1.0)


        except Exception as e:
            print(f"Error plotting polygon for {poly_type}: {e}")
            continue

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
            ax.fill(x, y, color=color, alpha=1)
            ax.plot(x, y, color=color, alpha=1)


        except Exception as e:
            print(f"Error plotting polygon for {poly_type}: {e}")
            continue

    # Remove axes and grid
    ax.axis('off')

    # Re-apply limits after all plotting
    ax.set_xlim(-fig_width_mm / 2, fig_width_mm / 2)
    ax.set_ylim(-fig_height_mm / 2, fig_height_mm / 2)

    # Ensure the layout is tight and there's no padding
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    output_path = os.path.join(output_dir, f'floorplan_semantic.png')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()    
            

def plot_floorplan_walls_only(annos, polygons, scene_number, output_dir, resolution=0.01, dpi=100, door_buffer=0.1):
    resolution_mm = resolution * 1000  # Convert to mm/pixel
    
    # Hardcoded figure size: 50x50 meters
    fig_width_mm = 80000  # 50 meters in millimeters
    fig_height_mm = 80000  # 50 meters in millimeters
    
    # Calculate figure size in pixels
    width_pixels = int(fig_width_mm / resolution_mm)
    height_pixels = int(fig_height_mm / resolution_mm)
    
    # Convert to inches for matplotlib
    fig_width_in = width_pixels / dpi
    fig_height_in = height_pixels / dpi
    
    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=dpi)
    
    # Set the plot limits to ensure the correct scaling
    ax.set_xlim(-fig_width_mm / 2, fig_width_mm / 2)
    ax.set_ylim(-fig_height_mm / 2, fig_height_mm / 2)
    
    # Ensure aspect ratio is equal
    ax.set_aspect('equal')

    # Plot floorplan polygons
    junctions = np.array([junc['coordinate'][:2] for junc in annos['junctions']])

    # Store outwall polygons for line intersection checks
    outwall_lines = []

    # First pass: Fill the "outwall" polygons
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
            
            outwall_lines.append(polygon_shape.boundary)  # Store the boundary line of the outer wall
            
            color = 'black'
            x, y = polygon_shape.exterior.xy
            ax.fill(x, y, color=color, alpha=1.0)

        except Exception as e:
            print(f"Error plotting polygon for {poly_type}: {e}")
            continue
        
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
                        
            color = 'white'
            x, y = polygon_shape.exterior.xy
            ax.fill(x, y, color=color, alpha=1.0)
            
        except Exception as e:
            print(f"Error plotting polygon for {poly_type}: {e}")
            continue

    # Check doors and color only those that are near/touching outer walls by using a buffer
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
            
            color = 'white'
            
            x, y = door_polygon.exterior.xy
            ax.fill(x, y, color=color, alpha=1)

        except Exception as e:
            print(f"Error plotting polygon for {poly_type}: {e}")
            continue
        
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

    # Remove axes and grid
    ax.axis('off')

    # Re-apply limits after all plotting
    ax.set_xlim(-fig_width_mm / 2, fig_width_mm / 2)
    ax.set_ylim(-fig_height_mm / 2, fig_height_mm / 2)

    # Ensure the layout is tight and there's no padding
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Save the plot with exact dimensions and no padding
    image_path = os.path.join(output_dir, f'floorplan_walls_only.png')
    plt.savefig(image_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.show()

def visualize_floorplan(annos, scene_number, base_output_dir, resolution=0.01,dpi=100):
    output_dir = os.path.join(base_output_dir, f'scene_{scene_number}')
    os.makedirs(output_dir, exist_ok=True)

    planes = []
    for semantic in annos['semantics']:
        for planeID in semantic['planeID']:
            if annos['planes'][planeID]['type'] == 'floor':
                planes.append({'planeID': planeID, 'type': semantic['type']})

        if semantic['type'] == 'outwall':
            outerwall_planes = semantic['planeID']

    lines_holes = []
    for semantic in annos['semantics']:
        if semantic['type'] in ['window', 'door']:
            for planeID in semantic['planeID']:
                lines_holes.extend(np.where(np.array(annos['planeLineMatrix'][planeID]))[0].tolist())
    lines_holes = np.unique(lines_holes)

    junctions = np.array([junc['coordinate'] for junc in annos['junctions']])
    junction_floor = np.where(np.isclose(junctions[:, -1], 0))[0]

    polygons = []
    for plane in planes:
        lineIDs = np.where(np.array(annos['planeLineMatrix'][plane['planeID']]))[0].tolist()
        junction_pairs = [np.where(np.array(annos['lineJunctionMatrix'][lineID]))[0].tolist() for lineID in lineIDs]
        polygon = convert_lines_to_vertices(junction_pairs)
        polygons.append([polygon[0], plane['type']])

    outerwall_floor = []
    for planeID in outerwall_planes:
        lineIDs = np.where(np.array(annos['planeLineMatrix'][planeID]))[0].tolist()
        lineIDs = np.setdiff1d(lineIDs, lines_holes)
        junction_pairs = [np.where(np.array(annos['lineJunctionMatrix'][lineID]))[0].tolist() for lineID in lineIDs]
        for start, end in junction_pairs:
            if start in junction_floor and end in junction_floor:
                outerwall_floor.append([start, end])

    outerwall_polygon = convert_lines_to_vertices(outerwall_floor)
    polygons.append([outerwall_polygon[0], 'outwall'])

    plot_floorplan_semantics(annos, polygons, scene_number, output_dir, resolution,dpi)
    plot_floorplan_walls_only(annos, polygons, scene_number, output_dir, resolution,dpi)
    
    # Crop the semantic and walls-only floorplan images and return the bounding box coordinates
    semantic_image_path = os.path.join(output_dir, 'floorplan_semantic.png')
    walls_only_image_path = os.path.join(output_dir, 'floorplan_walls_only.png')

    semantic_bbox = crop_floorplan(semantic_image_path)
    walls_only_bbox = crop_floorplan(walls_only_image_path)

    return semantic_bbox, walls_only_bbox

def crop_floorplan(image_path):
    with Image.open(image_path) as img:
        # Convert image to grayscale
        img_gray = img.convert("L")
        
        # Invert the grayscale image
        img_inverted = ImageOps.invert(img_gray)
        
        # Get the original image dimensions
        original_width, original_height = img.size
        
        # Get bounding box of non-white areas
        bbox = img_inverted.getbbox()
        
        if bbox:
            # Crop the image to the bounding box
            img_cropped = img.crop(bbox)
            img_cropped.save(image_path)
            
            # Calculate the center of the original image
            original_center_x = original_width / 2
            original_center_y = original_height / 2
            
            # Calculate the delta of the bottom-left corner of the bounding box from the original center (0,0)
            bbox_left, bbox_top, _, _ = bbox
            delta_x = bbox_left - original_center_x
            delta_y = bbox_top- original_center_y
            return (delta_x, delta_y)  # Return the delta for the bottom-left corner
        else:
            print("No non-white pixels found. Image was not cropped.")
            return (0, 0)


def parse_args():
    parser = argparse.ArgumentParser(description="Structured3D 3D Visualization")
    parser.add_argument("--path", required=True, help="dataset path", metavar="DIR")
    parser.add_argument("--scene", required=True, help="scene id", type=int)
    parser.add_argument("--resolution", default=0.01, type=float, help="Resolution in meters per pixel")
    parser.add_argument("--dpi", default=100, type=float, help="dpi")
    return parser.parse_args()

def main():
    args = parse_args()

    with open(os.path.join(args.path, f"scene_{args.scene:05d}", "annotation_3d.json")) as file:
        annos = json.load(file)
    visualize_floorplan(annos, args.scene, 'floorplans', args.resolution, args.dpi)

if __name__ == "__main__":
    main()
    """
    python visualize_3d.py --path /datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data --scene 3201 --resolution 0.01 --dpi 100
    """

