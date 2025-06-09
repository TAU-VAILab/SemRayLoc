"""
ZInD Dataset Casting Files Creation Module

This module handles the creation of casting files for the ZInD dataset, including
camera positions, ray casting, and room mapping. It processes scene data to generate
visualizations and metadata files.
"""

import os
import json
import re
from typing import List, Tuple, Dict, Optional
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from shapely import geometry as geom

from utils.raycast_utils import ray_cast
from modules.semantic.semantic_mapper import ObjectType, object_to_color
from zind_utils import Polygon, PolygonType, rot_verts
from render_floorplans_zind import render_jpg_image

def plot_camera_positions_and_rays(
    camera_positions: str,
    img: np.ndarray,
    ray_data: Dict,
    output_path: str,
    position_key: str = 'semantic'
) -> None:
    """Plot camera positions and rays on the floorplan.
    
    Args:
        camera_positions: Path to camera positions file
        img: Floorplan image
        ray_data: Ray casting data
        output_path: Output directory path
        position_key: Key for position type ('semantic' or 'walls')
    """
    img_height, img_width = img.shape[:2]
    dpi = 100
    
    # Figure 1: Camera positions and rays
    fig1, ax1 = plt.subplots(figsize=(img_width / dpi, img_height / dpi), dpi=dpi)
    ax1.imshow(img)
    
    for i, camera_data in enumerate(ray_data['cameras']):
        x = camera_data['camera_position_pixel_semantic']['x']
        y = camera_data['camera_position_pixel_semantic']['y']
        
        ax1.plot(x, y, 'bo', markersize=5)
        ax1.text(x, y - 0.04, f"{i}", color='green', fontsize=30, fontweight='bold',
                 ha='center', va='top')
        
        for ray in camera_data['rays']:
            end_x = ray['end_position']['x']
            end_y = ray['end_position']['y']
            object_type = ObjectType(ray['prediction_class'])
            color = object_to_color.get(object_type, 'black')
            ax1.plot([x, end_x], [y, end_y], color=color, lw=0.5)
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax1.axis('equal')
    ax1.axis('off')
    out1 = os.path.join(output_path, f'camera_positions_with_rays_{position_key}.png')
    fig1.savefig(out1, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig1)
    
    # Figure 2: Camera positions with circles
    fig2, ax2 = plt.subplots(figsize=(img_width / dpi, img_height / dpi), dpi=dpi)
    ax2.imshow(img)
    
    for i, camera_data in enumerate(ray_data['cameras']):
        x = camera_data['camera_position_pixel_semantic']['x']
        y = camera_data['camera_position_pixel_semantic']['y']
        
        ax2.plot(x, y, 'bo', markersize=5)
        ax2.text(x, y - 0.04, f"{i}", color='green', fontsize=30, fontweight='bold',
                 ha='center', va='top')
        
        circle = Circle((x, y), radius=80, edgecolor='green', facecolor='none', lw=1)
        ax2.add_patch(circle)
        
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax2.axis('equal')
    ax2.axis('off')
    out2 = os.path.join(output_path, f'camera_positions_with_circles_{position_key}.png')
    fig2.savefig(out2, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig2)

def dataset_to_ray_cast(angle_rad: float) -> float:
    """Transform dataset angle to ray_cast angle.
    
    Args:
        angle_rad: Angle in radians
        
    Returns:
        Transformed angle in radians
    """
    return (np.radians(90) - angle_rad) % (2 * np.pi)

def create_raycast_file(
    camera_positions: str,
    img_walls: np.ndarray,
    img_semantic: np.ndarray,
    output_path: str,
    rot_rad: Optional[List[float]] = None
) -> Dict:
    """Create raycast file for the scene.
    
    Args:
        camera_positions: Path to camera positions file
        img_walls: Walls image
        img_semantic: Semantic image
        output_path: Output directory path
        rot_rad: List of rotation angles in radians
        
    Returns:
        Dictionary containing raycast data
    """
    scene_data = {'cameras': []}
    ray_n = 40  # Number of FOV segments
    F_W = 1 / np.tan(0.698132) / 2  # ~ 1/(2*tan(40 deg))

    with open(camera_positions, "r") as f:
        poses_txt = [line.strip() for line in f.readlines()]
    
    for i, camera_info in enumerate(poses_txt):
        pose_split = camera_info.split()
        pose_vals = np.array([float(s) for s in pose_split], dtype=float)
        x, y, th = pose_vals[0], pose_vals[1], pose_vals[2]
        
        center_angs = np.flip(np.arctan2((np.arange(ray_n) - np.arange(ray_n).mean()),
                                         ray_n * F_W))
        angs = center_angs + (rot_rad[i] if rot_rad else 0)
        angs = dataset_to_ray_cast(angs)[::-1]
        
        cam_data = {
            'camera_number': i,
            'camera_position_m_semantic': {'x': x, 'y': y},
            'camera_position_pixel_semantic': {'x': x*100, 'y': y*100},
            'camera_position_m_walls': {'x': x, 'y': y},
            'camera_position_pixel_walls': {'x': x*100, 'y': y*100},
            'th': th,
            'rays': []
        }

        for j, ang in enumerate(angs):
            dist, pred_class, hit_coords_walls, normal = ray_cast(
                img_semantic, 
                np.array([x*100, y*100]), 
                ang, 
                dist_max=15 * 100, 
                min_dist=5, 
                cast_type=2
            )

            end_x, end_y = hit_coords_walls
            ray_data = {
                'angle': np.rad2deg(ang),
                'distance_m': dist * 0.01,
                'prediction_class': pred_class,
                'start_position_semantic': {'x': x, 'y': y},
                'start_position_walls': {'x': x, 'y': y},
                'end_position': {'x': end_x, 'y': end_y},
                'normal': normal
            }
            cam_data['rays'].append(ray_data)

        scene_data['cameras'].append(cam_data)

    out_file = os.path.join(output_path, 'camera_rays.json')
    with open(out_file, 'w') as f:
        json.dump(scene_data, f, indent=4)
    return scene_data

def create_additional_files(ray_data: Dict, img: np.ndarray, output_path: str) -> None:
    """Create additional files for the scene.
    
    Args:
        ray_data: Ray casting data
        img: Semantic image
        output_path: Output directory path
    """
    depth_file = os.path.join(output_path, 'depth.txt')
    colors_file = os.path.join(output_path, 'semantic.txt')
    pitch_file = os.path.join(output_path, 'pitch.txt')
    roll_file = os.path.join(output_path, 'roll.txt')

    with open(depth_file, 'w') as df, \
         open(colors_file, 'w') as cf, \
         open(pitch_file, 'w') as pf, \
         open(roll_file, 'w') as rf:
        for cam in ray_data['cameras']:
            for ray in cam['rays']:
                df.write(f"{ray['distance_m']} ")
                cf.write(f"{ray['prediction_class']} ")
            df.write('\n')
            cf.write('\n')

            pf.write("0\n")
            rf.write("0\n")

def is_polygon_clockwise(lines: np.ndarray) -> bool:
    """Check if polygon vertices are in clockwise order.
    
    Args:
        lines: Array of line segments
        
    Returns:
        True if clockwise, False otherwise
    """
    return np.sum(np.cross(lines[:, 0], lines[:, 1], axis=-1)) > 0

def poly_verts_to_lines(verts: np.ndarray) -> np.ndarray:
    """Convert polygon vertices to line segments.
    
    Args:
        verts: Array of vertices
        
    Returns:
        Array of line segments
    """
    n_verts = verts.shape[0]
    if n_verts == 0:
        return np.array([])
    assert n_verts > 1
    return np.stack([verts[:-1], verts[1:]], axis=1)

def extract_polygons_from_zind(
    data: Dict,
    floor_name: str,
    global_rot: float = 0.0
) -> List[Polygon]:
    """Extract polygons from ZInD data.
    
    Args:
        data: ZInD data dictionary
        floor_name: Name of the floor
        global_rot: Global rotation angle
        
    Returns:
        List of polygons
    """
    polygon_list = []
    floor_scale = data["scale_meters_per_coordinate"].get(floor_name)
    if floor_scale is None:
        return polygon_list

    floor_data = data["merger"].get(floor_name, {})
    for room_name, partial_rooms in floor_data.items():
        first_partial_room_key = list(partial_rooms.keys())[0]
        proom_node = partial_rooms[first_partial_room_key]
        first_pano_key = list(proom_node.keys())[0]
        pano_node = proom_node[first_pano_key]

        pano_transform = pano_node["floor_plan_transformation"]
        layout = pano_node.get("layout_complete", {})

        # Process room vertices
        if "vertices" in layout:
            rv = np.array(layout["vertices"], dtype=float)
            rv = np.concatenate([rv, rv[0:1]], axis=0)
            rv = rot_verts(rv, pano_transform["rotation"])
            rv *= pano_transform["scale"]
            rv += np.array(pano_transform["translation"], dtype=float)
            rv *= floor_scale
            if abs(global_rot) > 1e-6:
                rv = rot_verts(rv, global_rot)
                
            lines = poly_verts_to_lines(rv)
            if not is_polygon_clockwise(lines):
                rv = np.flip(rv, axis=0)
            polygon_list.append(
                Polygon(points=[(pt[0], pt[1]) for pt in rv],
                        type=PolygonType.ROOM,
                        name=f"room_{room_name}")
            )

        # Process doors
        if "doors" in layout and layout["doors"]:
            dv = np.array(layout["doors"], dtype=float)
            dv = rot_verts(dv, pano_transform["rotation"])
            dv *= pano_transform["scale"]
            dv += np.array(pano_transform["translation"], dtype=float)
            dv *= floor_scale
            if abs(global_rot) > 1e-6:
                dv = rot_verts(dv, global_rot)
            polygon_list.append(
                Polygon(points=[(pt[0], pt[1]) for pt in dv],
                        type=PolygonType.DOOR,
                        name=f"doors_{room_name}")
            )

        # Process windows
        if "windows" in layout and layout["windows"]:
            wv = np.array(layout["windows"], dtype=float)
            wv = rot_verts(wv, pano_transform["rotation"])
            wv *= pano_transform["scale"]
            wv += np.array(pano_transform["translation"], dtype=float)
            wv *= floor_scale
            if abs(global_rot) > 1e-6:
                wv = rot_verts(wv, global_rot)
            polygon_list.append(
                Polygon(points=[(pt[0], pt[1]) for pt in wv],
                        type=PolygonType.WINDOW,
                        name=f"windows_{room_name}")
            )

    # Process partial rooms
    for room_name, partial_rooms in floor_data.items():
        for p_room_key in partial_rooms:
            labels = []
            for pano_key, pano_node in partial_rooms[p_room_key].items():
                label = pano_node.get("label")
                if label is not None:
                    labels.append(label)
            majority_label = Counter(labels).most_common(1)[0][0] if labels else "undefined"
                
            first_pano_in_p_room = list(partial_rooms[p_room_key].keys())[0]
            pano_node = partial_rooms[p_room_key][first_pano_in_p_room]
            pano_transform = pano_node["floor_plan_transformation"]
            layout = pano_node.get("layout_raw", {})
            
            if "vertices" in layout:
                rv = np.array(layout["vertices"], dtype=float)
                rv = np.concatenate([rv, rv[0:1]], axis=0)
                rv = rot_verts(rv, pano_transform["rotation"])
                rv *= pano_transform["scale"]
                rv += np.array(pano_transform["translation"], dtype=float)
                rv *= floor_scale
                if abs(global_rot) > 1e-6:
                    rv = rot_verts(rv, global_rot)
                    
                lines = poly_verts_to_lines(rv)
                if not is_polygon_clockwise(lines):
                    rv = np.flip(rv, axis=0)
                    
                polygon_list.append(
                    Polygon(points=[(pt[0], pt[1]) for pt in rv],
                            type=PolygonType.PARTIAL_ROOM,
                            name=majority_label)
                )
        
    return polygon_list

def create_posses_and_copy_images(
    scene_path: str,
    output_path: str,
    posses_file_name: str = "poses.txt",
    fov: int = 80,
    camera_poses_shifted: List[Tuple[float, float, float, str]] = None
) -> List[float]:
    """Create poses file and copy images.
    
    Args:
        scene_path: Path to scene directory
        output_path: Output directory path
        posses_file_name: Name of poses file
        fov: Field of view in degrees
        camera_poses_shifted: List of camera poses
        
    Returns:
        List of rotation angles in radians
    """
    if camera_poses_shifted is None:
        camera_poses_shifted = []
        
    gt_rots = []
    gt_fovs = []
    original_path = []

    for idx, (x_shifted, y_shifted, rot_deg, img_path) in enumerate(camera_poses_shifted):
        rot_deg = (rot_deg % 360 + 360) % 360
        pano_image_path = os.path.join(scene_path, img_path)
        original_path.append(img_path)
        
        if not os.path.isfile(pano_image_path):
            print(f"[create_posses_and_copy_images] Missing pano image: {pano_image_path}")
            continue

        gt_rots.append(float(rot_deg))
        gt_fovs.append(float(fov))

    # Write camera poses
    poses_file = os.path.join(output_path, posses_file_name)
    with open(poses_file, "w") as f_pose:
        for i, (x_shifted, y_shifted, rot_deg, _) in enumerate(camera_poses_shifted):
            rot_deg = (rot_deg % 360 + 360) % 360
            rot_rad = np.deg2rad(rot_deg)
            rot_rad = dataset_to_ray_cast(rot_rad)
            f_pose.write(f"{x_shifted} {y_shifted} {rot_rad}\n")

    # Write metadata
    meta = {"gt_rot": gt_rots, "gt_fov": gt_fovs, "original_path": original_path}
    json_out = os.path.join(output_path, "metadata.json")
    with open(json_out, "w") as jf:
        json.dump(meta, jf, indent=4)

    return [np.deg2rad(r) for r in gt_rots]

def shift_polygons_and_cameras(
    zind_poly_list: List[Polygon],
    camera_info: List[Tuple[float, float, float, str]]
) -> Tuple[List[Polygon], List[Tuple[float, float, float, str]]]:
    """Shift polygons and cameras to origin.
    
    Args:
        zind_poly_list: List of polygons
        camera_info: List of camera poses
        
    Returns:
        Tuple of (shifted polygons, shifted camera poses)
    """
    if not zind_poly_list and not camera_info:
        return zind_poly_list, camera_info

    all_x = []
    all_y = []

    for poly in zind_poly_list:
        for (px, py) in poly.points:
            all_x.append(px)
            all_y.append(py)

    for (cx, cy, _, _) in camera_info:
        all_x.append(cx)
        all_y.append(cy)

    if not all_x:
        return zind_poly_list, camera_info

    min_x = min(all_x)
    min_y = min(all_y)

    shifted_polygons = []
    for poly in zind_poly_list:
        shifted_pts = []
        for (px, py) in poly.points:
            sx = px - min_x
            sy = py - min_y
            shifted_pts.append((sx, sy))
        new_poly = Polygon(points=shifted_pts, type=poly.type, name=poly.name)
        shifted_polygons.append(new_poly)

    shifted_cameras = []
    for (cx, cy, rot_deg, img_path) in camera_info:
        sx = cx - min_x
        sy = cy - min_y
        shifted_cameras.append((sx, sy, rot_deg, img_path))

    return shifted_polygons, shifted_cameras

def compute_camera_poses_zind(
    zind_data: Dict,
    floor_name: str,
    global_rot_deg: float = 0.0
) -> List[Tuple[float, float, float, str]]:
    """Compute camera poses from ZInD data.
    
    Args:
        zind_data: ZInD data dictionary
        floor_name: Name of the floor
        global_rot_deg: Global rotation angle
        
    Returns:
        List of camera poses
    """
    camera_info = []
    floor_scale = zind_data["scale_meters_per_coordinate"].get(floor_name)
    floor_dict = zind_data["merger"].get(floor_name, {})

    if floor_scale is None:
        return camera_info

    for room_name in floor_dict:
        partial_dict = floor_dict[room_name]
        for partial_room_name in partial_dict:
            pano_dict = partial_dict[partial_room_name]
            for _, pano_node in pano_dict.items():
                img_path = pano_node["image_path"]
                fp_trans = pano_node["floor_plan_transformation"]
                pano_loc = np.array(fp_trans["translation"]) * floor_scale
                pano_loc = rot_verts(pano_loc, global_rot_deg)
                pano_rot_deg = fp_trans["rotation"] + global_rot_deg

                x, y = pano_loc
                camera_info.append((x, y, pano_rot_deg, img_path))

    return camera_info

def process_scene(
    scene_id: str,
    base_path: str,
    output_base_path: str
) -> None:
    """Process a single scene.
    
    Args:
        scene_id: Scene ID
        base_path: Base directory path
        output_base_path: Output directory path
    """
    scene_folder = os.path.join(base_path, str(scene_id))
    zind_json = os.path.join(scene_folder, "zind_data.json")
    if not os.path.isfile(zind_json):
        print(f"[process_scene] Missing {zind_json}, skipping.")
        return

    with open(zind_json, "r") as f:
        data = json.load(f)
    
    scaled_floors = [
        floor_name for floor_name, sc in data["scale_meters_per_coordinate"].items()
        if sc is not None
    ]
    if not scaled_floors:
        print(f"No valid floors in scene {scene_id}")
        return
    
    for floor_name in scaled_floors:
        print(f"[process_scene] Processing floor: {floor_name}")
        global_rot_deg = 0.0
        if "floorplan_to_redraw_transformation" in data:
            if floor_name in data["floorplan_to_redraw_transformation"]:
                global_rot_deg = -data["floorplan_to_redraw_transformation"][floor_name]["rotation"]
        
        zind_poly_list = extract_polygons_from_zind(data, floor_name, global_rot=global_rot_deg)
        if not zind_poly_list:
            print(f"[process_scene] Floor '{floor_name}' found no polygons, skipping.")
            continue
        
        camera_info = compute_camera_poses_zind(data, floor_name, global_rot_deg)
        shifted_polygons, shifted_cameras = shift_polygons_and_cameras(zind_poly_list, camera_info)
        polygon_list_points = [p.points for p in shifted_polygons]

        out_dir = os.path.join(output_base_path, f"scene_{int(scene_id)}_{floor_name}")
        os.makedirs(out_dir, exist_ok=True)

        # Render walls
        walls_png = os.path.join(out_dir, "floorplan_walls_only.png")
        render_jpg_image(
            polygon_list=zind_poly_list,
            polygon_list_points=polygon_list_points,
            jpg_file_name=walls_png,
            rendering_type="wall_only",
            output_path=out_dir,
            floor_scale=1.0
        )

        # Render semantic
        sem_png = os.path.join(out_dir, "floorplan_semantic.png")
        render_jpg_image(
            polygon_list=zind_poly_list,
            polygon_list_points=polygon_list_points,
            jpg_file_name=sem_png,
            rendering_type="semantic",
            output_path=out_dir,
            floor_scale=1.0
        )

        # Create poses and metadata
        poses_txt = "poses.txt"
        posses_file_path = os.path.join(out_dir, poses_txt)
        rot_rads = create_posses_and_copy_images(
            scene_path=os.path.join(base_path, scene_id),
            output_path=out_dir,
            posses_file_name=poses_txt,
            camera_poses_shifted=shifted_cameras
        )

        # Raycasting
        img_walls = plt.imread(walls_png)
        img_semantic = plt.imread(sem_png)
        ray_data = create_raycast_file(
            posses_file_path,
            img_walls,
            img_semantic,
            out_dir,
            rot_rad=rot_rads
        )
        create_additional_files(ray_data, img_semantic, out_dir)

        # Plot camera positions
        plot_camera_positions_and_rays(
            posses_file_path,
            img_semantic,
            ray_data,
            out_dir,
            position_key='semantic'
        )

        create_multilabel_room_map(
            zind_data=data,
            shifted_polygons=shifted_polygons,
            floor_name=floor_name,
            global_rot_deg=global_rot_deg,
            shifted_cameras=shifted_cameras,
            out_dir=out_dir,
            semantic_img_path=sem_png
        )

def create_multilabel_room_map(
    zind_data: Dict,
    shifted_polygons: List[Polygon],
    floor_name: str,
    global_rot_deg: float,
    shifted_cameras: List[Tuple[float, float, float, str]],
    out_dir: str,
    semantic_img_path: str
) -> None:
    """Create multilabel room map.
    
    Args:
        zind_data: ZInD data dictionary
        shifted_polygons: List of shifted polygons
        floor_name: Name of the floor
        global_rot_deg: Global rotation angle
        shifted_cameras: List of shifted camera poses
        out_dir: Output directory path
        semantic_img_path: Path to semantic image
    """
    # Filter for PARTIAL_ROOM polygons
    partial_rooms = [p for p in shifted_polygons if p.type == PolygonType.PARTIAL_ROOM]

    # Group polygons by label
    room_type_to_polys = {}
    for poly_obj in partial_rooms:
        label = poly_obj.name if poly_obj.name is not None else "undefined"
        px_coords = [(pt[0] * 100, pt[1] * 100) for pt in poly_obj.points]
        sh_poly = geom.Polygon(px_coords)
        if not sh_poly.is_valid or sh_poly.is_empty:
            print(f"[DEBUG] Partial room polygon invalid for label '{label}', skipping.")
            continue
        coords_xy = list(sh_poly.exterior.coords)
        if coords_xy and coords_xy[0] == coords_xy[-1]:
            coords_xy = coords_xy[:-1]
        rounded_coords = [[round(x, 5), round(y, 5)] for x, y in coords_xy]
        poly_dict = {"coordinates": rounded_coords}
        room_type_to_polys.setdefault(label, []).append(poly_dict)

    # Write room types JSON
    final_json_list = [
        {"room_type": room_type, "polygons": polys}
        for room_type, polys in room_type_to_polys.items()
    ]
    out_json = os.path.join(out_dir, "room_types_rectangles.json")
    with open(out_json, "w") as jf:
        json.dump(final_json_list, jf, indent=4)

    # Process camera data
    camera_data = []
    for idx, (cx_m, cy_m, _, cam_img_path) in enumerate(shifted_cameras):
        location = (cx_m * 100, cy_m * 100)
        point = geom.Point(location)
        assigned_label = "undefined"
        for poly_obj in partial_rooms:
            px_coords = [(pt[0] * 100, pt[1] * 100) for pt in poly_obj.points]
            sh_poly = geom.Polygon(px_coords)
            if sh_poly.is_valid and not sh_poly.is_empty and sh_poly.contains(point):
                assigned_label = poly_obj.name if poly_obj.name is not None else "undefined"
                break
        camera_data.append({
            "index": idx,
            "location": location,
            "assigned_label": assigned_label
        })
        
    # Write room type per image
    room_type_per_image_path = os.path.join(out_dir, "room_type_per_image.txt")
    camera_data_sorted = sorted(camera_data, key=lambda x: x["index"])
    with open(room_type_per_image_path, "w") as f:
        for cam in camera_data_sorted:
            f.write(f"{cam['assigned_label']}\n")

    # Create overlay image
    if not os.path.exists(semantic_img_path):
        print(f"Semantic image {semantic_img_path} not found.")
        return

    base_img = plt.imread(semantic_img_path)
    H, W = base_img.shape[:2]
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.imshow(base_img)

    # Plot partial room polygons
    for poly_obj in partial_rooms:
        px_coords = [(pt[0] * 100, pt[1] * 100) for pt in poly_obj.points]
        sh_poly = geom.Polygon(px_coords)
        if not sh_poly.is_valid or sh_poly.is_empty:
            continue
        xx, yy = sh_poly.exterior.xy
        ax.plot(xx, yy, color='magenta', linewidth=2)
        center_pt = sh_poly.centroid
        label = poly_obj.name if poly_obj.name is not None else "undefined"
        ax.text(center_pt.x, center_pt.y, label, color='red', fontsize=8)

    # Plot cameras
    for cam in camera_data:
        cx, cy = cam["location"]
        ax.plot(cx, cy, 'bo', markersize=4)
        label_text = f"{cam['index']}:{cam['assigned_label']}"
        ax.text(cx, cy - 5, label_text, color='blue', fontsize=8)

    ax.set_xlim([0, W])
    ax.set_ylim([H, 0])
    ax.axis('off')
    out_map = os.path.join(out_dir, "floorplan_with_roomlabels.png")
    plt.savefig(out_map, bbox_inches='tight', pad_inches=0)
    plt.close(fig)