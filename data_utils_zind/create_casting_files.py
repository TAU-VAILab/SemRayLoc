#!/usr/bin/env python3

import os
import json
import re
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import List, Tuple

# You already have these modules in your environment:
#   from utils.raycast_utils import ray_cast
#   from modules.semantic.semantic_mapper import ObjectType, object_to_color
#   from zind_utils import Polygon, PolygonType, pano2persp, rot_verts
#   from render_floorplans_zind import render_jpg_image
#
# Just be sure they are importable or in your PYTHONPATH.

from utils.raycast_utils import ray_cast
from modules.semantic.semantic_mapper import ObjectType, object_to_color
from zind_utils import Polygon, PolygonType, pano2persp, rot_verts
from render_floorplans_zind import render_jpg_image


DEFAULT_RENDER_RESOLUTION = 2048

def plot_camera_positions_and_rays(camera_positions, img, ray_data, output_path,
                                   resolution=0.01, dpi=100, position_key='semantic'):
    img_height, img_width = img.shape[:2]
    
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


def dataset_to_ray_cast(angle_rad):
    # Transform dataset angle to ray_cast angle by rotating 90 deg minus angle, mod 360
    return (np.radians(90) - angle_rad) % (2 * np.pi)


def create_raycast_file(camera_positions, img_walls, img_semantic, output_path,
                        resolution=0.01, fov_segments=40, epsilon=0.01, depth=15, rot_rad=None):
    resolution_mm_per_pixel = resolution * 1000
    scene_data = {'cameras': []}
    
    ray_n = fov_segments
    F_W = 1 / np.tan(0.698132) / 2  # ~ 1/(2*tan(40 deg))

    with open(camera_positions, "r") as f:
        poses_txt = [line.strip() for line in f.readlines()]
    
    for i, camera_info in enumerate(poses_txt):
        pose_split = camera_info.split()
        pose_vals = np.array([float(s) for s in pose_split], dtype=float)
        x, y, th = pose_vals[0], pose_vals[1], pose_vals[2]
        
        center_angs = np.flip(np.arctan2((np.arange(ray_n) - np.arange(ray_n).mean()),
                                         ray_n * F_W))
        angs = center_angs + rot_rad[i]
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
            # dist, _, hit_coords_walls, _ = ray_cast(img_walls, np.array([x*100, y*100]),
            #                                         ang, dist_max=depth*100, cast_type = 2)
            # _, pred_class, _, normal = ray_cast(img_semantic, np.array([x*100, y*100]),
            #                                     ang, dist_max=depth*100, min_dist=80)
            
            dist, pred_class, hit_coords_walls, normal = ray_cast(img_semantic, np.array([x*100, y*100]), ang, dist_max= 15 * 100, min_dist=5, cast_type=2)

            distance_adjusted = dist  # optionally multiply by cos(...) if desired

            end_x, end_y = hit_coords_walls
            ray_data = {
                'angle': np.rad2deg(ang),
                'distance_m': distance_adjusted * resolution_mm_per_pixel / 1000,
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


def create_additional_files(ray_data, img, output_path):
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

            pf.write("0\n")  # Just as placeholder
            rf.write("0\n")  # Just as placeholder


def extract_floor_room_pano_from_name(name: str):
    match = re.search(r'floor_(-?\d+)_partial_room_(\d+)_pano_(\d+)\.jpg', name)
    if match:
        floor = int(match.group(1))
        room = int(match.group(2))
        pano = int(match.group(3))
        return floor, room, pano
    return None, None, None

def is_polygon_clockwise(lines):
    return np.sum(np.cross(lines[:, 0], lines[:, 1], axis=-1)) > 0


def poly_verts_to_lines(verts):
    n_verts = verts.shape[0]
    if n_verts == 0:
        return
    assert n_verts > 1
    lines = np.stack([verts[:-1], verts[1:]], axis=1)  # N,2,2
    return lines

from collections import Counter

def extract_polygons_from_zind(data: dict, floor_name: str, global_rot: float = 0.0) -> List[Polygon]:
    polygon_list = []
    floor_scale = data["scale_meters_per_coordinate"].get(floor_name, None)
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

        # ROOM
        if "vertices" in layout:
            rv = np.array(layout["vertices"], dtype=float)
            # if len(rv) >= 3:
            rv = np.concatenate([rv, rv[0:1]], axis=0)
            rv = rot_verts(rv, pano_transform["rotation"])
            rv *= pano_transform["scale"]
            rv += np.array(pano_transform["translation"], dtype=float)
            rv *= floor_scale
            if abs(global_rot) > 1e-6:
                rv = rot_verts(rv, global_rot)
                
            # Convert the vertex list into line segments for orientation checking.
            lines = poly_verts_to_lines(rv)
            # Check if the polygon is clockwise using the provided function.
            if not is_polygon_clockwise(lines):
                # If not clockwise, flip the vertex order.
                rv = np.flip(rv, axis=0)
            polygon_list.append(
                Polygon(points=[(pt[0], pt[1]) for pt in rv],
                        type=PolygonType.ROOM,
                        name=f"room_{room_name}")
            )

        # DOORS
        if "doors" in layout and len(layout["doors"]) > 0:
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

        # WINDOWS
        if "windows" in layout and len(layout["windows"]) > 0:
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
    for room_name, partial_rooms in floor_data.items():
        for p_room_key in partial_rooms.keys():
            labels = []
            for pano_key, pano_node in partial_rooms[p_room_key].items():
                label = pano_node.get("label", None)
                if label is not None:
                    labels.append(label)
            if labels:
                majority_label = Counter(labels).most_common(1)[0][0]
            else:
                majority_label = "undefined"
                
            firts_pano_in_p_room = list(partial_rooms[p_room_key].keys())[0]
            pano_node = partial_rooms[p_room_key][firts_pano_in_p_room]
            pano_transform = pano_node["floor_plan_transformation"]
            layout = pano_node.get("layout_raw", {})
            if "vertices" in layout:
                rv = np.array(layout["vertices"], dtype=float)
                # if len(rv) >= 3:
                rv = np.concatenate([rv, rv[0:1]], axis=0)
                rv = rot_verts(rv, pano_transform["rotation"])
                rv *= pano_transform["scale"]
                rv += np.array(pano_transform["translation"], dtype=float)
                rv *= floor_scale
                if abs(global_rot) > 1e-6:
                    rv = rot_verts(rv, global_rot)
                    
                # Convert the vertex list into line segments for orientation checking.
                lines = poly_verts_to_lines(rv)
                # Check if the polygon is clockwise using the provided function.
                if not is_polygon_clockwise(lines):
                    # If not clockwise, flip the vertex order.
                    rv = np.flip(rv, axis=0)
                    
                polygon_list.append(
                    Polygon(points=[(pt[0], pt[1]) for pt in rv],
                            type=PolygonType.PARTIAL_ROOM,
                            name=f"{majority_label}")
                )
        
    return polygon_list

def create_posses_and_copy_images(scene_path: str,
                                  output_path: str,
                                  posses_file_name: str = "",
                                  fov: int = 80,
                                  camera_poses_shifted: List[tuple] = None):
    if camera_poses_shifted is None:
        camera_poses_shifted = []
        
    gt_rots = []
    gt_fovs = []
    original_path= []

    for idx, (x_shifted, y_shifted, rot_deg, img_path) in enumerate(camera_poses_shifted):
        rot_deg = (rot_deg % 360 + 360) % 360
        pano_image_path = os.path.join(scene_path, img_path)
        original_path.append(img_path)
        if not os.path.isfile(pano_image_path):
            print(f"[create_posses_and_copy_images] Missing pano image: {pano_image_path}")
            continue

        # pano_image = cv2.imread(pano_image_path, cv2.IMREAD_COLOR)
        # if pano_image is None:
        #     print(f"[create_posses_and_copy_images] Failed to load: {pano_image_path}")
        #     continue

        # query_image = pano2persp(pano_image, fov, 0, 0, 0, (360, 640))
        # rgb_outdir = os.path.join(output_path, "rgb")
        # os.makedirs(rgb_outdir, exist_ok=True)
        # final_path = os.path.join(rgb_outdir, f"{idx}.png")
        # cv2.imwrite(final_path, query_image)

        gt_rots.append(float(rot_deg))
        gt_fovs.append(float(fov))

    # Write SHIFTED camera poses to poses.txt
    poses_file = os.path.join(output_path, posses_file_name)
    with open(poses_file, "w") as f_pose:
        for i, (x_shifted, y_shifted, rot_deg, _) in enumerate(camera_poses_shifted):
            rot_deg = (rot_deg % 360 + 360) % 360
            rot_rad = np.deg2rad(rot_deg)
            rot_rad = dataset_to_ray_cast(rot_rad)
            f_pose.write(f"{x_shifted} {y_shifted} {rot_rad}\n")

    # Write metadata
    meta = {"gt_rot": gt_rots, "gt_fov": gt_fovs, "original_path":original_path}
    json_out = os.path.join(output_path, "metadata.json")
    with open(json_out, "w") as jf:
        json.dump(meta, jf, indent=4)

    return [np.deg2rad(r) for r in gt_rots]

def shift_polygons_and_cameras(zind_poly_list: List[Polygon],
                               camera_info: List[Tuple[float, float, float]]):
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


def compute_camera_poses_zind(zind_data: dict,
                              floor_name: str,
                              global_rot_deg: float = 0.0):
    camera_info = []
    floor_scale = zind_data["scale_meters_per_coordinate"].get(floor_name, None)
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


def process_scene(scene_id, base_path, output_base_path,
                  resolution=0.01, dpi=100, fov_segments=40, depth=15):
    scene_folder = os.path.join(base_path, str(scene_id))
    zind_json = os.path.join(scene_folder, "zind_data.json")
    if not os.path.isfile(zind_json):
        print(f"[process_scene] Missing {zind_json}, skipping.")
        return

    with open(zind_json, "r") as f:
        data = json.load(f)
    
    scaled_floors = []
    for floor_name, sc in data["scale_meters_per_coordinate"].items():
        if sc is not None:
            scaled_floors.append(floor_name)
    if not scaled_floors:
        print(f"No valid floors in scene {scene_id}")
        return
    
    for floor_name in scaled_floors:
        print(f"[process_scene] Processing floor: {floor_name}")
        global_rot_deg = 0.0
        if "floorplan_to_redraw_transformation" in data:
            if floor_name in data["floorplan_to_redraw_transformation"]:
                global_rot_deg = - data["floorplan_to_redraw_transformation"][floor_name]["rotation"]
        
        zind_poly_list = extract_polygons_from_zind(data, floor_name, global_rot=global_rot_deg)
        if not zind_poly_list:
            print(f"[process_scene] Floor '{floor_name}' found no polygons, skipping.")
            continue
        
        camera_info = compute_camera_poses_zind(data, floor_name, global_rot_deg)
        shifted_polygons, shifted_cameras = shift_polygons_and_cameras(zind_poly_list, camera_info)
        polygon_list_points = [p.points for p in shifted_polygons]

        out_dir = os.path.join(output_base_path, f"scene_{int(scene_id)}_{floor_name}")
        os.makedirs(out_dir, exist_ok=True)

        # 1) Render walls
        walls_png = os.path.join(out_dir, "floorplan_walls_only.png")
        render_jpg_image(
            polygon_list=zind_poly_list,
            polygon_list_points=polygon_list_points,
            jpg_file_name=walls_png,
            rendering_type="wall_only",
            output_path=out_dir,
            floor_scale=1.0
        )

        # # 2) Render semantic
        sem_png = os.path.join(out_dir, "floorplan_semantic.png")
        render_jpg_image(
            polygon_list=zind_poly_list,
            polygon_list_points=polygon_list_points,
            jpg_file_name=sem_png,
            rendering_type="semantic",
            output_path=out_dir,
            floor_scale=1.0
        )

        # 3) Create poses and metadata
        poses_txt = "poses.txt"
        posses_file_path = os.path.join(out_dir, poses_txt)
        rot_rads = create_posses_and_copy_images(
            scene_path=os.path.join(base_path, scene_id),
            output_path=out_dir,
            posses_file_name=poses_txt,
            fov=80,
            camera_poses_shifted=shifted_cameras
        )

        # 4) Raycasting
        img_walls = plt.imread(walls_png)
        img_semantic = plt.imread(sem_png)
        ray_data = create_raycast_file(posses_file_path, img_walls, img_semantic,
                                       out_dir, resolution, fov_segments, depth, rot_rad=rot_rads)
        create_additional_files(ray_data, img_semantic, out_dir)

        # 5) Plot camera positions
        plot_camera_positions_and_rays(posses_file_path, img_semantic, ray_data,
                                       out_dir, resolution, dpi, position_key='semantic')

        create_multilabel_room_map(
            zind_data=data,
            shifted_polygons=shifted_polygons,
            floor_name=floor_name,
            global_rot_deg=global_rot_deg,
            shifted_cameras=shifted_cameras,
            out_dir=out_dir,
            semantic_img_path=sem_png
        )
        
import shapely.geometry as geom
import shapely.geometry as g
import numpy as np
import json
import os
import re
import matplotlib.pyplot as plt
from zind_utils import PolygonType
def create_multilabel_room_map(
    zind_data: dict,
    shifted_polygons: List[Polygon],
    floor_name: str,
    global_rot_deg: float,
    shifted_cameras: List[Tuple[float, float, float, str]],
    out_dir: str,
    semantic_img_path: str,
):
    """
    Using the provided shifted_polygons (which are the PARTIAL_ROOM polygons with their label stored in .name)
    and the shifted_cameras, this function:
      1) Creates a JSON file ("room_types_rectangles.json") that groups polygons by their label.
      2) Creates an overlay image ("floorplan_with_roomlabels.png") on the semantic floorplan showing:
           - Each PARTIAL_ROOM polygon (drawn in magenta with its label in red)
           - Each camera (plotted as a blue dot with its index and label in blue)

    Note: Camera labels are extracted using a mapping from zind_data (by matching the camera's image path).
    """
    import os
    import json
    import matplotlib.pyplot as plt
    from shapely import geometry as geom

    # ----------------------------
    # 1. JSON: Group polygons by label
    # ----------------------------
    # Filter for PARTIAL_ROOM polygons.
    partial_rooms = [p for p in shifted_polygons if p.type == PolygonType.PARTIAL_ROOM]

    room_type_to_polys = {}
    for poly_obj in partial_rooms:
        # Use the polygon's name as the label; if missing, default to "undefined".
        label = poly_obj.name if poly_obj.name is not None else "undefined"
        # Convert polygon points to pixel coordinates (multiply by 100).
        px_coords = [(pt[0] * 100, pt[1] * 100) for pt in poly_obj.points]
        sh_poly = geom.Polygon(px_coords)
        if not sh_poly.is_valid or sh_poly.is_empty:
            print(f"[DEBUG] Partial room polygon invalid for label '{label}', skipping.")
            continue
        # Extract the exterior coordinates.
        coords_xy = list(sh_poly.exterior.coords)
        if coords_xy and coords_xy[0] == coords_xy[-1]:
            coords_xy = coords_xy[:-1]
        # Round for consistency.
        rounded_coords = [[round(x, 5), round(y, 5)] for x, y in coords_xy]
        poly_dict = {"coordinates": rounded_coords}
        room_type_to_polys.setdefault(label, []).append(poly_dict)

    final_json_list = [
        {"room_type": room_type, "polygons": polys}
        for room_type, polys in room_type_to_polys.items()
    ]
    out_json = os.path.join(out_dir, "room_types_rectangles.json")
    with open(out_json, "w") as jf:
        json.dump(final_json_list, jf, indent=4)

    camera_data = []
    for idx, (cx_m, cy_m, _, cam_img_path) in enumerate(shifted_cameras):
        location = (cx_m * 100, cy_m * 100)
        point = geom.Point(location)
        assigned_label = "undefined"
        # Check each PARTIAL_ROOM polygon.
        for poly_obj in partial_rooms:
            px_coords = [(pt[0] * 100, pt[1] * 100) for pt in poly_obj.points]
            sh_poly = geom.Polygon(px_coords)
            if sh_poly.is_valid and not sh_poly.is_empty and sh_poly.contains(point):
                assigned_label = poly_obj.name if poly_obj.name is not None else "undefined"
                break  # Assign the first matching label.
        camera_data.append({
            "index": idx,
            "location": location,
            "assigned_label": assigned_label
        })
        
    room_type_per_image_path = os.path.join(out_dir, "room_type_per_image.txt")
    camera_data_sorted = sorted(camera_data, key=lambda x: x["index"])
    with open(room_type_per_image_path, "w") as f:
        for cam in camera_data_sorted:
            f.write(f"{cam['assigned_label']}\n")

    # ----------------------------
    # 3. Create overlay image on the semantic floorplan
    # ----------------------------
    if not os.path.exists(semantic_img_path):
        print(f"Semantic image {semantic_img_path} not found.")
        return

    base_img = plt.imread(semantic_img_path)
    H, W = base_img.shape[:2]
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.imshow(base_img)

    # Plot each PARTIAL_ROOM polygon.
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

    # Plot cameras on top of the floorplan.
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




# def create_multilabel_room_map(
#     zind_data: dict,
#     shifted_polygons: List[Polygon],
#     floor_name: str,
#     global_rot_deg: float,
#     shifted_cameras: List[Tuple[float, float, float, str]],
#     out_dir: str,
#     semantic_img_path: str,
# ):
#     """
#     1) Gather 'redraw' pinned labels for each 'room_XX' and use them for the polygon labels.
#     2) Compute camera labels based solely on the 'merger' data by extracting the "label" field.
#     3) Create a JSON file (room_polygons.json) with each polygon (SHIFTED) and its redraw labels.
#     4) For each camera, if it is inside a polygon, use the corresponding merger label(s) (from the complete_room_* key)
#        to generate a comma-separated label string in room_type_per_image.txt.
#     5) Create an overlay image on the semantic floorplan showing the polygons and labels.
#     """

#     # Build a mapping from full image_path (from merger) to its label.
#     image_path_to_label = {}
#     floor_dict_merger = zind_data.get("merger", {}).get(floor_name, {})
#     for comp_room_key, partial_rooms_dict in floor_dict_merger.items():
#         for partial_room_key, pano_dict in partial_rooms_dict.items():
#             for pano_key, pano_info in pano_dict.items():
#                 if isinstance(pano_info, dict) and "image_path" in pano_info:
#                     img_path = pano_info["image_path"]
#                     label = pano_info.get("label", None)
#                     if label:
#                         image_path_to_label[img_path] = label

#     # For each camera in shifted_cameras, match the camera's image path (4th element)
#     # to obtain the corresponding merger label.
#     camera_data = []  # Each element: dict with keys: index, location, image_path, merger_label
#     for idx, (cx_m, cy_m, _, cam_img_path) in enumerate(shifted_cameras):
#         pt_px = g.Point(cx_m * 100, cy_m * 100)
#         # Lookup the label using the full image path.
#         merger_label = image_path_to_label.get(cam_img_path, "undefined")
#         camera_data.append({
#             "index": idx,
#             "location": (cx_m * 100, cy_m * 100),
#             "image_path": cam_img_path,
#             "merger_label": merger_label
#         })

    
#     dummy_cams = []
#     # Extract SHIFTED polygons from the standard pipeline (for type=ROOM)
#     # all_polys_floor = extract_polygons_from_zind(zind_data, floor_name, global_rot_deg)

#     # shifted_rooms, _ = shift_polygons_and_cameras(floor_rooms, dummy_cams)
#     shifted_rooms = [p for p in shifted_polygons if p.type == PolygonType.ROOM]

#     threshold = 5
    
#     # Build final SHIFTED polygons using the merger labels of the cameras that fall within each polygon.
#     final_room_data = []
#     for poly_obj in shifted_rooms:
#         # SHIFT to pixel coords (multiply by 100)
#         px_coords = [(pt[0] * 100, pt[1] * 100) for pt in poly_obj.points]
#         sh_poly = geom.Polygon(px_coords)
#         if not sh_poly.is_valid or sh_poly.is_empty:
#             print(f"[DEBUG] SHIFTED polygon invalid for {poly_obj.name}, skipping.")
#             continue

#         # Use a set to store unique camera labels.
#         cam_labels = set()
#         for cam in camera_data:
#             pt = geom.Point(cam["location"])
#             # Check if the point is inside or within the threshold distance from the polygon.
#             if sh_poly.contains(pt) or sh_poly.distance(pt) < threshold:
#                 cam_labels.add(cam["merger_label"])
        
#         # Optionally, if no camera is found, set a default value.
#         if not cam_labels:
#             cam_labels = {"undefined"}

#         final_room_data.append({
#             "poly_px": sh_poly,
#             "labels": sorted(list(cam_labels))  # Now a set of labels.
#         })
#     # --- Create room_polygons.json grouped by room type ---
#     room_type_to_polys = {}  # dict mapping room_type to list of polygons
#     for rd in final_room_data:
#         # Get the full set of exterior coordinates from the polygon.
#         coords_xy = list(rd["poly_px"].exterior.coords)
#         # (Optional) Remove the repeated last coordinate if it’s the same as the first.
#         if coords_xy[0] == coords_xy[-1]:
#             coords_xy = coords_xy[:-1]
#         # Round the coordinates for consistency.
#         rounded_coords = [[round(x, 5), round(y, 5)] for x, y in coords_xy]
        
#         # Create a dictionary representing the polygon.
#         polygon_dict = {"coordinates": rounded_coords}
        
#         # Add this polygon for each label associated with the room.
#         for label in rd["labels"]:
#             room_type_to_polys.setdefault(label, []).append(polygon_dict)

#     # Convert the grouping to a list of dictionaries in the desired format.
#     final_json_list = []
#     for room_type, polys in room_type_to_polys.items():
#         final_json_list.append({
#             "room_type": room_type,
#             "polygons": polys
#         })

#     out_json = os.path.join(out_dir, "room_types_rectangles.json")
#     with open(out_json, "w") as jf:
#         json.dump(final_json_list, jf, indent=4)


#     # --- Create room_type_per_image.txt ---
#     # Write one line per camera (in order of camera index) with its merger label.
#     room_type_per_image_path = os.path.join(out_dir, "room_type_per_image.txt")
#     # Ensure the camera_data is sorted by index.
#     camera_data_sorted = sorted(camera_data, key=lambda x: x["index"])
#     with open(room_type_per_image_path, "w") as f:
#         for cam in camera_data_sorted:
#             f.write(f"{cam['merger_label']}\n")

#     # --- Create an overlay image on the semantic floorplan ---
#     if not os.path.exists(semantic_img_path):
#         return

#     base_img = plt.imread(semantic_img_path)
#     H, W = base_img.shape[:2]
#     fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
#     ax.imshow(base_img)

#     for rd in final_room_data:
#         poly_px = rd["poly_px"]
#         xx, yy = poly_px.exterior.xy
#         ax.plot(xx, yy, color='magenta', linewidth=2)
#         center_pt = poly_px.centroid
#         # show redraw labels joined by slash
#         label_text = "/".join(rd["labels"])
#         ax.text(center_pt.x, center_pt.y, label_text, color='red', fontsize=8)

#     # Plot cameras
#     for cam in camera_data:
#         cx_px, cy_px = cam["location"]
#         ax.plot(cx_px, cy_px, 'bo', markersize=4)
#         label_text = f"{cam['index']}:{cam['merger_label']}"
#         ax.text(cx_px, cy_px - 5, label_text, color='blue', fontsize=8)
        
#     ax.set_xlim([0, W])
#     ax.set_ylim([H, 0])
#     ax.axis('off')
#     out_map = os.path.join(out_dir, "floorplan_with_roomlabels.png")
#     plt.savefig(out_map, bbox_inches='tight', pad_inches=0)
#     plt.close(fig)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process ZInD scene data.')
    parser.add_argument('--scene_id', required=True, help='Scene ID')
    parser.add_argument('--base_path', required=True, help='ZInD dataset path')
    parser.add_argument('--output_base_path', required=True, help='Output path')
    parser.add_argument('--resolution', type=float, default=0.01)
    parser.add_argument('--dpi', type=int, default=100)
    parser.add_argument('--fov_segments', type=int, default=40)
    parser.add_argument('--depth', type=float, default=15.0,
                        help='Max distance (in meters) for ray casting')
    args = parser.parse_args()

    process_scene(args.scene_id,
                  args.base_path,
                  args.output_base_path,
                  resolution=args.resolution,
                  dpi=args.dpi,
                  fov_segments=args.fov_segments,
                  depth=args.depth)
