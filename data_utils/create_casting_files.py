import os
import json
import numpy as np
import matplotlib.pyplot as plt
import shutil
import argparse

from utils.raycast_utils import ray_cast
from modules.semantic.semantic_mapper import ObjectType, object_to_color
from create_floorplans import visualize_floorplan

import math
from shapely.geometry import Point  # for room-type checking
from scipy.spatial.transform import Rotation
from shapely.geometry import Polygon, box, Point

def plot_camera_positions_and_rays(camera_positions, img, ray_data,
                                   output_path, resolution=0.01, dpi=100,
                                   position_key='semantic'):
    """
    Visualize camera positions and rays on top of the floorplan image
    (semantic or walls).
    """
    img_height, img_width = img.shape[:2]
    fig, ax = plt.subplots(figsize=(img_width / dpi, img_height / dpi), dpi=dpi)
    ax.imshow(img)

    # Plot camera positions
    for camera_info in camera_positions:
        if position_key == 'semantic':
            camera_x = camera_info['vx_semantic']
            camera_y = camera_info['vy_semantic']
        elif position_key == 'walls':
            camera_x = camera_info['vx_walls']
            camera_y = camera_info['vy_walls']
        else:
            raise ValueError(f"Invalid position_key: {position_key}.")

        ax.plot(camera_x, camera_y, 'bo', markersize=5)

    # Plot rays
    for camera_data in ray_data['cameras']:
        for ray in camera_data['rays']:
            if position_key == 'semantic':
                start_x = ray['start_position_semantic']['x']
                start_y = ray['start_position_semantic']['y']
            else:  # walls
                start_x = ray['start_position_walls']['x']
                start_y = ray['start_position_walls']['y']

            end_x = ray['end_position']['x']
            end_y = ray['end_position']['y']
            # object type color
            object_type = ObjectType(ray['prediction_class'])
            color = object_to_color.get(object_type, 'black')
            ax.plot([start_x, end_x], [start_y, end_y], color=color, lw=0.5)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.axis('equal')
    ax.axis('off')
    output_image_path = os.path.join(output_path, f'camera_positions_with_rays_{position_key}_closed_doors.png') # for closed doors
    plt.savefig(output_image_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()


def copy_and_rename_images(scene_id, base_path, output_base_path):
    """
    Copies each sub-trajectory's main RGB image to a single folder
    with incrementing names 0.png, 1.png, ...
    """
    int_scene_id = str(int(scene_id))
    scene_base_path = os.path.join(base_path, f'scene_{scene_id.zfill(5)}', '2D_rendering')
    rgb_output_path = os.path.join(output_base_path, f'scene_{int_scene_id}', 'rgb')
    os.makedirs(rgb_output_path, exist_ok=True)
    image_counter = 0

    for traj in sorted(os.listdir(scene_base_path)):
        traj_dir = os.path.join(scene_base_path, traj)
        perspective_dir = os.path.join(traj_dir, 'perspective', 'full')
        if not os.path.isdir(perspective_dir):
            continue

        for sub_traj in sorted(os.listdir(perspective_dir)):
            sub_traj_dir = os.path.join(perspective_dir, sub_traj)
            if not os.path.isdir(sub_traj_dir):
                continue

            # Find the rgb image
            for file in sorted(os.listdir(sub_traj_dir)):
                if file.endswith('.png') and 'rgb_rawlight' in file:
                    source_file = os.path.join(sub_traj_dir, file)
                    dest_file = os.path.join(rgb_output_path, f"{image_counter}.png")
                    shutil.copyfile(source_file, dest_file)
                    image_counter += 1


def create_raycast_file(camera_positions, img_semantic,
                        output_path, resolution=0.01, fov_segments=40,
                        epsilon=0.01):
    """
    Creates camera_rays.json describing each camera's set of rays.
    """
    resolution_mm_per_pixel = resolution * 1000
    scene_data = {'cameras': []}

    ray_n = fov_segments
    # F_W is a field-of-view related constant used in your code
    F_W = 1 / np.tan(0.698132) / 2

    for i, camera_info in enumerate(camera_positions):
        camera_x_walls = camera_info['vx_walls']
        camera_y_walls = camera_info['vy_walls']
        camera_x_semantic = camera_info['vx_semantic']
        camera_y_semantic = camera_info['vy_semantic']

        view_dir = np.array([camera_info['tx'], camera_info['ty'], camera_info['tz']])
        up_dir = np.array([camera_info['ux'], camera_info['uy'], camera_info['uz']])
        right_dir = np.cross(up_dir, view_dir)

        th = np.arctan2(camera_info['ty'], camera_info['tx'])
        view_dir /= np.linalg.norm(view_dir)
        up_dir /= np.linalg.norm(up_dir)
        right_dir /= np.linalg.norm(right_dir)

        center_angs = np.flip(np.arctan2((np.arange(ray_n) - np.arange(ray_n).mean()), ray_n * F_W))
        angs = center_angs + th

        camera_data = {
            'room_id': camera_info['room_id'],
            'camera_number': camera_info['camera_number'],
            'camera_position_m_semantic': {
                'x': camera_x_semantic / 100, 'y': camera_y_semantic / 100
            },
            'camera_position_pixel_semantic': {
                'x': camera_x_semantic, 'y': camera_y_semantic
            },
            'camera_position_m_walls': {
                'x': camera_x_walls / 100, 'y': camera_y_walls / 100
            },
            'camera_position_pixel_walls': {
                'x': camera_x_walls, 'y': camera_y_walls
            },
            'normalize_dir': {
                'view_dir': view_dir.tolist(),
                'up_dir': up_dir.tolist(),
                'right_dir': right_dir.tolist()
            },
            'camera_full_info': camera_info,
            'th': np.rad2deg(th),
            'rays': []
        }

        for j, ang in enumerate(angs):
            dist, pred_class, hit_coord, _ = ray_cast(img_semantic, np.array([camera_x_semantic, camera_y_semantic]),
                                                ang, dist_max=15*100, min_dist=5,  cast_type = 2) 
            hit_coords_walls = hit_coord
            hit_coords_semantic = hit_coord 
            distance_adjusted = dist * np.cos(center_angs[j])
            end_x, end_y = hit_coords_walls
            end_x_semantic, end_y_semantic = hit_coords_semantic
            

            ray_data = {
                'angle': np.rad2deg(ang),
                'distance_m': distance_adjusted * resolution_mm_per_pixel / 1000,
                'prediction_class': pred_class,
                'start_position_semantic': {'x': camera_x_semantic, 'y': camera_y_semantic},
                'start_position_walls': {'x': camera_x_walls, 'y': camera_y_walls},
                'end_position': {'x': end_x, 'y': end_y},
                'end_position_semantic': {'x': end_x_semantic, 'y': end_y_semantic},
                'normal': None
            }
            camera_data['rays'].append(ray_data)

        scene_data['cameras'].append(camera_data)

    output_file = os.path.join(output_path, 'camera_rays.json')
    # output_file = os.path.join(output_path, 'camera_rays_closed_doors.json')
    with open(output_file, 'w') as f:
        json.dump(scene_data, f, indent=4)

    return scene_data


def create_additional_files(ray_data, img, output_path):
    """
    Create depth.txt, poses.txt, and semantic.txt for each camera row.
    """
    # depth_file = os.path.join(output_path, 'depth.txt')
    depth_file = os.path.join(output_path, 'depth_closed_doors.txt') #for closed doors
    poses_file = os.path.join(output_path, 'poses.txt')
    colors_file = os.path.join(output_path, 'semantic.txt')

    # with open(depth_file, 'w') as df, open(poses_file, 'w') as pf, open(colors_file, 'w') as cf:
    with open(depth_file, 'w') as df:
        for camera in ray_data['cameras']:
            # Depth / semantic
            for ray in camera['rays']:
                df.write(f"{ray['distance_m']} ")
                # cf.write(f"{ray['prediction_class']} ")
            df.write('\n')
            # cf.write('\n')

            # yaw_rad = np.deg2rad(camera['th'])
            # pf.write(f"{camera['camera_position_m_semantic']['x']} "
            #          f"{camera['camera_position_m_semantic']['y']} "
            #          f"{yaw_rad}\n")


def create_pitch_roll_files_from_json(output_path):
    """
    From camera_rays.json, parse the orientation vectors to compute
    pitch, roll, etc. Then store them in pitch.txt, roll.txt
    """
    camera_rays_path = os.path.join(output_path, 'camera_rays.json')
    with open(camera_rays_path, 'r') as f:
        ray_data = json.load(f)

    pitch_file = os.path.join(output_path, 'pitch.txt')
    roll_file = os.path.join(output_path, 'roll.txt')

    with open(pitch_file, 'w') as pf, open(roll_file, 'w') as rf:
        for camera in ray_data['cameras']:
            tx = float(camera['camera_full_info']['tx'])
            ty = float(camera['camera_full_info']['ty'])
            tz = float(camera['camera_full_info']['tz'])
            ux = float(camera['camera_full_info']['ux'])
            uy = float(camera['camera_full_info']['uy'])
            uz = float(camera['camera_full_info']['uz'])

            t = np.array([tx, ty, tz]); t /= np.linalg.norm(t)
            u = np.array([ux, uy, uz]); u /= np.linalg.norm(u)
            w = np.cross(u, t)
            u = np.cross(t, w)
            R = np.stack([t, w, u], axis=1)
            r = Rotation.from_matrix(R)
            theta, pitch, roll = r.as_euler('ZYX')
            pf.write(f"{pitch}\n")
            rf.write(f"{roll}\n")


# ---------------------------------------------------------------------
# NEW (Room Types) - We now expect a DICTIONARY: {rtype: [list_of_polygons], ...}
# ---------------------------------------------------------------------
def create_roomtype_data(camera_positions, room_type_polygons,  # dictionary
                         semantic_bbox, output_path, resolution=0.01):
    """
    1. For each recognized room type, we keep ALL polygons in a list.
       We save each polygon's bounding box in room_types_rectangles.json.
    2. We plot all bounding boxes on top of floorplan_semantic.png =>
       floorplan_with_roomtypes.png
    3. For each camera, if it is inside ANY polygon for that room type,
       we assign that room type. Otherwise "undefined".
    """
    dx, dy = semantic_bbox
    shifted_polygons_dict = {}
    # Shift all polygons
    for rtype, polygons in room_type_polygons.items():
        shifted_list = []
        for poly in polygons:
            shifted = Polygon([
                (p[0]/10 - dx, -p[1]/10 - dy)
                for p in np.array(poly.exterior.coords)
            ])
            shifted_list.append(shifted)
        shifted_polygons_dict[rtype] = shifted_list

    # Build room_types_data with duplicates removed
    room_types_data = []
    for rtype, poly_list in shifted_polygons_dict.items():
        seen_bboxes = set()
        poly_boxes = []
        for poly in poly_list:
            if not poly.is_empty:
                minx, miny, maxx, maxy = poly.bounds
                # Round or not, depending on your need
                bbox_tuple = (
                    round(minx, 5),
                    round(miny, 5),
                    round(maxx, 5),
                    round(maxy, 5)
                )
                if bbox_tuple not in seen_bboxes:
                    seen_bboxes.add(bbox_tuple)
                    poly_boxes.append({
                        "min_x": bbox_tuple[0],
                        "min_y": bbox_tuple[1],
                        "max_x": bbox_tuple[2],
                        "max_y": bbox_tuple[3]
                    })
        room_types_data.append({
            "room_type": rtype,
            "polygons": poly_boxes
        })

    # Save JSON
    with open(os.path.join(output_path, "room_types_rectangles.json"), 'w') as f:
        json.dump(room_types_data, f, indent=4)

    # Plot bounding boxes on top of the floorplan_semantic.png
    floorplan_path = os.path.join(output_path, "floorplan_semantic.png")
    if os.path.exists(floorplan_path):
        img = plt.imread(floorplan_path)
        fig, ax = plt.subplots(figsize=(img.shape[1]/100, img.shape[0]/100), dpi=100)
        ax.imshow(img)

        # Draw each bounding box with a label
        for item in room_types_data:
            rtype = item['room_type']
            for bbox in item['polygons']:
                x1, y1 = bbox["min_x"], bbox["min_y"]
                x2, y2 = bbox["max_x"], bbox["max_y"]
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     fill=False, edgecolor='magenta', linewidth=1.5)
                ax.add_patch(rect)
                ax.text(x1, y1, rtype, color='magenta', fontsize=8)

        # Plot camera positions
        for i, cam_info in enumerate(camera_positions):
            cam_x = cam_info['vx_semantic']
            cam_y = cam_info['vy_semantic']
            ax.plot(cam_x, cam_y, 'ro', markersize=3)
            ax.text(cam_x, cam_y, str(i), color='green', fontsize=10)

        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)
        ax.axis('off')
        out_fig_path = os.path.join(output_path, "floorplan_with_roomtypes.png")
        plt.savefig(out_fig_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    # 1) Define a small priority map for your special ordering
    priority_map = {
        "bedroom": 9,
        "living room": 10
        # anything not in this map will default to a large value = checked last
    }

    def get_priority(rtype):
        return priority_map.get(rtype, 0)  # fallback priority

    # 2) Sort room_types_data by ascending priority
    #    so that 'bathroom' appears first, then 'bedroom', then 'living room', etc.
    room_types_data_sorted = sorted(
        room_types_data,
        key=lambda item: get_priority(item["room_type"])
    )
    
    roomtype_per_camera = []
    for i, cam_info in enumerate(camera_positions):
        cam_x = cam_info['vx_semantic']
        cam_y = cam_info['vy_semantic']
        assigned = "undefined"

        # Loop over each room type and its list of bounding boxes
        for item in room_types_data_sorted:  
            rtype = item['room_type']
            for bbox in item['polygons']:
                # Check if camera position is within [min_x, max_x] x [min_y, max_y]
                if (bbox['min_x'] <= cam_x <= bbox['max_x'] and
                    bbox['min_y'] <= cam_y <= bbox['max_y']):
                    assigned = rtype
                    break

            # If assigned, exit the rtype loop
            if assigned != "undefined":
                break

        roomtype_per_camera.append(assigned)

    # Write out file
    room_file = os.path.join(output_path, "room_type_per_image.txt")
    with open(room_file, 'w') as f:
        for rtype in roomtype_per_camera:
            f.write(f"{rtype}\n")


def process_scene(scene_id, base_path, output_base_path,
                  resolution=0.01, dpi=100, fov_segments=40):
    int_scene_id = str(int(scene_id))
    
    # 1) Load annotation and produce floorplans
    annotation_file = os.path.join(base_path,
                                   f'scene_{scene_id.zfill(5)}',
                                   'annotation_3d.json')
    with open(annotation_file, 'r') as file:
        annos = json.load(file)

    # We'll get the bounding box offsets and the polygons (dictionary) for each valid room type
    semantic_bbox, walls_only_bbox, room_type_polygons = visualize_floorplan(
        annos, int_scene_id, output_base_path, resolution, dpi
    )

    camera_positions = []
    scene_base_path = os.path.join(base_path, f'scene_{scene_id.zfill(5)}', '2D_rendering')

    # 2) Collect camera poses
    for traj in sorted(os.listdir(scene_base_path)):
        traj_dir = os.path.join(scene_base_path, traj)
        pers_dir = os.path.join(traj_dir, 'perspective', 'full')
        if not os.path.isdir(pers_dir):
            continue

        for sub_traj in sorted(os.listdir(pers_dir)):
            sub_traj_dir = os.path.join(pers_dir, sub_traj)
            if not os.path.isdir(sub_traj_dir):
                continue

            camera_pose_file = os.path.join(sub_traj_dir, 'camera_pose.txt')
            if not os.path.exists(camera_pose_file):
                continue

            room_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(traj_dir))))
            position_id = os.path.basename(sub_traj_dir)
            resolution_mm_per_pixel = resolution * 1000

            with open(camera_pose_file, 'r') as file:
                for line in file:
                    try:
                        (vx_raw, vy_raw, vz,
                         tx, ty, tz,
                         ux, uy, uz,
                         xfov, yfov, _) = map(float, line.strip().split())

                        vx_semantic = vx_raw/resolution_mm_per_pixel - semantic_bbox[0]
                        vy_semantic = -vy_raw/resolution_mm_per_pixel - semantic_bbox[1]

                        vx_walls = vx_raw/resolution_mm_per_pixel - walls_only_bbox[0]
                        vy_walls = -vy_raw/resolution_mm_per_pixel - walls_only_bbox[1]

                        camera_positions.append({
                            'room_id': room_id,
                            'camera_number': position_id,
                            'vx_semantic': vx_semantic,
                            'vy_semantic': vy_semantic,
                            'vx_walls': vx_walls,
                            'vy_walls': vy_walls,
                            'vz': vz,
                            'tx': tx, 'ty': ty, 'tz': tz,
                            'ux': ux, 'uy': uy, 'uz': uz,
                            'xfov': xfov
                        })
                    except ValueError as e:
                        print(f"Error parsing line: {line}")
                        print(f"Error details: {e}")

    # if not camera_positions:
    #     print("No camera positions were found.")
    #     return

    # # 3) Load floorplan images
    # floorplan_walls_path = os.path.join(output_base_path,
    #                                     f'scene_{int_scene_id}',
    #                                     'floorplan_walls_only.png')
    floorplan_semantic_path = os.path.join(output_base_path,
                                           f'scene_{int_scene_id}',
                                           'floorplan_semantic.png')
    # if not os.path.exists(floorplan_walls_path) or not os.path.exists(floorplan_semantic_path):
    #     print("Floorplan images missing; cannot proceed.")
    #     return

    # img_walls = plt.imread(floorplan_walls_path)
    img_semantic = plt.imread(floorplan_semantic_path)

    # 4) Create raycast data
    ray_data = create_raycast_file(camera_positions,  img_semantic,
                                   os.path.join(output_base_path, f'scene_{int_scene_id}'),
                                   resolution, fov_segments)

    # 5) Create depth, poses, semantic
    # create_additional_files(ray_data, img_semantic,
    #                         os.path.join(output_base_path, f'scene_{int_scene_id}'))

    # # 6) Create pitch & roll
    # create_pitch_roll_files_from_json(os.path.join(output_base_path, f'scene_{int_scene_id}'))

    # 7) Plot camera positions
    # plot_camera_positions_and_rays(camera_positions, img_semantic, ray_data,
    #                                os.path.join(output_base_path, f'scene_{int_scene_id}'),
    #                                resolution, dpi, 'semantic')

    # # 8) Copy/rename rawlight images
    # copy_and_rename_images(scene_id, base_path, output_base_path)

    # # -------------------------------------------------------------------
    # # NEW (Room Types) - create the additional 3 files
    # # -------------------------------------------------------------------
    # create_roomtype_data(
    #     camera_positions,            # in the same order as images
    #     room_type_polygons,          # dictionary from visualize_floorplan
    #     semantic_bbox,               # needed to shift polygons & camera
    #     os.path.join(output_base_path, f'scene_{int_scene_id}'),
    #     resolution
    # )


def process_all_scenes(base_path, output_base_path,
                       resolution=0.01, dpi=100, fov_segments=40):
    """
    Example top-level driver that processes multiple scenes in a batch.
    Adjust 'scenes_to_process' as needed.
    """
    # If you want to process all scenes:
    # scenes = [d for d in os.listdir(base_path)
    #           if os.path.isdir(os.path.join(base_path, d)) and d.startswith('scene_')]
    # scenes.sort(key=lambda x: int(x.split('_')[-1]))
    # ...
    scenes_to_process = ['scene_00000']  # Change to your set
    scenes = [d for d in scenes_to_process if os.path.isdir(os.path.join(base_path, d))]
    total_scenes = len(scenes)
    print(f"Total scenes to process: {total_scenes}\n")
    failed_scenes = []
    
    for idx, scene_dir in enumerate(scenes, start=1):
        scene_id = str(int(scene_dir.split('_')[-1]))
        try:
            print(f"Processing Scene {scene_id} ({idx}/{total_scenes})...")
            process_scene(scene_id, base_path, output_base_path,
                          resolution, dpi, fov_segments)
            print(f"Finished processing Scene {scene_id}.\n")
        except Exception as e:
            print(f"Failed to process Scene {scene_id}. Error: {e}\n")
            failed_scenes.append(scene_id)
    
    if failed_scenes:
        print("The following scenes failed to process:")
        for scene in failed_scenes:
            print(f"Scene {scene}")


def main():
    parser = argparse.ArgumentParser(description='Process scene data.')
    parser.add_argument('--scene_id', type=str, required=False, help='Scene ID to process.')
    parser.add_argument('--base_path', type=str, required=True, help='Base path to the scene data.')
    parser.add_argument('--output_base_path', type=str, required=True, help='Output base path for processed data.')
    parser.add_argument('--resolution', type=float, default=0.01, help='Resolution in meters per pixel.')
    parser.add_argument('--dpi', type=int, default=100, help='DPI for image processing.')
    parser.add_argument('--fov_segments', type=int, default=40, help='Number of segments for FOV rays.')

    args = parser.parse_args()

    if args.scene_id is not None:
        # Process only one scene
        process_scene(args.scene_id, args.base_path, args.output_base_path,
                      args.resolution, args.dpi, args.fov_segments)
    else:
        # Process multiple scenes
        process_all_scenes(args.base_path, args.output_base_path,
                           args.resolution, args.dpi, args.fov_segments)

if __name__ == "__main__":
    main()
