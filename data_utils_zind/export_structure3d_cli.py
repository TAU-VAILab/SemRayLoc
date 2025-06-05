#!/usr/bin/env python3
"""
Copyright [2021] <Zillow Inc.>

Script to transfer data for the Zillow Indoor Dataset (ZInD) to Structure3D data format

Example usage:
python export_structure3d_cli.py -i <input_folder> -o <output_folder>
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import shapely.geometry
import trimesh
from pyquaternion import Quaternion
from tqdm import tqdm

sys.path.append("../")
import utils
from pano_image import PanoImage
from transformations import Transformation2D

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
LOG = logging.getLogger(__name__)


class ConvertS3D:
    """Class that loads ZInD data formats and transfers to Structure3D data formats"""

    def __init__(self, json_file_name):
        """Create a floor map polygon object from ZinD JSON file."""
        # #print(f"Initializing ConvertS3D with JSON file: {json_file_name}")
        with open(json_file_name) as json_file:
            self._floor_map_json = json.load(json_file)
        self._public_guid = int(str(json_file_name).split("/")[-2])
        self._input_folder = Path(json_file_name).resolve().parent
        self._panos_list = []

        # Global counters to put all local primitives into global IDs
        self.junctions_idx_counter = 0
        self.lines_idx_counter = 0
        self.planes_idx_counter = 0

    def _to_trimesh(self, vertices_list, ceiling_height):
        """Prepare a render scene by setting the geometry and camera (with the specified resolution)"""
        # #print(f"Creating trimesh with vertices: {vertices_list} and ceiling_height: {ceiling_height}")
        num_vertices = len(vertices_list)
        LOG.debug(f"Number of vertices: {num_vertices}")
        
        if num_vertices == 0:
            # #print("No vertices provided for trimesh creation.")
            return None, None

        # Convert vertices_list to a 2D array directly instead of creating a shapely polygon
        vertices_2d = np.array([[p[0], p[1]] for p in vertices_list])

        LOG.debug(f"Vertices being passed to trimesh: {vertices_2d}, type: {type(vertices_2d)}")
        # ##print(f"Vertices array shape: {vertices_2d.shape}")

        try:
            # Create shapely polygon for extrusion
            polygon = shapely.geometry.Polygon(vertices_2d)
            ###print(f"Created shapely polygon: {polygon}")

            # Specify the triangulation engine here. Replace 'earcut' with 'triangle' or 'mapbox_earcut' if preferred.
            room_3d = trimesh.creation.extrude_polygon(polygon, height=ceiling_height, engine='triangle')
            ##print("Extrusion created successfully.")

            room_3d_obj = trimesh.exchange.obj.export_obj(room_3d)
            return room_3d, room_3d_obj
        except Exception as e:
            LOG.error(f"Failed to create extrusion: {e}")
            ##print(f"Exception during extrusion: {e}")
            return None, None


    def to_global_s3d(self, transformation, coordinates):
        """
        Apply transformation on a list of 2D points to transform them from local to global frame of reference.

        :param coordinates: List of 2D coordinates in local frame of reference.

        :return: The transformed list of 2D coordinates.
        """
        ##print(f"Transforming coordinates from local to global using transformation: {transformation}")
        coordinates = coordinates.dot(transformation.rotation_matrix)
        coordinates += transformation.translation / transformation.scale
        ##print(f"Transformed coordinates: {coordinates}")
        return coordinates

    # Compute a canonical pose
    def _canonical_pose(self, partial_room_data, room_vertices_global_all):
        ##print("Computing canonical pose.")
        for pano_id, pano_data in partial_room_data.items():
            if not pano_data["is_primary"]:
                continue
            transformation = Transformation2D.from_zind_data(
                pano_data["floor_plan_transformation"]
            )
            ##print(f"Transformation for pano_id {pano_id}: {transformation}")
            room_vertices_local = np.asarray(pano_data["layout_raw"]["vertices"])
            ##print(f"Local room vertices before removing collinear points: {room_vertices_local}")
            # in case if there is collinear in the data
            room_vertices_local = utils.remove_collinear(room_vertices_local)
            ##print(f"Local room vertices after removing collinear points: {room_vertices_local}")
            num_vertices = len(room_vertices_local)
            if num_vertices == 0:
                ##print("No vertices left after removing collinear points.")
                continue
            room_vertices_global = self.to_global_s3d(
                transformation, room_vertices_local
            )
            ##print(f"Global room vertices: {room_vertices_global}")
            room_vertices_global_all.extend(room_vertices_global)
            return room_vertices_global_all
        ##print("No primary pano data found for canonical pose.")
        return room_vertices_global_all

    def _get_camera_center_global(self, pano_data, canonical_transformation):
        ##print("Getting camera center in global coordinates.")
        rotation_angle = pano_data["floor_plan_transformation"]["rotation"]
        transformation = Transformation2D.from_zind_data(
            pano_data["floor_plan_transformation"]
        )
        camera_center_global = self.to_global_s3d(transformation, np.asarray([[0, 0]]))
        camera_center_global = camera_center_global[0].tolist()
        camera_center_global.append(pano_data["camera_height"])
        camera_center_global = np.asarray(camera_center_global)
        ##print(f"Camera center global: {camera_center_global}")
        # Align the pano texture to the expected Structure3D coordinate system, Z is up and looking at Y
        rot_axis = np.array([0, 1, 0])
        rotation_angle_rad = np.deg2rad(rotation_angle)
        rot = Quaternion(axis=rot_axis, angle=rotation_angle_rad).rotation_matrix
        ##print(f"Rotation matrix for pano: {rot}")
        pano_image = PanoImage.from_file(
            os.path.join(self._input_folder, pano_data["image_path"])
        )
        ##print(f"Pano image loaded: {pano_image}")
        # transfer the image from original to global position
        pano_image = pano_image.rotate_pano(rot)
        ##print("Pano image rotated.")
        return pano_image, camera_center_global

    def _arrange_trimesh(self, pano_data, trimesh_scene):
        ##print("Arranging trimesh.")
        transformation = Transformation2D.from_zind_data(pano_data["floor_plan_transformation"])
        room_vertices_local = np.asarray(pano_data["layout_raw"]["vertices"])
        ##print(f"Local room vertices before removing collinear points: {room_vertices_local}")
        room_vertices_local = utils.remove_collinear(room_vertices_local)
        ##print(f"Local room vertices after removing collinear points: {room_vertices_local}")
        num_vertices = len(room_vertices_local)
        if num_vertices == 0:
            ##print("No vertices left after removing collinear points.")
            return False

        room_vertices_global = self.to_global_s3d(transformation, room_vertices_local)
        room_vertices_global = room_vertices_global.tolist()
        ##print(f"Global room vertices: {room_vertices_global}")

        trimesh_geometry, _ = self._to_trimesh(room_vertices_global, pano_data["ceiling_height"])

        # Check if trimesh_geometry is None due to extrusion failure
        if trimesh_geometry is None:
            LOG.error("Failed to arrange trimesh due to None geometry.")
            ##print("Trimesh geometry is None.")
            return False  # Exit if geometry creation failed

        trimesh_scene.add_geometry(trimesh_geometry)
        ##print("Added geometry to trimesh scene.")
        
        # Proceed with processing facets and boundaries
        facet_edges_dict = defaultdict(list)
        for facet_idx, facet_boundary in enumerate(trimesh_geometry.facets_boundary):
            for facet_edge in facet_boundary:
                facet_edges_dict[tuple(facet_edge)].append(facet_idx)

        facet_vertices_dict = defaultdict(list)
        facet_edges_list = sorted(facet_edges_dict.items(), key=lambda x: x[0])
        for facet_edge_idx, facet_edge in enumerate(facet_edges_list):
            vert0, vert1 = facet_edge[0]
            facet_vertices_dict[vert0].append(facet_edge_idx)
            facet_vertices_dict[vert1].append(facet_edge_idx)
        facet_vertices_list = sorted(facet_vertices_dict.items(), key=lambda x: x[0])
        ##print(f"Facet vertices list: {facet_vertices_list}")
        return trimesh_scene, facet_vertices_list, trimesh_geometry, facet_edges_list

    # Populate the junctions primitives
    def _populate_junctions(
        self, trimesh_geometry, facet_vertices_list, junctions_list,
    ):
        ##print("Populating junctions.")
        trimesh_vertices = trimesh_geometry.vertices
        for facet_vertex in facet_vertices_list:
            vertex_idx = facet_vertex[0]
            junction_dict = {}
            junction_dict["ID"] = vertex_idx + self.junctions_idx_counter
            junction_dict["coordinate"] = list(trimesh_vertices[vertex_idx, :])
            ##print(f"Adding junction: {junction_dict}")
            junctions_list.append(junction_dict)

    # Populate the lines primitives
    def _populate_lines(
        self,
        trimesh_geometry,
        facet_edges_list,
        lines_list,
        junctions_lines_list,
        lines_planes_list,
    ):
        ##print("Populating lines.")
        trimesh_vertices = trimesh_geometry.vertices
        for edge_idx, facet_edge in enumerate(facet_edges_list):
            v0_idx = facet_edge[0][0]
            v1_idx = facet_edge[0][1]
            edge_point = trimesh_vertices[v0_idx, :]
            edge_direction = trimesh_vertices[v1_idx, :] - trimesh_vertices[v0_idx, :]
            line_dict = {}
            edge_idx_global = edge_idx + self.lines_idx_counter
            line_dict["ID"] = edge_idx_global
            line_dict["point"] = list(edge_point)
            line_dict["direction"] = list(edge_direction)
            ##print(f"Adding line: {line_dict}")
            lines_list.append(line_dict)
            v0_idx_global = v0_idx + self.junctions_idx_counter
            v1_idx_global = v1_idx + self.junctions_idx_counter
            junctions_lines_list.append((v0_idx_global, edge_idx_global))
            junctions_lines_list.append((v1_idx_global, edge_idx_global))
            for plane_idx in facet_edge[1]:
                plane_idx_global = plane_idx + self.planes_idx_counter
                lines_planes_list.append((edge_idx_global, plane_idx_global))
        ##print("Lines populated successfully.")

    # Populate the "planes" primitives
    def _populate_planes(self, trimesh_geometry, planes_list):
        ##print("Populating planes.")
        facets_normal = -trimesh_geometry.facets_normal
        facets_origin = trimesh_geometry.facets_origin
        planes_list_idx = []
        for facet_idx, facet_normal in enumerate(facets_normal):
            facet_origin = facets_origin[facet_idx]
            # Solve for D from the plane equation Ax + By + Cz + D = 0
            facet_offset = -np.dot(facet_origin, facet_normal)
            # Compute the plane type by comparing the normal vector with the gravity vector
            angular_dist_to_gravity = np.dot(facet_normal, [0.0, 0.0, 1.0])
            plane_type = "wall"
            if abs(angular_dist_to_gravity + 1.0) < 1e-3:
                plane_type = "ceiling"
            elif abs(angular_dist_to_gravity - 1.0) < 1e-3:
                plane_type = "floor"
            plane_dict = {}
            plane_dict["ID"] = facet_idx + self.planes_idx_counter
            plane_dict["type"] = plane_type
            plane_dict["normal"] = list(facet_normal)
            plane_dict["offset"] = facet_offset
            ##print(f"Adding plane: {plane_dict}")
            planes_list.append(plane_dict)
            planes_list_idx.append(plane_dict["ID"])
        ##print("Planes populated successfully.")
        return planes_list_idx

    def _add_wall_element(
        self,
        wdo_ordered_list,
        wdo_type,
        junctions_list,
        planes_list,
        semantics_list,
        pano_id_int,
        lines_list,
        junctions_lines_list,
        lines_planes_list,
    ):
        """
        Add wall into the lists of junctions, junction lines, plans, lines planes, and semantics.
        These lists are used for final Structure3D json generation
        """

        ##print(f"Adding wall element of type {wdo_type} with ordered list: {wdo_ordered_list}")
        junctions_idx_base_counter = len(junctions_list)
        for wdo_idx, wdo_junction in enumerate(wdo_ordered_list):
            junction_dict = {}
            junction_dict["ID"] = len(junctions_list)
            junction_dict["coordinate"] = wdo_junction.tolist()
            ##print(f"Adding junction for WDO: {junction_dict}")
            junctions_list.append(junction_dict)

        wdo_normal = -np.cross(
            wdo_ordered_list[3] - wdo_ordered_list[0],
            wdo_ordered_list[1] - wdo_ordered_list[0],
        )
        wdo_offset = -np.dot(wdo_normal, wdo_ordered_list[0])
        ##print(f"WDO normal: {wdo_normal}, WDO offset: {wdo_offset}")

        best_dist = -1
        wdo_plane_id = -1
        # find the overall close wall as the plane wall
        for plane_wall in planes_list:
            curr_dist = 0
            if plane_wall["type"] != "wall":
                continue
            plane_norm_factor = np.linalg.norm(plane_wall["normal"])
            for wdo_point in wdo_ordered_list:
                curr_dist += (
                    abs(np.dot(plane_wall["normal"], wdo_point) + plane_wall["offset"])
                    / plane_norm_factor
                )
            if best_dist == -1 or curr_dist < best_dist:
                best_dist = curr_dist
                wdo_plane_id = plane_wall["ID"]
        ##print(f"Selected plane ID for WDO: {wdo_plane_id}")

        plane_dict = {}
        plane_dict["ID"] = len(planes_list)
        plane_dict["type"] = wdo_type
        plane_dict["normal"] = wdo_normal.tolist()
        plane_dict["offset"] = wdo_offset.tolist()
        ##print(f"Adding new plane for WDO: {plane_dict}")
        planes_list.append(plane_dict)

        semantic_dict = {}
        semantic_dict["ID"] = pano_id_int
        semantic_dict["type"] = wdo_type
        semantic_dict["planeID"] = [plane_dict["ID"]]
        ##print(f"Adding semantic: {semantic_dict}")
        semantics_list.append(semantic_dict)

        for idx in range(4):
            idx_next = idx + 1
            if idx_next >= 4:
                idx_next = 0

            line_dict = {}
            line_dict["ID"] = len(lines_list)
            line_dict["point"] = wdo_ordered_list[idx].tolist()
            line_dict["direction"] = (
                wdo_ordered_list[idx_next] - wdo_ordered_list[idx]
            ).tolist()
            ##print(f"Adding line for WDO: {line_dict}")
            lines_list.append(line_dict)
            v0_idx_global = junctions_idx_base_counter + idx
            v1_idx_global = junctions_idx_base_counter + idx_next
            junctions_lines_list.append((v0_idx_global, line_dict["ID"]))
            junctions_lines_list.append((v1_idx_global, line_dict["ID"]))
            lines_planes_list.append((line_dict["ID"], plane_dict["ID"]))
            # If we were able to assign the window to an existing plane, we assign it
            if wdo_plane_id != -1:
                lines_planes_list.append((line_dict["ID"], wdo_plane_id))
                ##print(f"Linking line ID {line_dict['ID']} to existing plane ID {wdo_plane_id}")

        return planes_list

    # Transform windows / doors / openings
    def _transfer_wdo(
        self,
        pano_data,
        junctions_list,
        planes_list,
        semantics_list,
        pano_id_int,
        lines_list,
        junctions_lines_list,
        lines_planes_list,
    ):
        #print("Transferring windows/doors/openings.")
        transformation = Transformation2D.from_zind_data(
            pano_data["floor_plan_transformation"]
        )
        for wdo_type in ["windows", "doors", "openings"]:
            wdo_vertices_local = np.asarray(pano_data["layout_raw"][wdo_type])
            # Skip if there are no elements of this type
            if len(wdo_vertices_local) == 0:
                #print(f"No {wdo_type} found.")
                continue
            #print(f"Processing {wdo_type} with vertices: {wdo_vertices_local}")
            # Transform the local W/D/O vertices to the global frame of reference
            num_wdo = len(wdo_vertices_local) // 3
            wdo_left_right_bound = []
            # save top/down list, note: door down is close to -1 due to camera height
            wdo_top_down_bound = []
            for wdo_idx in range(num_wdo):
                wdo_left_right_bound.extend(
                    wdo_vertices_local[wdo_idx * 3 : wdo_idx * 3 + 2]
                )
                wdo_top_down_bound.extend(
                    wdo_vertices_local[wdo_idx * 3 + 2 : wdo_idx * 3 + 3]
                )
            #print(f"WDO left/right bound: {wdo_left_right_bound}")
            #print(f"WDO top/down bound: {wdo_top_down_bound}")
            wdo_vertices_global = self.to_global_s3d(
                transformation, np.array(wdo_left_right_bound)
            )
            wdo_vertices_global = wdo_vertices_global.tolist()
            #print(f"WDO vertices global: {wdo_vertices_global}")
            for wdo_points in zip(
                wdo_vertices_global[::2], wdo_vertices_global[1::2], wdo_top_down_bound,
            ):
                top_down = wdo_points[2::3]
                top_down = top_down[0]
                wdo_points_0 = wdo_points[0]
                wdo_points_1 = wdo_points[1]
                wdo_points = [wdo_points_0, wdo_points_1]
                y_bottom = top_down[0] + 1  # camera height
                y_top = top_down[1] + 1

                bottom_left = np.asarray([wdo_points[0][0], wdo_points[0][1], y_bottom])
                top_left = np.asarray([wdo_points[0][0], wdo_points[0][1], y_top])
                bottom_right = np.asarray(
                    [wdo_points[1][0], wdo_points[1][1], y_bottom]
                )
                top_right = np.asarray([wdo_points[1][0], wdo_points[1][1], y_top])
                wdo_type_structure3d = "door"
                if wdo_type == "windows":
                    wdo_type_structure3d = "window"

                wdo_ordered_list = [
                    bottom_left,
                    bottom_right,
                    top_right,
                    top_left,
                ]
                #print(f"Ordered WDO list: {wdo_ordered_list}")
                self._add_wall_element(
                    wdo_ordered_list,
                    wdo_type_structure3d,
                    junctions_list,
                    planes_list,
                    semantics_list,
                    pano_id_int,
                    lines_list,
                    junctions_lines_list,
                    lines_planes_list,
                )

    # export to json file
    def _export_json(
        self,
        trimesh_scene,
        junctions_list,
        lines_list,
        planes_list,
        semantics_list,
        lines_planes_list,
        junctions_lines_list,
        output_folder_scene,
    ):
        #print(f"Exporting JSON to folder: {output_folder_scene}")
        self.junctions_idx_counter = len(junctions_list)
        self.lines_idx_counter = len(lines_list)
        self.planes_idx_counter = len(planes_list)

        if not trimesh_scene.is_empty:
            structure3d_dict = {}
            structure3d_dict["junctions"] = junctions_list
            structure3d_dict["lines"] = lines_list
            structure3d_dict["planes"] = planes_list
            structure3d_dict["semantics"] = semantics_list
            plane_line_matrix = np.zeros(
                (self.lines_idx_counter, self.planes_idx_counter), dtype=int
            )
            for line_plane in lines_planes_list:
                line_idx = line_plane[0]
                plane_idx = line_plane[1]
                plane_line_matrix[line_idx][plane_idx] = 1
            structure3d_dict["planeLineMatrix"] = plane_line_matrix.T.tolist()
            #print(f"PlaneLineMatrix: {structure3d_dict['planeLineMatrix']}")
            line_junction_matrix = np.zeros(
                (self.junctions_idx_counter, self.lines_idx_counter), dtype=int,
            )
            for junction_line in junctions_lines_list:
                junction_idx = junction_line[0]
                line_idx = junction_line[1]
                line_junction_matrix[junction_idx][line_idx] = 1
            structure3d_dict["lineJunctionMatrix"] = line_junction_matrix.T.tolist()
            #print(f"LineJunctionMatrix: {structure3d_dict['lineJunctionMatrix']}")
            structure3d_dict["cuboids"] = []
            structure3d_dict["manhattan"] = []

            def convert(o):
                if isinstance(o, np.int64):
                    return int(o)
                raise TypeError

            with open(
                os.path.join(output_folder_scene, "annotation_3d.json"), "w",
            ) as outfile:
                #print("Writing annotation_3d.json")
                json.dump(structure3d_dict, outfile, default=convert)
                #print("annotation_3d.json written successfully.")
        else:
            print("Trimesh scene is empty. Skipping JSON export.")

    def export(self, output_folder: str):
        #print(f"Exporting data to output folder: {output_folder}")
        merger_data = self._floor_map_json["merger"]
        for floor_idx, (floor_id, floor_data) in enumerate(merger_data.items()):
            output_folder_scene = os.path.join(
                output_folder,
                f"scene_{int(self._public_guid):04d}{floor_idx}",
            )
            #print(f"Processing floor {floor_idx}, output folder: {output_folder_scene}")
            os.makedirs(output_folder_scene, exist_ok=True)
            output_folder_2d_rendering = os.path.join(
                output_folder_scene, "2D_rendering"
            )
            os.makedirs(output_folder_2d_rendering, exist_ok=True)
            # Create a list of all the ZInD polygons for this floor: rooms, windows, doors, openings
            trimesh_scene = trimesh.Scene()

            # Global counters to put all local primitives into global IDs
            self.junctions_idx_counter = 0
            self.lines_idx_counter = 0
            self.planes_idx_counter = 0

            # Keep track of the binary relationship
            junctions_lines_list = []
            lines_planes_list = []
            # Prepare the Structure3D data
            junctions_list = []
            lines_list = []
            planes_list = []
            semantics_list = []
            room_vertices_global_all = []

            for complete_room_id, complete_room_data in floor_data.items():
                for (partial_room_id, partial_room_data,) in complete_room_data.items():
                    #print(f"Processing complete_room_id: {complete_room_id}, partial_room_id: {partial_room_id}")
                    # Compute a canonical pose first, that will translate the global coordinates into an axis-oriented BB.
                    room_vertices_global_all = self._canonical_pose(
                        partial_room_data, room_vertices_global_all
                    )
                    #print(f"Room vertices global all after canonical pose: {room_vertices_global_all}")
                    # Make sure we have an axis-aligned final shape, even though that should be the case with the final version of ZInD
                    (canonical_transformation, _) = trimesh.bounds.oriented_bounds_2D(
                        room_vertices_global_all
                    )
                    #print(f"Canonical transformation: {canonical_transformation}")

                    # Inner level data is per-pano (for each floor)
                    for pano_id, pano_data in partial_room_data.items():
                        if not pano_data["is_primary"]:
                            #print(f"Pano {pano_id} is not primary. Skipping.")
                            continue
                        # Get the integer portion of the pano name, which is "pano_{:03d}"
                        pano_id_int = int(pano_id.split("_")[-1])
                        output_folder_room_id = os.path.join(
                            output_folder_2d_rendering, f"{pano_id_int}",
                        )
                        os.makedirs(output_folder_room_id, exist_ok=True)
                        output_folder_pano = os.path.join(
                            output_folder_room_id, "panorama"
                        )
                        os.makedirs(output_folder_pano, exist_ok=True)
                        output_folder_pano_full = os.path.join(
                            output_folder_pano, "full"
                        )
                        os.makedirs(output_folder_pano_full, exist_ok=True)

                        # get_camera_center_global
                        (
                            pano_image,
                            camera_center_global,
                        ) = self._get_camera_center_global(
                            pano_data, canonical_transformation
                        )

                        output_file_name = os.path.join(
                            output_folder_pano_full, "rgb_rawlight.png"
                        )
                        #print(f"Writing pano image to {output_file_name}")
                        pano_image.write_to_file(output_file_name)
                        camera_xyz_file_path = os.path.join(
                            output_folder_pano, "camera_xyz.txt"
                        )
                        #print(f"Saving camera center to {camera_xyz_file_path}")
                        np.savetxt(camera_xyz_file_path, camera_center_global)

                        arrange_result = self._arrange_trimesh(pano_data, trimesh_scene)
                        if arrange_result == False:
                            #print("Arrangement of trimesh failed. Continuing to next pano.")
                            continue
                        else:
                            (
                                trimesh_scene,
                                facet_vertices_list,
                                trimesh_geometry,
                                facet_edges_list,
                            ) = arrange_result

                        # Populate the "junctions" primitives
                        self._populate_junctions(
                            trimesh_geometry, facet_vertices_list, junctions_list,
                        )

                        # Populate the "lines" primitives
                        self._populate_lines(
                            trimesh_geometry,
                            facet_edges_list,
                            lines_list,
                            junctions_lines_list,
                            lines_planes_list,
                        )

                        # Populate the "planes" primitives
                        planes_list_idx = self._populate_planes(
                            trimesh_geometry, planes_list
                        )

                        # All planes of the given pano form a room type.
                        semantic_dict = {}
                        semantic_dict["ID"] = pano_id_int
                        semantic_dict["type"] = "undefined"
                        semantic_dict["planeID"] = planes_list_idx
                        #print(f"Adding semantic for room: {semantic_dict}")
                        semantics_list.append(semantic_dict)

                        # Transform windows / doors / openings
                        self._transfer_wdo(
                            pano_data,
                            junctions_list,
                            planes_list,
                            semantics_list,
                            pano_id_int,
                            lines_list,
                            junctions_lines_list,
                            lines_planes_list,
                        )

                        self.junctions_idx_counter = len(junctions_list)
                        self.lines_idx_counter = len(lines_list)
                        self.planes_idx_counter = len(planes_list)

                    self._export_json(
                        trimesh_scene,
                        junctions_list,
                        lines_list,
                        planes_list,
                        semantics_list,
                        lines_planes_list,
                        junctions_lines_list,
                        output_folder_scene,
                    )


def main():
    parser = argparse.ArgumentParser(
        description="Transform ZinD data format to a sub-set of the Structure3D data format"
    )
    parser.add_argument(
        "--input",
        "-i",
        help="Input JSON file (or folder with ZinD data)",
        required=True,
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output folder where Structure3D format will be saved to",
        required=True,
    )
    args = parser.parse_args()

    # Collect all the feasible input JSON files
    input = args.input
    input_files_list = [input]
    if Path(input).is_dir():
        input_files_list = sorted(Path(input).glob("**/zind_data.json"))
    #print(f"Found {len(input_files_list)} input files.")
    for input_file in tqdm(
        input_files_list,
        desc="Transforming ZinD data format to a sub-set of the Structure3D data format",
    ):
        #print(f"Processing input file: {input_file}")
        zindass3d = ConvertS3D(input_file)
        zindass3d.export(args.output)


if __name__ == "__main__":
    main()
