"""
This module contains some common routines and types used by other modules.
"""
import collections
from enum import Enum
from typing import List, NamedTuple, Tuple

import numpy as np
import shapely.geometry
import cv2

# We use OpenCV's type as the underlying 2D image type.
Image = np.ndarray

CHECK_RIGHT_ANGLE_THRESH = 0.1


class Point2D(collections.namedtuple("Point2D", "x y")):
    @classmethod
    def from_tuple(cls, t: Tuple[float, float]):
        return cls._make(t)


# The type of supported polygon/wall/point objects.
class PolygonType(Enum):
    ROOM = "room"
    WINDOW = "window"
    DOOR = "door"
    OPENING = "opening"
    PRIMARY_CAMERA = "primary_camera"
    SECONDARY_CAMERA = "secondary_camera"
    PIN_LABEL = "pin_label"
    PARTIAL_ROOM = "partial_room"


PolygonTypeMapping = {
    "windows": PolygonType.WINDOW,
    "doors": PolygonType.DOOR,
    "openings": PolygonType.OPENING,
}


class Polygon(
    NamedTuple(
        "Polygon", [("type", PolygonType), ("points", List[Point2D]), ("name", str)]
    )
):
    """
    Polygon class that can be used to represent polygons/lines as a list of points, the type and (optional) name
    """

    __slots__ = ()

    def __new__(cls, type, points, name=""):
        return super(Polygon, cls).__new__(cls, type, points, name)

    @staticmethod
    def list_to_points(points: List[Tuple[float, float]]):
        return [Point2D._make(p) for p in points]

    @property
    def to_list(self):
        return [(p.x, p.y) for p in self.points]

    @property
    def num_points(self):
        return len(self.points)

    @property
    def to_shapely_poly(self):
        # Use this function when converting a closed room shape polygon
        return shapely.geometry.polygon.Polygon(self.to_list)

    @property
    def to_shapely_line(self):
        # Use this function when converting W/D/O elements since those are represented as lines.
        return shapely.geometry.LineString(self.to_list)

def rot_verts(verts, rot):
    theta = np.deg2rad(rot)
    R = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], float
    )
    org_shape = verts.shape
    return ((R @ verts.reshape(-1, 2).T).T).reshape(org_shape)

def compute_dot_product(x_prev, y_prev, x_curr, y_curr, x_next, y_next):
    """Compute the oriented angle (in radians) given the camera position and
    two vertices.
    """
    vec_prev = np.array([x_prev - x_curr, y_prev - y_curr])
    vec_prev_norm = np.linalg.norm(vec_prev)
    vec_next = np.array([x_next - x_curr, y_next - y_curr])
    vec_next_norm = np.linalg.norm(vec_next)
    # The function expects non-degenerate case, e.g if one of the line is a point, then this will fail
    return np.dot(vec_prev, vec_next) / (vec_prev_norm * vec_next_norm)


def remove_collinear(room_vertices):
    room_vertices_updated = []
    for idx_curr, vert_curr in enumerate(room_vertices):
        idx_prev = idx_curr - 1
        if idx_prev < 0:
            idx_prev = len(room_vertices) - 1
        idx_next = idx_curr + 1
        if idx_next >= len(room_vertices):
            idx_next = 0
        vert_prev = room_vertices[idx_prev]
        vert_next = room_vertices[idx_next]
        angle = compute_dot_product(
            vert_prev[0],
            vert_prev[1],
            vert_curr[0],
            vert_curr[1],
            vert_next[0],
            vert_next[1],
        )
        if abs(abs(angle) - 1.0) < 1e-3:
            continue
        room_vertices_updated.append([vert_curr[0], vert_curr[1]])
    return np.asarray(room_vertices_updated)

# def pano2persp(img, fov, yaw, pitch, roll, size, RADIUS=128):

#     equ_h, equ_w = img.shape[:2]
#     equ_cx = (equ_w - 1) / 2.0
#     equ_cy = (equ_h - 1) / 2.0

#     height, width = size
#     wFOV = fov
#     hFOV = float(height) / width * wFOV

#     c_x = (width - 1) / 2.0
#     c_y = (height - 1) / 2.0    

#     wangle = (180 - wFOV) / 2.0
#     w_len = 2 * RADIUS * np.sin(np.radians(wFOV / 2.0)) / np.sin(np.radians(wangle))
#     w_interval = w_len / (width - 1)

#     hangle = (180 - hFOV) / 2.0
#     h_len = 2 * RADIUS * np.sin(np.radians(hFOV / 2.0)) / np.sin(np.radians(hangle))
#     h_interval = h_len / (height - 1)
#     x_map = np.zeros([height, width], np.float32) + RADIUS
#     y_map = np.tile((np.arange(0, width) - c_x) * w_interval, [height, 1])
#     z_map = -np.tile((np.arange(0, height) - c_y) * h_interval, [width, 1]).T
#     D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
#     xyz = np.zeros([height, width, 3], float)
#     xyz[:, :, 0] = (RADIUS / D * x_map)[:, :]
#     xyz[:, :, 1] = (RADIUS / D * y_map)[:, :]
#     xyz[:, :, 2] = (RADIUS / D * z_map)[:, :]

#     y_axis = np.array([0.0, 1.0, 0.0], np.float32)
#     z_axis = np.array([0.0, 0.0, 1.0], np.float32)
#     x_axis = np.array([1.0, 0.0, 0.0], np.float32)
#     [R1, _] = cv2.Rodrigues(z_axis * np.radians(yaw - 180))
#     [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-pitch))
#     [R3, _] = cv2.Rodrigues(np.dot(R2 @ R1, x_axis) * np.radians(-roll))

#     xyz = xyz.reshape(height * width, 3).T
#     xyz = ((R3 @ R2 @ R1) @ xyz).T
#     lat = np.arcsin(xyz[:, 2] / RADIUS)
#     lon = np.arctan2(xyz[:, 1], xyz[:, 0])
#     lon = ((lon / np.pi + 1) * equ_cx).reshape(height, width)
#     lat = ((-lat / np.pi * 2 + 1) * equ_cy).reshape(height, width)

#     persp = cv2.remap(
#         img,
#         lon.astype(np.float32),
#         lat.astype(np.float32),
#         cv2.INTER_CUBIC,
#         borderMode=cv2.BORDER_WRAP,
#     )
#     return persp  # , lon, lat, (lon/equ_cx*np.pi)[:width] #[0----pi-----2pi]

def pano2persp(img, fov, yaw, pitch, roll, size, RADIUS=128):

    equ_h, equ_w = img.shape[:2]
    equ_cx = (equ_w - 1) / 2.0
    equ_cy = (equ_h - 1) / 2.0

    height, width = size
    wFOV = fov
    hFOV = float(height) / width * wFOV

    c_x = (width - 1) / 2.0
    c_y = (height - 1) / 2.0

    wangle = (180 - wFOV) / 2.0
    w_len = 2 * RADIUS * np.sin(np.radians(wFOV / 2.0)) / np.sin(np.radians(wangle))
    w_interval = w_len / (width - 1)

    hangle = (180 - hFOV) / 2.0
    h_len = 2 * RADIUS * np.sin(np.radians(hFOV / 2.0)) / np.sin(np.radians(hangle))
    h_interval = h_len / (height - 1)
    x_map = np.zeros([height, width], np.float32) + RADIUS
    y_map = np.tile((np.arange(0, width) - c_x) * w_interval, [height, 1])
    z_map = -np.tile((np.arange(0, height) - c_y) * h_interval, [width, 1]).T
    D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
    xyz = np.zeros([height, width, 3], float)
    xyz[:, :, 0] = (RADIUS / D * x_map)[:, :]
    xyz[:, :, 1] = (RADIUS / D * y_map)[:, :]
    xyz[:, :, 2] = (RADIUS / D * z_map)[:, :]

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    [R1, _] = cv2.Rodrigues(z_axis * np.radians(yaw - 180))
    [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-pitch))
    [R3, _] = cv2.Rodrigues(np.dot(R2 @ R1, x_axis) * np.radians(-roll))

    xyz = xyz.reshape(height * width, 3).T
    xyz = ((R3 @ R2 @ R1) @ xyz).T
    lat = np.arcsin(xyz[:, 2] / RADIUS)
    lon = np.arctan2(xyz[:, 1], xyz[:, 0])
    lon = ((lon / np.pi + 1) * equ_cx).reshape(height, width)
    lat = ((-lat / np.pi * 2 + 1) * equ_cy).reshape(height, width)

    persp = cv2.remap(
        img,
        lon.astype(np.float32),
        lat.astype(np.float32),
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_WRAP,
    )
    return persp  # , lon, lat, (lon/equ_cx*np.pi)[:width] #[0----pi-----2pi]