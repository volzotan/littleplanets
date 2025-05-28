import os
from pathlib import Path

import openexr_numpy

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

import numpy as np
import pyvista as pv

ASSET_DIR = Path("..", "assets2")


def _normalize_vector(v: np.array) -> np.array:
    return v / np.linalg.norm(v)


def _normalize_vectors(v: np.ndarray) -> np.array:
    return v / np.linalg.norm(v, axis=1).reshape(-1, 1)


# def _normalize_vectors_2d(v: np.ndarray) -> np.array:
#     return v / np.linalg.norm(v, axis=2).reshape(-1, -1, 1) # broken


def visualize(centers: np.ndarray, vectors: list[np.ndarray], light_axis: np.array) -> pv.Plotter:
    plotter = pv.Plotter()

    # X, Y, Z axes
    plotter.add_mesh(pv.Spline(np.array([[0, 0, 0], [1, 0, 0]]), 10).tube(radius=0.005), color=[255, 0, 0])
    plotter.add_mesh(pv.Spline(np.array([[0, 0, 0], [0, 1, 0]]), 10).tube(radius=0.005), color=[0, 255, 0])
    plotter.add_mesh(pv.Spline(np.array([[0, 0, 0], [0, 0, 1]]), 10).tube(radius=0.005), color=[0, 0, 255])

    # light axis
    spline = pv.Spline(np.array([[0, 0, 0], light_axis], dtype=np.float32)).tube(radius=0.005)
    plotter.add_mesh(spline, color=[255, 255, 0])

    colors = ["red", "green", "blue"]

    for vi, vector in enumerate(vectors):
        for i in range(len(vector)):
            if np.isnan(np.sum(vector[i])):
                continue
            if np.sum(np.abs(vector[i])) == 0.0:
                continue
            arrow = pv.Arrow(centers[i], vector[i], scale=0.05)
            plotter.add_mesh(arrow, color=colors[vi])

    return plotter


def _line_plane_intersection(
    plane_normal: np.array, plane_point: np.array, line_point: np.array, line_direction: np.array, tol: float = 1e-6
) -> np.array:
    denom = np.dot(plane_normal, line_direction)

    if abs(denom) < tol:  # the line is parallel to the plane
        if abs(np.dot(plane_normal, line_point - plane_point)) < tol:
            return line_point  # the line lies on the plane
        else:
            return line_direction  # no intersection

    t = -np.dot(plane_normal, line_point - plane_point) / denom
    intersection = line_point + t * line_direction

    return intersection


def _compute_intersections(centers: np.ndarray, normals: np.ndarray, axis: np.array, tol=1e-6) -> np.ndarray:
    intersections = np.zeros_like(centers)
    line_point = np.array([0, 0, 0], dtype=np.float32)

    for i in range(centers.shape[0]):
        intersections[i, :] = _line_plane_intersection(normals[i], centers[i], line_point, axis)

    return intersections


# img_normals = openexr_numpy.imread("../assets2/Sphere_Normals.exr", "XYZ")
# img_depth = openexr_numpy.imread("../assets2/Sphere_Depth.exr", "V")
# img_gray = cv2.imread("../assets2/Sphere_Image.tif", cv2.IMREAD_GRAYSCALE)
# img_pxpos = np.load("../assets2/Sphere_Raytracing.npy")

img_normals = openexr_numpy.imread(str(ASSET_DIR / "Moon_Normals.exr"), "XYZ")
img_depth = openexr_numpy.imread(str(ASSET_DIR / "Moon_Depth.exr"), "V")
img_gray = cv2.imread(str(ASSET_DIR / "Moon_Image.tif"), cv2.IMREAD_GRAYSCALE)
img_pxpos = np.load(ASSET_DIR / "Moon_Raytracing.npy")

# Blender coordinate system adjustments:
# Blenders uses a "bottom-left" origin, the numpy's XY
# cartesian system in the image plane uses top-left.
# img_normals[:, :, 1] *= -1

light_axis = _normalize_vector(np.array([0.7, 0.7, 1]))

resize_size = [500, 500]
img_normals = cv2.resize(img_normals, resize_size)
img_depth = cv2.resize(img_depth, resize_size)
img_pxpos = cv2.resize(img_pxpos, resize_size)

centers = []
normals = []

for x in range(img_normals.shape[1]):
    for y in range(img_normals.shape[0]):
        if abs(np.linalg.norm(img_normals[y, x, :])) < 0.01:
            continue

        # centers.append(np.array([
        #     (x / img_normals.shape[1] - 0.5) * 2,
        #     (y / img_normals.shape[0] - 0.5) * 2 * -1, # Y axis coordinate flip: numpy origin top-left, blender bottom-left
        #     img_pxpos[y, x, 2]
        # ]))

        centers.append(img_pxpos[y, x, :])
        normals.append(img_normals[y, x, :])

centers = np.array(centers)
normals = np.array(normals)

intersections = _compute_intersections(centers, normals, light_axis)
directions = _normalize_vectors(intersections - centers)

# flip direction if intersection point is on the opposite end of the axis
# necessary to avoid a flipping of direction signs when moving from one face to the next one
opposite_directions = np.full_like(directions, 1, dtype=np.float32)
opposite_directions[np.dot(intersections, light_axis) < 0] = -1
directions *= opposite_directions

ELEVATION_VECTOR_WEIGHT = 0.6
elevation_vectors = _normalize_vectors(centers)
field_elevation_vectors = []
for i in range(len(elevation_vectors)):
    projected = elevation_vectors[i] - (np.dot(elevation_vectors[i], normals[i])) * normals[i]
    magnitude = np.arccos(np.dot(elevation_vectors[i], normals[i]))
    combined = _normalize_vector(
        directions[i] * (1 - magnitude) * (1 - ELEVATION_VECTOR_WEIGHT)
        + projected * magnitude * ELEVATION_VECTOR_WEIGHT
    )
    field_elevation_vectors.append(combined)

# visualize(centers, [normals, directions], light_axis).show()
# visualize(centers, [directions, field_elevation_vectors], light_axis).show()
# visualize(centers, [directions], light_axis).show()


intersections = np.zeros_like(img_normals)
for x in range(img_normals.shape[1]):
    for y in range(img_normals.shape[0]):
        intersections[y, x, :] = _line_plane_intersection(
            img_normals[y, x], img_pxpos[y, x], np.array([0.0, 0.0, 0.0]), light_axis
        )

img_directions = intersections - img_pxpos

# flip direction if intersection point is on the opposite end of the axis
# necessary to avoid a flipping of direction signs when moving from one face to the next one
opposite_directions = np.full_like(img_directions, 1, dtype=np.float32)
opposite_directions[np.dot(img_directions, light_axis) < 0] = -1
img_directions *= opposite_directions
img_angle = np.atan2(img_directions[:, :, 1], img_directions[:, :, 0])
img_angle = (img_angle + np.pi) / (np.pi * 2.0)
img_angle = (img_angle * 255).astype(np.uint8)

cv2.imwrite("img_angle.png", img_angle)


# centers = img_pxpos.reshape([-1, 3])
# normals = img_normals.reshape([-1, 3])
# directions = img_directions.reshape([-1, 3])
#
# visualize(centers, normals, directions, light_axis).show()
