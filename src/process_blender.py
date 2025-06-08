import os
from pathlib import Path

import numpy as np
import pyvista as pv
import openexr_numpy
from loguru import logger
import cv2

# ASSET_DIR = Path("..", "assets_lowres")
ASSET_DIR = Path("..", "assets")
OUTPUT_DIR = Path("..", "output")

RESIZE = True
RESIZE_SIZE = [25, 25]
RESIZE_SIZE = [2000, 2000]

CROSS_FLOW = True

VISUALIZE = False
EXPORT = True

LIGHT_POS = [1.5, 0.3, 2]  # coordinate system: Y-UP
LIGHT_POS = [0.2, 0.2, 1]  # coordinate system: Y-UP


def _normalize_vector(v: np.array) -> np.array:
    return v / np.linalg.norm(v)


def _normalize_vectors(v: np.ndarray) -> np.array:
    return v / np.linalg.norm(v, axis=1).reshape(-1, 1)


def visualize(centers: np.ndarray, vectors: list[np.ndarray], light_axis: np.array) -> pv.Plotter:
    plotter = pv.Plotter()

    plotter.camera.position = (0.0, 0.0, 5.0)
    plotter.camera.focal_point = (0.0, 0.0, 0.0)
    plotter.camera.up = (0, 1, 0)

    # X, Y, Z axes
    plotter.add_mesh(
        pv.Spline(np.array([[0, 0, 0], [1, 0, 0]]), 10).tube(radius=0.005),
        color=[255, 0, 0],
    )
    plotter.add_mesh(
        pv.Spline(np.array([[0, 0, 0], [0, 1, 0]]), 10).tube(radius=0.005),
        color=[0, 255, 0],
    )
    plotter.add_mesh(
        pv.Spline(np.array([[0, 0, 0], [0, 0, 1]]), 10).tube(radius=0.005),
        color=[0, 0, 255],
    )

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
    plane_normal: np.array,
    plane_point: np.array,
    line_point: np.array,
    line_direction: np.array,
    tol: float = 1e-6,
) -> np.array:
    denom = np.dot(plane_normal, line_direction)

    if abs(denom) < tol:  # the line is parallel to the plane
        if abs(np.dot(plane_normal, line_point - plane_point)) < tol:
            return line_point  # the line lies on the plane
        else:
            return line_direction  # no intersection

    return line_point + (-np.dot(plane_normal, line_point - plane_point) / denom) * line_direction


def _compute_intersections(centers: np.ndarray, normals: np.ndarray, axis: np.array) -> np.ndarray:
    intersections = np.zeros_like(centers)
    line_point = np.array([0, 0, 0], dtype=np.float32)

    for i in range(centers.shape[0]):
        intersections[i, :] = _line_plane_intersection(normals[i], centers[i], line_point, axis)

    return intersections


logger.info("init")

# img_normals = openexr_numpy.imread(str(ASSET_DIR / "Moon_Normals.exr"), "XYZ")
# img_gray = cv2.imread(str(ASSET_DIR / "Moon_Image.tif"), cv2.IMREAD_GRAYSCALE)
# img_pxpos = np.load(ASSET_DIR / "Moon_Raytracing.npy")

# img_normals = openexr_numpy.imread(str(ASSET_DIR / "Moon_Normals_smooth.exr"), "XYZ")
# img_gray = cv2.imread(str(ASSET_DIR / "Moon_Image_smooth.tif"), cv2.IMREAD_GRAYSCALE)
# img_pxpos = np.load(ASSET_DIR / "Moon_Raytracing_smooth.npy")

img_normals = openexr_numpy.imread(str(ASSET_DIR / "Icosaeder_Normals.exr"), "XYZ")
img_gray = cv2.imread(str(ASSET_DIR / "Icosaeder_Image.tif"), cv2.IMREAD_GRAYSCALE)
img_pxpos = np.load(ASSET_DIR / "Icosaeder_Raytracing.npy")

# img_normals = openexr_numpy.imread(str(ASSET_DIR / "Icosaeder_negative_Normals.exr"), "XYZ")
# img_gray = cv2.imread(str(ASSET_DIR / "Icosaeder_negative_Image.tif"), cv2.IMREAD_GRAYSCALE)
# img_pxpos = np.load(ASSET_DIR / "Icosaeder_negative_Raytracing.npy")

logger.info("loaded")

light_axis = _normalize_vector(np.array(LIGHT_POS))

if RESIZE:
    img_normals = cv2.resize(img_normals, RESIZE_SIZE)
    img_gray = cv2.resize(img_gray, RESIZE_SIZE)
    img_pxpos = cv2.resize(img_pxpos, RESIZE_SIZE)

logger.info("resized")

# centers = []
# normals = []
#
# for x in range(img_normals.shape[1]):
#     for y in range(img_normals.shape[0]):
#         if abs(np.linalg.norm(img_normals[y, x, :])) < 0.01:
#             continue
#
#         # centers.append(np.array([
#         #     (x / img_normals.shape[1] - 0.5) * 2,
#         #     (y / img_normals.shape[0] - 0.5) * 2 * -1, # Y axis coordinate flip: numpy origin top-left, blender bottom-left
#         #     img_pxpos[y, x, 2]
#         # ]))
#
#         centers.append(np.nan_to_num(img_pxpos[y, x, :]))
#         normals.append(img_normals[y, x, :])
#
# centers = np.array(centers)
# normals = np.array(normals)
#
# intersections = _compute_intersections(centers, normals, light_axis)
# directions = _normalize_vectors(intersections - centers)
#
# # flip direction if intersection point is on the opposite end of the axis
# # necessary to avoid a flipping of direction signs when moving from one face to the next one
# opposite_directions = np.full_like(directions, 1, dtype=np.float32)
# opposite_directions[np.dot(intersections, light_axis) < 0] = -1
# directions *= opposite_directions
#
# ELEVATION_VECTOR_WEIGHT = 0.6
# elevation_vectors = _normalize_vectors(centers)
# field_elevation_vectors = []
# for i in range(len(elevation_vectors)):
#     projected = elevation_vectors[i] - (np.dot(elevation_vectors[i], normals[i])) * normals[i]
#     magnitude = np.arccos(np.dot(elevation_vectors[i], normals[i]))
#     combined = _normalize_vector(
#         directions[i] * (1 - magnitude) * (1 - ELEVATION_VECTOR_WEIGHT)
#         + projected * magnitude * ELEVATION_VECTOR_WEIGHT
#     )
#     field_elevation_vectors.append(combined)
#
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

img_field_elevation_vectors_0 = np.zeros_like(img_directions)
img_field_elevation_vectors_1 = np.zeros_like(img_directions)
img_field_elevation_vectors_2 = np.zeros_like(img_directions)
img_field_elevation_vectors_3 = np.zeros_like(img_directions)

ELEVATION_VECTOR_WEIGHT = 0.5

distance_point_to_light_axis = np.linalg.norm(light_axis - img_normals, axis=-1)
distance_weight = (distance_point_to_light_axis - np.min(distance_point_to_light_axis)) / np.ptp(
    distance_point_to_light_axis
)

for x in range(img_directions.shape[1]):
    for y in range(img_directions.shape[0]):
        elevation_vector = _normalize_vector(img_pxpos[y, x])
        projected = elevation_vector - (np.dot(elevation_vector, img_normals[y, x])) * img_normals[y, x]
        magnitude = np.arccos(np.dot(elevation_vector, img_normals[y, x]))

        img_field_elevation_vectors_0[y, x] = _normalize_vector(
            img_directions[y, x] * (1 - magnitude) * (1 - ELEVATION_VECTOR_WEIGHT)
            + projected * magnitude * ELEVATION_VECTOR_WEIGHT
        )

        img_field_elevation_vectors_1[y, x] = _normalize_vector(
            img_directions[y, x] * (1 - ELEVATION_VECTOR_WEIGHT) + projected * ELEVATION_VECTOR_WEIGHT
        )

        img_field_elevation_vectors_2[y, x] = _normalize_vector(
            img_directions[y, x] * (1 - distance_weight[y, x]) + projected * distance_weight[y, x]
        )

        img_field_elevation_vectors_3[y, x] = projected

if CROSS_FLOW:
    img_directions = np.cross(img_directions, img_normals)
    img_field_elevation_vectors_0 = np.cross(img_field_elevation_vectors_0, img_normals)
    img_field_elevation_vectors_1 = np.cross(img_field_elevation_vectors_1, img_normals)
    img_field_elevation_vectors_2 = np.cross(img_field_elevation_vectors_2, img_normals)
    img_field_elevation_vectors_3 = np.cross(img_field_elevation_vectors_3, img_normals)

logger.info("computed")

if VISUALIZE:
    centers = img_pxpos.reshape([-1, 3])
    normals = img_normals.reshape([-1, 3])
    directions = img_directions.reshape([-1, 3])
    # visualize(centers, [normals, directions], light_axis).show()
    # visualize(centers, [directions], light_axis).show()

    field_elevation_vectors = img_field_elevation_vectors_2.reshape([-1, 3])
    visualize(centers, [directions, field_elevation_vectors], light_axis).show()


def export_angles(arr):
    arr[:, :, 1] *= -1  # blender Y up / numpy Y down
    mapping_angle = np.atan2(arr[:, :, 1], arr[:, :, 0])
    mapping_angle = (mapping_angle + np.pi) / (np.pi * 2.0)
    mapping_angle = (mapping_angle * 255).astype(np.uint8)
    return mapping_angle


if EXPORT:
    mapping_distance = img_gray
    mapping_distance = ((mapping_distance - np.min(mapping_distance)) / np.ptp(mapping_distance) * 255).astype(np.uint8)
    cv2.imwrite(str(OUTPUT_DIR / "mapping_distance.png"), mapping_distance)

    img_directions[:, :, 1] *= -1  # blender Y up / numpy Y down
    mapping_angle = np.atan2(img_directions[:, :, 1], img_directions[:, :, 0])
    mapping_angle = (mapping_angle + np.pi) / (np.pi * 2.0)
    mapping_angle = (mapping_angle * 255).astype(np.uint8)
    cv2.imwrite(str(OUTPUT_DIR / "mapping_angle.png"), mapping_angle)

    # img_field_elevation_vectors[:, :, 1] *= -1 # blender Y up / numpy Y down
    # mapping_angle = np.atan2(img_field_elevation_vectors[:, :, 1], img_field_elevation_vectors[:, :, 0])
    # mapping_angle = (mapping_angle + np.pi) / (np.pi * 2.0)
    # mapping_angle = (mapping_angle * 255).astype(np.uint8)
    # cv2.imwrite(str(OUTPUT_DIR / f"mapping_angle_{ELEVATION_VECTOR_WEIGHT:3.1f}.png"), mapping_angle)

    cv2.imwrite(
        str(OUTPUT_DIR / "mapping_angle_0.png"),
        export_angles(img_field_elevation_vectors_0),
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "mapping_angle_1.png"),
        export_angles(img_field_elevation_vectors_1),
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "mapping_angle_2.png"),
        export_angles(img_field_elevation_vectors_2),
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "mapping_angle_3.png"),
        export_angles(img_field_elevation_vectors_3),
    )

    mapping_flat = np.zeros_like(img_pxpos, dtype=np.uint8)
    mapping_flat[np.isnan(img_pxpos)] = 255
    cv2.imwrite(str(OUTPUT_DIR / "mapping_flat.png"), mapping_flat)
