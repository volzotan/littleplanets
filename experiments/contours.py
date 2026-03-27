import argparse
import math
from pathlib import Path

import numpy as np
import pyvista as pv
import openexr_numpy
import cv2

from src.util.misc import normalize_vectors, normalize_vector


def _extract_camera_world_position_from_projection_matrix(P: np.ndarray) -> tuple[float, float, float]:
    _, _, Vt = np.linalg.svd(P)
    C_homogeneous = Vt[-1]
    C = C_homogeneous[:3] / C_homogeneous[3]
    return C.tolist()


def angle_between(u, v):
    dot = np.dot(u, v)
    dot = np.clip(dot, -1.0, 1.0)  # numerical safety
    return np.arccos(dot)


def _visualize(centers: np.ndarray, vectors: list[np.ndarray], points: list[np.ndarray]) -> pv.Plotter:
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

    colors = ["red", "green", "blue"]

    for vi, vector in enumerate(vectors):
        for i in range(len(vector)):
            if np.isnan(np.sum(centers[i])):
                continue

            if np.isnan(np.sum(vector[i])):
                continue

            if np.sum(np.abs(vector[i])) == 0.0:
                continue

            arrow = pv.Arrow(centers[i], vector[i], scale=0.05)
            plotter.add_mesh(arrow, color=colors[vi])

            # line = pv.Line(centers[i], centers[i] + vector[i])
            # plotter.add_mesh(line)

    for pi, point_set in enumerate(points):
        for point in point_set:
            sphere = pv.Sphere(0.005, point)
            plotter.add_mesh(sphere, color=colors[pi])

    return plotter


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("normals", type=Path, default="normals.exr", help="Normals (EXR)")
    parser.add_argument("raytrace", type=Path, default="raytrace.npy", help="Raytracing distance raster (NPY)")
    parser.add_argument("--projection-matrix", type=Path, default=None, help="3x4 projection matrix (NPY)")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug output")
    parser.add_argument("--visualize", action="store_true", default=False, help="Enable interactive visualization")

    args = parser.parse_args()

    img_normals = openexr_numpy.imread(str(args.normals), "XYZ")
    img_pxpos = np.load(args.raytrace)

    projection_matrix = np.load(args.projection_matrix)

    _, _, camera_z = _extract_camera_world_position_from_projection_matrix(projection_matrix)

    output = np.zeros(img_normals.shape[0:2])
    for i in range(img_normals.shape[0]):
        for j in range(img_normals.shape[1]):
            u = normalize_vector(img_normals[i, j])
            v = normalize_vector(np.array([0, 0, camera_z]) - img_pxpos[i, j])
            output[i, j] = angle_between(u, v) / (math.pi / 2)  # max 90 deg

    print(np.min(np.nan_to_num(output)))
    print(np.max(np.nan_to_num(output)))

    output_int = (output * 255).astype(np.uint8)
    output_int[output_int < 240] = 0

    cv2.imwrite("debug/foo.png", output_int)

    exit()

    if args.visualize:
        resize_size = [30, 30]
        img_normals = cv2.resize(img_normals, resize_size)
        img_pxpos = cv2.resize(img_pxpos, resize_size)

    if args.visualize:
        # in world space

        centers = img_pxpos.reshape([-1, 3])
        normals = img_normals.reshape([-1, 3])

        _visualize(centers, [normals], []).show()


if __name__ == "__main__":
    main()
