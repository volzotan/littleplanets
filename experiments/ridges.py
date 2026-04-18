import argparse
import datetime
from pathlib import Path

import cv2
import numpy as np
import openexr_numpy
import shapely
from scipy import ndimage
from shapely.geometry import LineString

import pyvista as pv

from util.misc import linestring_to_coordinate_pairs

DIR_DEBUG = Path("debug")

TOLERANCE: float = 1.0
OPENING_KERNEL_SIZE: int | None = 3
CLOSING_KERNEL_SIZE: int | None = 3

# RESIZE_SIZE = [2000, 2000]
# OUTPUT_SIZE = RESIZE_SIZE # [1000, 1000]

VISUALIZE = True


def write_npz(filename: Path, linestrings: list[LineString], include_z: bool = False) -> None:
    arrays = [shapely.get_coordinates(l, include_z=include_z) for l in linestrings]
    np.savez(filename, *arrays)


def visualize(centers: np.ndarray, vectors: list[np.ndarray], light_axis: np.ndarray | None) -> pv.Plotter:
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
    if light_axis is not None:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("normals", type=Path, default="Normals.exr", help="Normals (EXR)")
    parser.add_argument("raytrace", type=Path, default="raytrace.npy", help="Raytracing distance raster (NPY)")
    parser.add_argument("--output", type=Path, default="ridges.npz", help="Output filename [NPZ]")
    parser.add_argument("--debug", action="store_true", default=False, help="Write debug output")
    args = parser.parse_args()

    timer_start = datetime.datetime.now()

    img_normals = openexr_numpy.imread(str(args.normals), "XYZ")
    img_pxpos = np.load(args.raytrace)

    # if RESIZE_SIZE is not None:
    #     img_normals = cv2.resize(img_normals, RESIZE_SIZE)
    #     img_pxpos = cv2.resize(img_pxpos, RESIZE_SIZE)

    def _normalize_vectors(v: np.ndarray) -> np.array:
        return v / np.linalg.norm(v, axis=2)[:, :, np.newaxis]

    img_pxpos_normalized = _normalize_vectors(img_pxpos)

    # dot = np.abs(np.cos(np.dot(img_pxpos_normalized, img_normals)))
    # dot = np.dot(img_pxpos_normalized, img_normals)

    dot = np.sum(img_pxpos_normalized * img_normals, axis=2)

    # distance from pixel location to origin
    img_distance = np.linalg.norm(img_pxpos, axis=-1)
    img_distance = np.nan_to_num(img_distance)

    WINDOW_SIZE = 10
    MAX_WIN_VAR = 1e-6
    win_mean = ndimage.uniform_filter(img_distance, (WINDOW_SIZE, WINDOW_SIZE))
    win_sqr_mean = ndimage.uniform_filter(img_distance**2, (WINDOW_SIZE, WINDOW_SIZE))
    win_var = win_sqr_mean - win_mean**2

    win_var = np.clip(win_var, 0, MAX_WIN_VAR)
    win_var = win_var * -1 + MAX_WIN_VAR

    win_var = (np.iinfo(np.uint8).max * ((win_var - np.min(win_var)) / np.ptp(win_var))).astype(np.uint8)

    if args.debug:
        dot_non_nan = np.abs(np.nan_to_num(dot))
        dot_non_nan = (dot_non_nan / np.max(dot_non_nan) * 255).astype(np.uint8)
        cv2.imwrite(str(DIR_DEBUG / "ridges.png"), dot_non_nan)

        cv2.imwrite(str(DIR_DEBUG / "ridges_var.png"), win_var)

    print(dot)
    exit()

    if VISUALIZE:
        centers = img_pxpos.reshape([-1, 3])
        # normals = img_normals.reshape([-1, 3])
        dot_reshaped = dot.reshape([-1, 3])
        visualize(centers, [dot_reshaped], None).show()

    exit()

    # get only the silhouette of the rendered mesh

    mask_nan = np.isnan(np.sum(img_pxpos, axis=2))
    img_non_nan = np.full(mask_nan.shape, 255, dtype=np.uint8)
    img_non_nan[mask_nan] = 0
    contours, hierarchy = cv2.findContours(img_non_nan, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    linestrings_silhouette = []
    for contour in contours:
        linestrings_silhouette.append(LineString(contour[:, 0, :].tolist()))

    if args.debug:
        cv2.imwrite(str(DIR_DEBUG / "mask_nan.png"), img_non_nan)

        img_silhouette = np.zeros([img_non_nan.shape[0], img_non_nan.shape[1], 3], dtype=np.uint8)
        for ls in linestrings_silhouette:
            for pair in linestring_to_coordinate_pairs(ls):
                pt1 = [int(c) for c in pair[0]]
                pt2 = [int(c) for c in pair[1]]
                cv2.line(img_silhouette, pt1, pt2, (255, 255, 255), 2)
        cv2.imwrite(str(DIR_DEBUG / "silhouette.png"), img_silhouette)

    write_npz(args.output, linestrings_silhouette)
    print(f"total time: {(datetime.datetime.now() - timer_start).total_seconds():5.2f}s")
