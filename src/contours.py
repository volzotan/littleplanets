import argparse
import datetime
from pathlib import Path

import cv2
import numpy as np
import openexr_numpy
import shapely
from shapely.geometry import LineString, Polygon
from skimage.morphology import skeletonize

import random

from util.misc import linestring_to_coordinate_pairs

DIR_DEBUG = Path("debug")

TOLERANCE: float = 1.0
OPENING_KERNEL_SIZE: int | None = 3
CLOSING_KERNEL_SIZE: int | None = 3

SILHOUETTE_SHRINKING = 2.0
SILHOUETTE_SIMPLIFY = 0.1

# RESIZE_SIZE = [2000, 2000]
# OUTPUT_SIZE = RESIZE_SIZE # [1000, 1000]

random.seed("littleplanets")


def write_npz(filename: Path, linestrings: list[LineString], include_z: bool = False) -> None:
    arrays = [shapely.get_coordinates(l, include_z=include_z) for l in linestrings]
    np.savez(filename, *arrays)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("normals", type=Path, default="Normals.exr", help="Normals (EXR)")
    parser.add_argument("raytrace", type=Path, default="raytrace.npy", help="Raytracing distance raster (NPY)")
    parser.add_argument("--output", type=Path, default="contours.npz", help="Output filename [NPZ]")
    parser.add_argument("--debug", action="store_true", default=False, help="Write debug output")
    args = parser.parse_args()

    img_normals = openexr_numpy.imread(str(args.normals), "XYZ")
    img_pxpos = np.load(args.raytrace)

    # if RESIZE_SIZE is not None:
    #     img_normals = cv2.resize(img_normals, RESIZE_SIZE)
    #     img_pxpos = cv2.resize(img_pxpos, RESIZE_SIZE)

    # get only the silhouette of the rendered mesh

    mask_nan = np.isnan(np.sum(img_pxpos, axis=2))
    img_non_nan = np.full(mask_nan.shape, 255, dtype=np.uint8)
    img_non_nan[mask_nan] = 0
    contours, hierarchy = cv2.findContours(img_non_nan, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    polygons_silhouette = []
    for contour in contours:
        p = Polygon(contour[:, 0, :].tolist())
        if p.area > 10.0:
            polygons_silhouette.append(p)

    polygons_silhouette = [p.buffer(-SILHOUETTE_SHRINKING).simplify(SILHOUETTE_SIMPLIFY) for p in polygons_silhouette]
    linestrings_silhouette = [LineString(p.exterior.coords) for p in polygons_silhouette]

    if args.debug:
        cv2.imwrite(str(DIR_DEBUG / "mask_nan.png"), img_non_nan)

        img_silhouette = np.zeros([img_non_nan.shape[0], img_non_nan.shape[1], 3], dtype=np.uint8)
        for ls in linestrings_silhouette:
            color = random.choices(range(256), k=3)
            for pair in linestring_to_coordinate_pairs(ls):
                pt1 = [int(c) for c in pair[0]]
                pt2 = [int(c) for c in pair[1]]
                cv2.line(img_silhouette, pt1, pt2, color, 2)
        cv2.imwrite(str(DIR_DEBUG / "silhouette.png"), img_silhouette)

    write_npz(args.output, linestrings_silhouette)

    exit()  # TODO

    camera_axis = np.array([0, 0, 7])
    camera_pos = np.array([0, 0, 7])

    # img_pxpos = np.nan_to_num(img_pxpos)
    dot = np.full(img_normals.shape[0:2], 0, dtype=float)
    for i in range(img_normals.shape[0]):
        for j in range(img_normals.shape[1]):
            dot[i, j] = np.dot(img_normals[i, j], camera_pos - img_pxpos[i, j])

    out = np.zeros_like(dot)
    out[np.abs(dot) < TOLERANCE] = 255
    out[np.isnan(np.sum(img_pxpos, axis=2))] = 0

    if OPENING_KERNEL_SIZE is not None:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((OPENING_KERNEL_SIZE, OPENING_KERNEL_SIZE), dtype=np.uint8))

    if CLOSING_KERNEL_SIZE is not None:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((CLOSING_KERNEL_SIZE, CLOSING_KERNEL_SIZE), dtype=np.uint8))

    skeleton = skeletonize(out)

    out = cv2.resize(out, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
    skeleton = cv2.resize(skeleton.astype(np.uint8) * 255, OUTPUT_SIZE, interpolation=cv2.INTER_AREA).astype(bool)

    img = np.dstack([out, skeleton.astype(np.uint8) * 255, np.zeros_like(out)])

    if args.debug:
        # cv2.imwrite(str(OUTPUT_DIR / "cross.png"), out)
        # cv2.imwrite(str(OUTPUT_DIR / "skeleton.png"), skeleton.astype(np.uint8)*255)
        cv2.imwrite(str(DIR_DEBUG / "skeleton.png"), img)

        exit()

    # cross = np.dot(img_normals, camera_axis-img_normals)
    # cross = np.abs(cross)
    # cross = np.mean(cross, axis=2)
    # cross = cross / np.max(cross)
    #
    # print(np.min(cross), np.max(cross))
    #
    # cv2.imwrite(str(OUTPUT_DIR / "cross.png"), (cross * 255).astype(np.uint8))

    timer_start = datetime.datetime.now()

    from trace_skeleton import *

    im0 = np.zeros([skeleton.shape[0], skeleton.shape[1], 3], dtype=np.uint8)

    rects = []
    polys = traceSkeleton(skeleton, 0, 0, skeleton.shape[1], skeleton.shape[0], 10, 999, rects)

    for l in polys:
        c = (200 * random.random(), 200 * random.random(), 200 * random.random())
        for i in range(0, len(l) - 1):
            cv2.line(im0, (l[i][0], l[i][1]), (l[i + 1][0], l[i + 1][1]), c)

    # cv2.imshow('',im0);cv2.waitKey(0)

    print(f"total time: {(datetime.datetime.now() - timer_start).total_seconds():5.2f}s")

    if args.debug:
        cv2.imwrite(str(DIR_DEBUG / "traced.png"), im0)

    write_npz(args.output, [LineString(p) for p in polys])
