import datetime
from pathlib import Path

import cv2
import numpy as np
import openexr_numpy
import shapely
from shapely.geometry import LineString
from skimage.morphology import skeletonize

ASSET_DIR: Path = Path("..", "assets")
OUTPUT_DIR: Path = ASSET_DIR

TOLERANCE: float = 2.0
OPENING_KERNEL_SIZE: int | None = None  # 3

RESIZE_SIZE = [1000, 1000]

OUTPUT_FILENAME = "contours.npz"

def write_npz(filename: Path, linestrings: list[LineString]) -> None:
    arrays = [shapely.get_coordinates(l, include_z=True) for l in linestrings]
    np.savez(filename, *arrays)


# img_normals = openexr_numpy.imread(str(ASSET_DIR / "Icosaeder_Normals.exr"), "XYZ")
# img_pxpos = np.load(ASSET_DIR / "Icosaeder_Raytracing.npy")

img_normals = openexr_numpy.imread(str(ASSET_DIR / "Icosaeder_negative_Normals.exr"), "XYZ")
img_pxpos = np.load(ASSET_DIR / "Icosaeder_negative_Raytracing.npy")

# img_normals = openexr_numpy.imread(str(ASSET_DIR / "Moon_Normals_smooth.exr"), "XYZ")
# img_pxpos = np.load(ASSET_DIR / "Moon_Raytracing_smooth.npy")

if RESIZE_SIZE is not None:
    img_normals = cv2.resize(img_normals, RESIZE_SIZE)
    img_pxpos = cv2.resize(img_pxpos, RESIZE_SIZE)

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

skeleton = skeletonize(out)

img = np.dstack([out, skeleton.astype(np.uint8) * 255, np.zeros_like(out)])

# cv2.imwrite(str(OUTPUT_DIR / "cross.png"), out)
# cv2.imwrite(str(OUTPUT_DIR / "skeleton.png"), skeleton.astype(np.uint8)*255)
cv2.imwrite(str(OUTPUT_DIR / "skeleton.png"), img)

# cross = np.dot(img_normals, camera_axis-img_normals)
# cross = np.abs(cross)
# cross = np.mean(cross, axis=2)
# cross = cross / np.max(cross)
#
# print(np.min(cross), np.max(cross))
#
# cv2.imwrite(str(OUTPUT_DIR / "cross.png"), (cross * 255).astype(np.uint8))

timer_start = datetime.datetime.now()

import random

from trace_skeleton import *

im = thinning(skeleton)
im0 = np.zeros([skeleton.shape[0], skeleton.shape[1], 3], dtype=np.uint8)

rects = []
polys = traceSkeleton(im, 0, 0, im.shape[1], im.shape[0], 10, 999, rects)

for l in polys:
    c = (200 * random.random(), 200 * random.random(), 200 * random.random())
    for i in range(0, len(l) - 1):
        cv2.line(im0, (l[i][0], l[i][1]), (l[i + 1][0], l[i + 1][1]), c)

# cv2.imshow('',im0);cv2.waitKey(0)

print(f"total time: {(datetime.datetime.now() - timer_start).total_seconds():5.2f}s")

cv2.imwrite(str(OUTPUT_DIR / "traced.png"), im0)

write_npz(OUTPUT_DIR / OUTPUT_FILENAME, [LineString(p) for p in polys])
