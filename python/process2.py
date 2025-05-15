import math
import os
from pathlib import Path

import flowlines_py
import openexr_numpy
import shapely.affinity

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np

from shapely import LineString


def get_slope(
    data: np.ndarray, sampling_step: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes angle (in rad) and magnitude of the given 2D array of values
    """
    test_slice = data[::sampling_step, ::sampling_step]
    r, c = np.shape(data)
    Y, X = np.mgrid[0:r:sampling_step, 0:c:sampling_step]
    dY, dX = np.gradient(test_slice)  # order! Y X

    angles = np.arctan2(dY, dX)
    magnitude = np.hypot(dY, dX)

    if sampling_step > 1:
        angles = cv2.resize(angles, data.shape)
        magnitude = cv2.resize(magnitude, data.shape)

    return (X, Y, dX, dY, angles, magnitude)

def _linestring_to_coordinate_pairs(linestring: LineString) -> list[list[tuple[float, float]]]:
    pairs = []

    for i in range(len(linestring.coords) - 1):
        pairs.append([linestring.coords[i], linestring.coords[i + 1]])

    return pairs

def draw_line_image(canvas: np.ndarray, lines: LineString) -> np.ndarray:

    for linestring in lines:
        for pair in _linestring_to_coordinate_pairs(linestring):
            pt1 = [int(c) for c in pair[0]]
            pt2 = [int(c) for c in pair[1]]
            cv2.line(canvas, pt1, pt2, (255, 255, 255), 2)

    return canvas



img_depth = openexr_numpy.imread("assets/Depth0001.exr", "V")
img_gray = cv2.imread("assets/Image0001.tif", cv2.IMREAD_GRAYSCALE)

_, _, dX, dY, angles, magnitude = get_slope(img_depth, 1)

angles += math.pi/2 # contour, not slope-following

img_gray = ((img_gray.astype(float) - np.min(img_gray)) / np.ptp(img_gray) * 255).astype(np.uint8)
mapping_distance = ~img_gray # draw white, not black

cv2.imwrite("narf.png", mapping_distance)

# uint8 image must be centered around 128 to deal with negative values
mapping_angle = ((angles + math.pi) / math.tau * 255.0).astype(np.uint8)

mapping_max_length = np.zeros_like(mapping_distance)

mapping_flat = np.zeros_like(mapping_distance)
mapping_flat[img_depth > 10] = 255

# mapping_distance = np.zeros_like(mapping_distance)

dimensions = [750, 750]
output_dimensions = [3000, 3000]
mappings = [
    mapping_distance,
    mapping_angle,
    mapping_max_length,
    mapping_flat,
]

config = flowlines_py.FlowlinesConfig()

config.line_distance = (0.5, 25)
config.line_max_length = (50, 50)

lines: list[list[tuple[float, float]]] = flowlines_py.hatch(
    dimensions, config, *mappings
)
linestrings = [LineString(l) for l in lines]
linestrings = [
    shapely.simplify(
        shapely.affinity.scale(
            l,
            xfact=output_dimensions[1]/dimensions[1],
            yfact=output_dimensions[0]/dimensions[0],
            origin=(0, 0)
        ),
        0.1,
    )
for l in linestrings]

print(len(linestrings))

canvas = cv2.resize((img_gray * 0.5).astype(np.uint8), output_dimensions)
canvas = np.zeros(output_dimensions, dtype=np.uint8)
cv2.imwrite("foo.png", draw_line_image(canvas, linestrings))





