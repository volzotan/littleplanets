import math
import os
from pathlib import Path

import flowlines_py
import openexr_numpy
import shapely.affinity


import cv2
import numpy as np

from shapely import LineString

import flowlines


def _linestring_to_coordinate_pairs(
    linestring: LineString,
) -> list[list[tuple[float, float]]]:
    pairs = []

    for i in range(len(linestring.coords) - 1):
        pairs.append([linestring.coords[i], linestring.coords[i + 1]])

    return pairs


def draw_line_image(canvas: np.ndarray, lines: LineString, dimensions: list[int, int]) -> np.ndarray:
    scale_x = canvas.shape[1] / dimensions[0]
    scale_y = canvas.shape[0] / dimensions[1]

    for linestring in lines:
        for pair in _linestring_to_coordinate_pairs(linestring):
            pt1 = [int(pair[0][0] * scale_x), int(pair[0][1] * scale_y)]
            pt2 = [int(pair[1][0] * scale_x), int(pair[1][1] * scale_y)]
            cv2.line(canvas, pt1, pt2, (255, 255, 255), 2)

    return canvas


if __name__ == "__main__":

    # uint8 image must be centered around 128 to deal with negative values
    mapping_angle = cv2.imread("../output/mapping_angle.png", cv2.IMREAD_GRAYSCALE)
    mapping_distance = cv2.imread("../output/mapping_distance.png", cv2.IMREAD_GRAYSCALE)
    mapping_flat = cv2.imread("../output/mapping_flat.png", cv2.IMREAD_GRAYSCALE)

    mapping_angle = cv2.blur(mapping_angle, )


    # white ink on black paper, invert grayscale image
    mapping_distance = ~mapping_distance

    # mapping_distance = np.zeros_like(mapping_angle, dtype=np.uint8)
    mapping_max_length = np.zeros_like(mapping_angle)

    dimensions = [500, 500]

    # mappings = [
    #     mapping_distance,
    #     mapping_angle,
    #     mapping_max_length,
    #     mapping_flat,
    # ]
    #
    # config = flowlines_py.FlowlinesConfig()
    # config.line_distance = (15, 50)
    # config.line_max_length = (1000, 1000)
    # lines: list[list[tuple[float, float]]] = flowlines_py.hatch((dimensions[1], dimensions[0]), config, *mappings)
    # linestrings = [shapely.simplify(LineString(l), 1) for l in lines]

    config = flowlines.FlowlineHatcherConfig()
    config.LINE_DISTANCE = (.5, 10)
    config.LINE_MAX_LENGTH = (10, 10)
    config.LINE_DISTANCE_END_FACTOR = 0.25
    hatcher = flowlines.FlowlineHatcher(
        dimensions,
        mapping_distance,
        mapping_angle,
        mapping_max_length,
        mapping_flat,
        config
    )
    linestrings: list[LineString] = hatcher.hatch()
    linestrings = [shapely.simplify(l, 1) for l in linestrings]

    print(f"num linestrings: {len(linestrings)}")

    # canvas = cv2.resize((img_gray * 0.5).astype(np.uint8), output_dimensions)
    canvas = np.zeros([
        int(dimensions[0] * 5),
        int(dimensions[1] * 5),
        3
    ], dtype=np.uint8)
    cv2.imwrite("foo.png", draw_line_image(canvas, linestrings, dimensions))
