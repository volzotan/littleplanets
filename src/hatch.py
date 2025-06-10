import math
import os
from pathlib import Path

import flowlines_py
import openexr_numpy
import shapely
import shapely.ops
import cv2
import numpy as np
from shapely import LineString
from shapely.geometry import Point

import flowlines


BLUR_MAPPING_ANGLE_KERNEL_SIZE = 1
BLUR_MAPPING_DISTANCE_KERNEL_SIZE = 1


def _linestring_to_coordinate_pairs(
    linestring: LineString,
) -> list[list[tuple[float, float]]]:
    pairs = []

    for i in range(len(linestring.coords) - 1):
        pairs.append([linestring.coords[i], linestring.coords[i + 1]])

    return pairs


def draw_line_image(canvas: np.ndarray, line_sets: list[list[LineString]], dimensions: list[int, int]) -> np.ndarray:
    scale_x = canvas.shape[1] / dimensions[0]
    scale_y = canvas.shape[0] / dimensions[1]

    colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]

    for li, lines in enumerate(line_sets):
        for linestring in lines:
            for pair in _linestring_to_coordinate_pairs(linestring):
                pt1 = [int(pair[0][0] * scale_x), int(pair[0][1] * scale_y)]
                pt2 = [int(pair[1][0] * scale_x), int(pair[1][1] * scale_y)]
                cv2.line(canvas, pt1, pt2, colors[li], 8)

    return canvas


def _project_linestring(ls: LineString, P: np.ndarray, scaling_factor: float) -> np.ndarray:
    xyz = shapely.get_coordinates(ls, include_z=True)
    coordinates = np.hstack([xyz, np.full([xyz.shape[0], 1], 1)])  # [x, y, z, w=1]
    coordinates = (P @ coordinates.T).T
    coordinates = coordinates[:, 0:2] / coordinates[:, 2][:, np.newaxis]
    coordinates = coordinates * scaling_factor
    return LineString(coordinates)


def _rotate_linestrings(lines: list[LineString], x: float, y: float, z: float) -> list[LineString]:
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(x), -np.sin(x)],
            [0, np.sin(x), np.cos(x)],
        ]
    )
    R_y = np.array(
        [
            [np.cos(y), 0, np.sin(y)],
            [0, 1, 0],
            [-np.sin(y), 0, np.cos(y)],
        ]
    )
    R_z = np.array(
        [
            [np.cos(z), -np.sin(z), 0],
            [np.sin(z), np.cos(z), 0],
            [0, 0, 1],
        ]
    )

    lines_rotated = []
    for line in lines:
        lines_rotated.append(shapely.ops.transform(lambda x, y, z: R_z @ R_y @ R_x @ np.array([x, y, z]), line))

    return lines_rotated


if __name__ == "__main__":
    # FILENAME_CAMERA_MATRIX = "../blender/P3x4.npy"
    FILENAME_CAMERA_MATRIX = "../blender/P3x4_2.npy"

    P = np.load(FILENAME_CAMERA_MATRIX)
    scaling_factor = 1000.0 / 6000.0

    # ls = LineString([
    #     [-1, 0, 0],
    #     [1, 0, 0]
    # ])

    # print(P @ np.array([0, 0, 0, 1]))

    linestrings_for_projection = []
    linestrings_for_projection.append(Point([0, 0]).buffer(0.10).boundary.segmentize(0.1))
    linestrings_for_projection.append(Point([0, 0]).buffer(0.20).boundary.segmentize(0.1))
    linestrings_for_projection.append(Point([0, 0]).buffer(0.30).boundary.segmentize(0.1))

    # add Z
    for i in range(len(linestrings_for_projection)):
        ls = linestrings_for_projection[i]
        coords = shapely.get_coordinates(ls)
        new_col = np.full([coords.shape[0], 1], 1.0)
        coords_with_z = np.concatenate((coords, new_col), axis=1)
        linestrings_for_projection[i] = LineString(coords_with_z)

    linestrings_for_projection = _rotate_linestrings(linestrings_for_projection, math.pi / 4, 0, math.pi / 4)
    linestrings_for_projection = [_project_linestring(l, P, scaling_factor) for l in linestrings_for_projection]

    FILENAME_OVERLAY = Path("../assets") / "Moon_linestrings_overlay.npz"
    overlay_npz = np.load(FILENAME_OVERLAY)
    linestrings_for_projection = [LineString(arr) for arr in overlay_npz.values()]
    linestrings_for_projection = [_project_linestring(l, P, scaling_factor) for l in linestrings_for_projection]


    exclusion_points = []
    for ls in linestrings_for_projection:
        exclusion_points += shapely.get_coordinates(ls).tolist()

    # FILENAME_MAPPING_ANGLE = "../output/mapping_angle.png"
    # FILENAME_MAPPING_ANGLE = "../output/mapping_angle_0.png"
    # FILENAME_MAPPING_ANGLE = "../output/mapping_angle_1.png"
    FILENAME_MAPPING_ANGLE = "../output/mapping_angle_2.png"
    # FILENAME_MAPPING_ANGLE = "../output/mapping_angle_3.png"

    # uint8 image must be centered around 128 to deal with negative values
    mapping_angle = cv2.imread(FILENAME_MAPPING_ANGLE, cv2.IMREAD_GRAYSCALE)
    mapping_distance = cv2.imread("../output/mapping_distance.png", cv2.IMREAD_GRAYSCALE)
    # mapping_distance = cv2.imread("../output/mapping_distance_increased_contrast.png", cv2.IMREAD_GRAYSCALE)
    mapping_flat = cv2.imread("../output/mapping_flat.png", cv2.IMREAD_GRAYSCALE)

    # mapping_angle = cv2.blur(mapping_angle, (BLUR_MAPPING_ANGLE_KERNEL_SIZE, BLUR_MAPPING_ANGLE_KERNEL_SIZE))
    # mapping_distance = cv2.blur(mapping_distance, (BLUR_MAPPING_DISTANCE_KERNEL_SIZE, BLUR_MAPPING_DISTANCE_KERNEL_SIZE))

    mapping_distance = ((mapping_distance - np.min(mapping_distance)) / np.ptp(mapping_distance) * 255).astype(np.uint8)

    # black_enhanced = np.zeros_like(mapping_distance, dtype=np.uint8)
    # black_enhanced[mapping_distance < 50] = 255
    # mapping_flat[black_enhanced > 0] = 255

    # cv2.imwrite("black_enhanced.png", black_enhanced)
    # exit()

    # white ink on black paper, invert grayscale image
    # mapping_distance = ~mapping_distance

    # mapping_distance = np.zeros_like(mapping_angle, dtype=np.uint8)
    mapping_max_length = np.zeros_like(mapping_angle)

    dimensions = [1000, 1000]
    mappings = [
        mapping_distance,
        mapping_angle,
        mapping_max_length,
        mapping_flat,
    ]

    config = flowlines_py.FlowlinesConfig()
    config.line_distance = (3.5, 10)
    config.line_max_length = [30] * 2
    config.line_step_distance = 0.25
    config.line_distance_end_factor = 0.5
    lines: list[list[tuple[float, float]]] = flowlines_py.hatch(dimensions, config, *mappings)
    linestrings = [shapely.simplify(LineString(l), 0.01) for l in lines]

    config = flowlines.FlowlineHatcherConfig()
    config.LINE_DISTANCE = (2.0, 9)
    config.LINE_MAX_LENGTH = [30] * 2
    config.LINE_STEP_DISTANCE = 0.25
    config.LINE_DISTANCE_END_FACTOR = 0.50
    hatcher = flowlines.FlowlineHatcher(dimensions, *mappings, config, exclusion_points=exclusion_points)
    linestrings: list[LineString] = hatcher.hatch()
    linestrings = [shapely.simplify(l, 0.01) for l in linestrings]

    print(f"num linestrings: {len(linestrings)}")

    # canvas = cv2.resize((img_gray * 0.5).astype(np.uint8), output_dimensions)
    canvas = np.full([int(dimensions[0] * 5), int(dimensions[1] * 5), 3], 255, dtype=np.uint8)

    cv2.imwrite(
        str(".." / Path("foo_" + Path(FILENAME_MAPPING_ANGLE).name)),
        draw_line_image(canvas, [linestrings, linestrings_for_projection], dimensions),
    )
