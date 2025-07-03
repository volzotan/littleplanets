import argparse
import datetime
import math
from pathlib import Path

import cv2

# import flowlines_py
import numpy as np
import shapely
import shapely.ops
from shapely import LineString
from shapely.geometry import Point

import flowlines

BLUR_MAPPING_ANGLE_KERNEL_SIZE = 1
BLUR_MAPPING_DISTANCE_KERNEL_SIZE = 1

INVERT_COLOR = False
CUTOUT_THRESHOLD = 230

INVERT_COLOR = True
CUTOUT_THRESHOLD = 50


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

    # colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
    colors = [(0, 0, 0)]
    # colors = [(255, 255, 255), (0, 255, 255), (0, 0, 255)]
    # colors = [(255, 255, 255)] * 10

    if INVERT_COLOR:
        colors = [(255, 255, 255)]

    colors *= 10

    for li, lines in enumerate(line_sets):
        for linestring in lines:
            for pair in _linestring_to_coordinate_pairs(linestring):
                pt1 = [int(pair[0][0] * scale_x), int(pair[0][1] * scale_y)]
                pt2 = [int(pair[1][0] * scale_x), int(pair[1][1] * scale_y)]
                cv2.line(canvas, pt1, pt2, colors[li], 3)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("mapping_angle", type=Path, default="mapping_angle.png", help="Mapping angle (PNG)")
    parser.add_argument("mapping_distance", type=Path, default="mapping_distance.png", help="Mapping distance (PNG)")
    parser.add_argument("mapping_flat", type=Path, default="mapping_flat.png", help="Mapping flat (PNG)")
    parser.add_argument("overlay", type=Path, default="overlay.npz", help="Overlay linestrings (NPZ)")
    parser.add_argument("projection_matrix", type=Path, default="P3x4.npy", help="3x4 projection matrix (NPY)")
    parser.add_argument("--output", type=Path, default="littleplanet.png", help="Output filename")
    args = parser.parse_args()

    # FILENAME_CONTOURS = Path("..", "assets") / "contours.npz"

    P = np.load(args.projection_matrix)
    scaling_factor = 1000.0 / 6000.0

    overlay_npz = np.load(args.overlay)
    linestrings_overlay = [LineString(arr) for arr in overlay_npz.values()]
    linestrings_overlay = [_project_linestring(l, P, scaling_factor) for l in linestrings_overlay]

    # contours_npz = np.load(FILENAME_CONTOURS)
    # TODO: contours don't need to be projected, but they need to be scaled (currently missing!)
    # linestrings_contours = [LineString(arr) for arr in contours_npz.values()]
    linestrings_contours = []

    exclusion_points = []
    for ls in linestrings_overlay:
        exclusion_points += shapely.get_coordinates(ls.segmentize(0.01)).tolist()

    contour_points = []
    for ls in linestrings_contours:
        contour_points += shapely.get_coordinates(ls.segmentize(0.01)).tolist()

    # FILENAME_MAPPING_ANGLE = "../output/mapping_angle.png"
    # FILENAME_MAPPING_ANGLE = "../output/mapping_angle_0.png"
    # FILENAME_MAPPING_ANGLE = "../output/mapping_angle_1.png"
    FILENAME_MAPPING_ANGLE = "../output/mapping_angle_2.png"
    # FILENAME_MAPPING_ANGLE = "../output/mapping_angle_3.png"

    # uint8 image must be centered around 128 to deal with negative values
    mapping_angle = cv2.imread(args.mapping_angle, cv2.IMREAD_GRAYSCALE)
    mapping_distance = cv2.imread(args.mapping_distance, cv2.IMREAD_GRAYSCALE)
    mapping_flat = cv2.imread(args.mapping_flat, cv2.IMREAD_GRAYSCALE)

    # mapping_angle = cv2.blur(mapping_angle, (BLUR_MAPPING_ANGLE_KERNEL_SIZE, BLUR_MAPPING_ANGLE_KERNEL_SIZE))
    # mapping_distance = cv2.blur(mapping_distance, (BLUR_MAPPING_DISTANCE_KERNEL_SIZE, BLUR_MAPPING_DISTANCE_KERNEL_SIZE))

    mapping_distance = ((mapping_distance - np.min(mapping_distance)) / np.ptp(mapping_distance) * 255).astype(np.uint8)

    if not INVERT_COLOR:
        # all areas above a brightness threshold should be kept empty
        mapping_flat[mapping_distance > CUTOUT_THRESHOLD] = 255
    else:  # white ink on black paper, invert grayscale image
        mapping_flat[mapping_distance < CUTOUT_THRESHOLD] = 255
        mapping_distance = ~mapping_distance

    # mapping_distance = np.zeros_like(mapping_angle, dtype=np.uint8)
    mapping_max_length = np.zeros_like(mapping_angle)

    dimensions = [1000, 1000]
    mappings = [
        mapping_distance,
        mapping_angle,
        mapping_max_length,
        mapping_flat,
    ]

    # config = flowlines_py.FlowlinesConfig()
    # config.line_distance = (3.5, 10)
    # config.line_max_length = [30] * 2
    # config.line_step_distance = 0.25
    # config.line_distance_end_factor = 0.5
    # lines: list[list[tuple[float, float]]] = flowlines_py.hatch(dimensions, config, *mappings)
    # linestrings = [shapely.simplify(LineString(l), 0.01) for l in lines]

    config = flowlines.FlowlineHatcherConfig()
    config.LINE_DISTANCE = (0.8, 15)
    config.LINE_MAX_LENGTH = [30, 60]  # [50] * 2 #[10, 200]
    config.LINE_STEP_DISTANCE = 0.25
    config.LINE_DISTANCE_END_FACTOR = 0.50
    # hatcher = flowlines.FlowlineHatcher(dimensions, *mappings, config, exclusion_points=exclusion_points + contour_points)
    # hatcher = flowlines.FlowlineHatcher(dimensions, *mappings, config, exclusion_points=exclusion_points, initial_seed_points=contour_points)
    hatcher = flowlines.FlowlineHatcher(dimensions, *mappings, config, exclusion_points=exclusion_points)
    linestrings: list[LineString] = hatcher.hatch()
    linestrings = [shapely.simplify(l, 0.01) for l in linestrings]

    print(f"num linestrings: {len(linestrings)}")

    # cut buffered overlay from hatched linestrings
    timer_start = datetime.datetime.now()
    stencil = shapely.ops.unary_union(linestrings_overlay).buffer(2.5)
    linestrings_cut = []
    for ls in linestrings:
        cut = shapely.difference(ls, stencil)
        if not cut.is_empty and type(cut) is LineString:
            linestrings_cut.append(cut)
    linestrings = linestrings_cut
    print(f"time: {(datetime.datetime.now() - timer_start).total_seconds():5.2f}")

    linestrings_stencil = []
    for g in stencil.boundary.geoms:
        linestrings_stencil.append(g)

    # canvas = cv2.resize((img_gray * 0.5).astype(np.uint8), output_dimensions)
    canvas = np.full([int(dimensions[0] * 10), int(dimensions[1] * 10), 3], 0 if INVERT_COLOR else 255, dtype=np.uint8)

    cv2.imwrite(
        # str(".." / Path("foo_" + Path(FILENAME_MAPPING_ANGLE).name)),
        str(args.output),
        # draw_line_image(canvas, [linestrings, linestrings_overlay, linestrings_contours], dimensions),
        draw_line_image(canvas, [linestrings, linestrings_overlay], dimensions),
        # draw_line_image(canvas, [linestrings_stencil], dimensions),
        # draw_line_image(canvas, [linestrings], dimensions),
    )
