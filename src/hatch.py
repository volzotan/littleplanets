import argparse
import datetime
from pathlib import Path
import math
import random
from typing import Any

import cv2

# import flowlines_py
import numpy as np
import shapely
import shapely.ops
from shapely import LineString
from skimage.color import rgb2lab, deltaE_cie76, deltaE_ciede2000, deltaE_ciede94

import flowlines
from util.misc import linestring_to_coordinate_pairs
from svgwriter import SvgWriter

DIR_DEBUG = Path("debug")

BLUR_MAPPING_ANGLE_KERNEL_SIZE = 1
BLUR_MAPPING_DISTANCE_KERNEL_SIZE = 1

# INVERT_COLOR = False
# CUTOUT_THRESHOLD = 230

INVERT_COLOR = True
# CUTOUT_THRESHOLD = 10


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
            for pair in linestring_to_coordinate_pairs(linestring):
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


def _blur_raster(raster: np.ndarray, perc: float) -> np.ndarray:
    kernel_size = int(max(*raster.shape) * (perc / 100.0))
    if kernel_size > 1:
        return cv2.blur(raster, (kernel_size, kernel_size))
    else:
        return raster


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mapping_color", type=Path, default="mapping_color.npy", help="Mapping color (NPY)")
    parser.add_argument("mapping_angle", type=Path, default="mapping_angle.png", help="Mapping angle (PNG)")
    parser.add_argument("mapping_distance", type=Path, default="mapping_distance.png", help="Mapping distance (PNG)")
    parser.add_argument("mapping_line_length", type=Path, default="mapping_length.png", help="Mapping line length (PNG)")
    parser.add_argument("mapping_flat", type=Path, default="mapping_flat.png", help="Mapping flat (PNG)")

    parser.add_argument("--overlay", type=Path, default=None, help="Overlay linestrings (NPZ)")
    parser.add_argument("--projection-matrix", type=Path, default=None, help="3x4 projection matrix (NPY)")

    parser.add_argument("--contours", type=Path, default=None, help="Contour linestrings (NPZ)")
    # parser.add_argument("--scaling-factor", type=float, default=1.0, help="Scaling factor of the mapping rasters with regard to the original blender export")

    parser.add_argument("--blur-color", type=float, default=0, help="Blurring kernel size. Percentage of raster size (float)")
    parser.add_argument("--blur-angle", type=float, default=0, help="Blurring kernel size. Percentage of raster size (float)")
    parser.add_argument("--blur-distance", type=float, default=0, help="Blurring kernel size. Percentage of raster size (float)")

    parser.add_argument("--output", type=Path, default="littleplanet.svg", help="Output filename (SVG)")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug (image) output")
    parser.add_argument("--suffix", type=str, default="", help="Filename suffix to be appended to all (debug) output")

    parser.add_argument("--config-line-distance-end-factor", type=float, default=None)

    args = parser.parse_args()

    dimensions = [750, 750]

    mapping_color = None
    if args.mapping_color.suffix == ".npy":
        mapping_color = np.load(args.mapping_color)
    else:
        mapping_color = cv2.imread(args.mapping_color)

    mapping_angle = cv2.imread(args.mapping_angle, cv2.IMREAD_GRAYSCALE)  # uint8 image must be centered around 128 to deal with negative values
    mapping_distance = cv2.imread(args.mapping_distance, cv2.IMREAD_GRAYSCALE)
    mapping_line_length = cv2.imread(args.mapping_line_length, cv2.IMREAD_GRAYSCALE)
    mapping_flat = cv2.imread(args.mapping_flat, cv2.IMREAD_GRAYSCALE)

    mapping_color = _blur_raster(mapping_color, args.blur_color)
    mapping_angle = _blur_raster(mapping_angle, args.blur_angle)
    mapping_distance = _blur_raster(mapping_distance, args.blur_distance)

    if args.debug:
        cv2.imwrite(str(DIR_DEBUG / f"hatch_mapping_color{args.suffix}.png"), mapping_color)
        cv2.imwrite(str(DIR_DEBUG / f"hatch_mapping_angle{args.suffix}.png"), mapping_angle)
        cv2.imwrite(str(DIR_DEBUG / f"hatch_mapping_distance{args.suffix}.png"), mapping_distance)

    scaling_factor = dimensions[0] / mapping_angle.shape[1]

    linestrings_overlay = []
    if args.overlay is not None:
        P = np.load(args.projection_matrix)

        overlay_npz = np.load(args.overlay)
        linestrings_overlay = [LineString(arr) for arr in overlay_npz.values()]
        linestrings_overlay = [_project_linestring(l, P, scaling_factor) for l in linestrings_overlay]

    # TODO: contours don't need to be projected, but they need to be scaled (currently missing!)
    linestrings_contours = []
    if args.contours is not None:
        contours_npz = np.load(args.contours)
        linestrings_contours = [LineString(arr) for arr in contours_npz.values()]
        linestrings_contours = [shapely.affinity.scale(ls, xfact=scaling_factor, yfact=scaling_factor, origin=(0, 0)) for ls in linestrings_contours]

    exclusion_points = []
    for ls in linestrings_overlay + linestrings_contours:
        exclusion_points += shapely.get_coordinates(ls.segmentize(0.01)).tolist()

    mapping_distance = ((mapping_distance - np.min(mapping_distance)) / np.ptp(mapping_distance) * 255).astype(np.uint8)

    # all areas above/below a brightness threshold should be kept empty
    # if not INVERT_COLOR:
    #     mapping_flat[mapping_distance > CUTOUT_THRESHOLD] = 255
    # else:
    #     mapping_flat[mapping_distance < CUTOUT_THRESHOLD] = 255

    if INVERT_COLOR:
        # white ink on black paper, invert grayscale image
        mapping_distance = ~mapping_distance

    # mapping_distance = np.zeros_like(mapping_angle, dtype=np.uint8)
    # mapping_line_length = np.zeros_like(mapping_angle)

    mappings = [
        mapping_distance,
        mapping_angle,
        mapping_line_length,
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
    config.LINE_DISTANCE = (0.8, 10)
    config.LINE_MAX_LENGTH = [10, 25]  # [20, 100]  # [10, 50]  # [50] * 2 #[10, 200]
    config.LINE_STEP_DISTANCE = 0.15
    config.LINE_DISTANCE_END_FACTOR = 0.25
    config.MAX_ANGLE_DISCONTINUITY = math.pi / 12

    if args.config_line_distance_end_factor is not None:
        config.LINE_DISTANCE_END_FACTOR = args.config_line_distance_end_factor

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

    # Coloring

    # Coloring Attempt 1
    # palette = [
    #     [255, 0, 0],
    #     [0, 255, 0],
    #     [0, 0, 255],
    # ]
    #
    # linestrings_split_by_palette = [[] for _ in range(len(palette))]
    #
    # if len(palette) > 0:
    #     palette_labColor = [rgb2lab(np.array(c) / 255.0) for c in palette]
    #
    #     mapping_color_rgb = cv2.cvtColor(mapping_color, cv2.COLOR_BGR2RGB)
    #
    #     for ls in linestrings:
    #         all_pixels = np.array([mapping_color_rgb[int(p[1] * 1 / scaling_factor), int(p[0] * 1 / scaling_factor)] for p in ls.coords])
    #         mean = np.mean(all_pixels, axis=0)
    #         # diffs = [deltaE_cie76(rgb2lab(mean/255.0), c) for c in palette_labColor]
    #         diffs = [deltaE_ciede2000(rgb2lab(mean / 255.0), c) for c in palette_labColor]
    #         # diffs = [deltaE_ciede94(rgb2lab(mean/255.0), c) for c in palette_labColor]
    #
    #         # print(diffs)
    #         # print(np.argmin(diffs))
    #
    #         palette_color_index = np.argmin(diffs)
    #         linestrings_split_by_palette[palette_color_index].append(ls)
    # else:
    #     linestrings_split_by_palette[0] = linestrings


    # canvas = np.full([int(dimensions[0] * 10), int(dimensions[1] * 10), 3], 0 if INVERT_COLOR else 255, dtype=np.uint8)
    #
    # cv2.imwrite(
    #     # str(".." / Path("foo_" + Path(FILENAME_MAPPING_ANGLE).name)),
    #     str(args.output),
    #     # draw_line_image(canvas, [linestrings, linestrings_overlay, linestrings_contours], dimensions),
    #     draw_line_image(canvas, [linestrings, linestrings_overlay], dimensions),
    #     # draw_line_image(canvas, [linestrings_stencil], dimensions),
    #     # draw_line_image(canvas, [linestrings], dimensions),
    # )


    palette = [
        [255, 255, 255]
    ]

    palette = [
        [240, 126, 50],
        [0, 154, 194],
    ]

    linestrings_split_by_palette = [[] for _ in range(len(palette))]

    if len(palette) > 1:

        if len(palette) != mapping_color.shape[2]:
            raise Exception(f"Palette size mismatch: {args.mapping_color} has {mapping_color.shape[2]} color(s), palette has {len(palette)}")

        for ls in linestrings:
            all_pixels = np.array([mapping_color[int(p[1] * 1 / scaling_factor), int(p[0] * 1 / scaling_factor)] for p in ls.coords])

            mean = np.mean(all_pixels, axis=0)

            # closest
            # palette_color_index = np.argmax(mean)

            # weighted random selection
            palette_color_index = random.choices(range(len(palette)), mean)[0]

            linestrings_split_by_palette[palette_color_index].append(ls)
    else:
        linestrings_split_by_palette[0] = linestrings







    svg = SvgWriter(args.output, dimensions)
    svg.background_color = "#000000"

    layer_styles: dict[str, dict[str, str]] = {}

    for i, color in enumerate(palette):
        layer_styles[f"lines_{i}"] = {
            "fill": "none",
            "stroke": f"rgb({color[0]},{color[1]},{color[2]})",
            "stroke-width": "0.30",
            "fill-opacity": "1.0",
        }

    layer_styles["contours"] = {
        "fill": "none",
        "stroke": f"rgb({palette[0][0]},{palette[0][1]},{palette[0][2]})",
        "stroke-width": "0.30",
        "fill-opacity": "1.0",
    }

    layer_styles["overlay"] = {
        "fill": "none",
        "stroke": "yellow",
        "stroke-width": "0.30",
        "fill-opacity": "1.0",
    }

    for k, v in layer_styles.items():
        svg.add_style(k, v)

    svg.add("overlay", linestrings_overlay)
    svg.add("contours", linestrings_contours)

    for i, colored_linestrings in enumerate(linestrings_split_by_palette):
        svg.add(f"lines_{i}", colored_linestrings)

    svg.write()
    # try:
    #     convert_svg_to_png(svg, svg.dimensions[0] * 10)
    # except Exception as e:
    #     print(f"SVG to PNG conversion failed: {e}")
