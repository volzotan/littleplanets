import argparse
import datetime
import itertools
import tomllib
from pathlib import Path
import math
import random

import cv2

import numpy as np
import shapely
import shapely.ops
from pydantic import BaseModel, Field
from shapely import LineString, MultiLineString, Geometry, Point

from loguru import logger

from util import flowlines
from svgwriter import SvgWriter

DIR_DEBUG = Path("debug")

PSEUDO_RANDOM_SEED = "littleplanets"

OVERLAY_STENCIL_CUT_DISTANCE = 4
CUTOUT_STENCIL_CUT_DISTANCE = 1
SEGMENTIZE_MAX_LENGTH = 15.0  # mm


class CombineConfig(BaseModel):
    dimensions: tuple[int, int] = (1000, 1000)

    colors: list[list[int]] = [[255, 255, 255]]
    layer_colors: list[list[int]] = []

    invert_background: bool = False

    # Blurring kernel size, percentage of raster size(float)
    blur_color_kernel_size_perc: float = Field(0, ge=0)


def _project_linestring(ls: LineString, P: np.ndarray, scaling_factor: float) -> LineString:
    xyz = shapely.get_coordinates(ls, include_z=True)
    coordinates = np.hstack([xyz, np.full([xyz.shape[0], 1], 1)])  # [x, y, z, w=1]
    coordinates = (P @ coordinates.T).T
    coordinates = coordinates[:, 0:2] / coordinates[:, 2][:, np.newaxis]  # P * [X Y 1] = [x y w], then divide x and y by w
    coordinates = coordinates * scaling_factor
    return LineString(coordinates)


def _blur_raster(raster: np.ndarray, perc: float) -> np.ndarray:
    kernel_size = int(max(*raster.shape) * (perc / 100.0))
    if kernel_size > 1:
        return cv2.blur(raster, (kernel_size, kernel_size))
    else:
        return raster


def _split_linestring(ls: LineString, max_length: float) -> list[LineString]:
    ls = shapely.segmentize(ls, max_length)
    coords = list(ls.coords)

    split_ls = []

    if len(coords) < 2:
        return []

    candidate = [coords[0]]
    candidate_length = 0

    for i in range(1, len(coords)):
        p1 = coords[i - 1]
        p2 = coords[i]
        p1_p2_length = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        if candidate_length + p1_p2_length > max_length:
            split_ls.append(LineString(candidate))
            candidate = [coords[i - 1], coords[i]]
            candidate_length = p1_p2_length
        else:
            candidate.append(coords[i])
            candidate_length += p1_p2_length

        if i == len(coords) - 1:
            split_ls.append(LineString(candidate))

    return split_ls


def _match_linestrings_to_palette(
    linestrings: list[LineString], mapping: np.ndarray, palette: np.ndarray, scaling_factor: float = 1.0
) -> list[list[LineString]]:
    linestrings_split_by_palette: list[list[LineString]] = [[] for _ in range(len(palette))]

    if len(palette) == 1:
        return [linestrings]

    for ls in linestrings:
        if len(ls.coords) < 2:
            continue

        all_pixels = np.nan_to_num(np.array([mapping[int(p[1] * 1 / scaling_factor), int(p[0] * 1 / scaling_factor)] for p in ls.coords]))
        mean = np.mean(all_pixels, axis=0)
        palette_color_index = 0

        if np.sum(mean) > 0.1:
            palette_color_index = random.choices(range(palette.shape[0]), mean)[0]

        linestrings_split_by_palette[palette_color_index].append(ls)

    return linestrings_split_by_palette


def _check_linestrings_within_bounds(linestrings: list[LineString], xmin: float, ymin: float, xmax: float, ymax: float) -> list[LineString]:
    checked_linestrings = []
    box = shapely.box(xmin, ymin, xmax - 1, ymax - 1)

    for ls in linestrings:
        if not ls.is_valid:
            continue

        if ls.within(box):
            checked_linestrings.append(ls)
        else:
            g = shapely.intersection(box, ls)
            match g:
                case LineString():
                    if len(g.coords) >= 2:
                        checked_linestrings.append(g)
                case MultiLineString():
                    for sg in g.geoms:
                        checked_linestrings.append(sg)
                case _:
                    logger.warning(f"unexpected geometry: {g}")

    logger.debug(f"check_linestrings_within_bounds, failed linestrings: {len(linestrings) - len(checked_linestrings)}")

    return checked_linestrings


def _cut(objects: list[LineString], tools: list[Geometry], buffer_radius: float) -> list[LineString]:
    linestrings_cut = []

    if len(tools) == 0:
        return objects

    stencil = shapely.ops.unary_union(tools).buffer(buffer_radius)

    for ls in objects:
        cut = shapely.difference(ls, stencil)

        match cut:
            case LineString():
                linestrings_cut.append(cut)
            case MultiLineString():
                for g in cut.geoms:
                    linestrings_cut.append(g)
            case _:
                print(f"unexpected geometry: {cut}")

    return linestrings_cut


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--mapping-color", type=Path, help="Mapping color (NPY)")
    parser.add_argument("--mapping-background", type=Path, help="Mapping background (PNG)")

    parser.add_argument("--hatchlines", type=Path, help="Hatchline linestrings (NPZ)")
    parser.add_argument("--cutouts", type=Path, nargs="*", default=[], help="Cutout linestrings, multiple filenames possible (NPZ)")
    parser.add_argument("--overlays", type=Path, nargs="*", default=[], help="Overlay linestrings, multiple filenames possible (NPZ)")
    parser.add_argument("--projection-matrix", type=Path, default=None, help="3x4 projection matrix (NPY)")
    parser.add_argument("--contours", type=Path, default=None, help="Contour linestrings (NPZ)")

    parser.add_argument("--config", type=Path, default=None, help="Config file (TOML)")
    parser.add_argument("--output", type=Path, default="littleplanet.svg", help="Output filename (SVG)")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug (image) output")
    parser.add_argument("--suffix", type=str, default="", help="Filename suffix to be appended to all (debug) output")

    args = parser.parse_args()

    config = CombineConfig()
    if args.config is not None and args.config.exists():
        with open(args.config, "rb") as f:
            data = tomllib.load(f)
            config = CombineConfig.model_validate(data)
    else:
        logger.warning(f"No config file not found")

    random.seed(PSEUDO_RANDOM_SEED)

    mapping_background = (
        cv2.imread(args.mapping_background, cv2.IMREAD_GRAYSCALE)
        if args.mapping_background is not None
        else np.zeros(list(config.dimensions), dtype=np.uint8)
    )

    mapping_color = None
    if args.mapping_color is not None:
        if args.mapping_color.suffix == ".npy":
            mapping_color = np.load(args.mapping_color)
            mapping_color[mapping_background > 0] = np.nan
        else:
            mapping_color = cv2.imread(args.mapping_color)
    else:
        mapping_color = np.zeros(list(config.dimensions) + [1], dtype=np.uint8)

    mapping_color = _blur_raster(mapping_color, config.blur_color_kernel_size_perc)

    scaling_factor = config.dimensions[0] / mapping_background.shape[1]
    linestrings = []

    if args.hatchlines is not None:
        hatchlines_npz = np.load(args.hatchlines)
        linestrings = [LineString(arr) for arr in hatchlines_npz.values()]
        # linestrings = [shapely.affinity.scale(ls, xfact=scaling_factor, yfact=scaling_factor, origin=(0, 0)) for ls in linestrings]

    linestrings_cutouts = []

    if len(args.cutouts) > 0:
        P = np.load(args.projection_matrix)
        for cutout_path in args.cutouts:
            cutout_npz = np.load(cutout_path)
            cutout_ls = [LineString(arr) for arr in cutout_npz.values()]
            cutout_ls = [_project_linestring(l, P, scaling_factor) for l in cutout_ls]
            linestrings_cutouts += cutout_ls

    linestrings_overlays: list[list[LineString]] = []

    P = np.identity(3)
    if args.projection_matrix is not None:
        P = np.load(args.projection_matrix)

    if len(args.overlays) > 0:
        for overlay_path in args.overlays:
            overlay_npz = np.load(overlay_path)
            overlay_ls = [LineString(arr) for arr in overlay_npz.values()]
            if len(overlay_ls) > 0 and overlay_ls[0].has_z:
                overlay_ls = [_project_linestring(l, P, scaling_factor) for l in overlay_ls]
            overlay_ls = _check_linestrings_within_bounds(overlay_ls, 0, 0, config.dimensions[0], config.dimensions[1])
            linestrings_overlays.append(overlay_ls)

    # TODO: contours don't need to be projected, but they need to be scaled (currently missing!)
    linestrings_contours = []
    if args.contours is not None:
        contours_npz = np.load(args.contours)
        linestrings_contours = [LineString(arr) for arr in contours_npz.values()]
        linestrings_contours = [shapely.affinity.scale(ls, xfact=scaling_factor, yfact=scaling_factor, origin=(0, 0)) for ls in linestrings_contours]

    # merge contours with hatchlines
    linestrings_contours_split = itertools.chain.from_iterable([_split_linestring(ls, SEGMENTIZE_MAX_LENGTH) for ls in linestrings_contours])
    linestrings += linestrings_contours_split

    # cut buffered overlay from hatched linestrings
    timer_start = datetime.datetime.now()
    linestrings = _cut(linestrings, linestrings_cutouts, CUTOUT_STENCIL_CUT_DISTANCE / 2)
    linestrings = _cut(linestrings, [ls for overlay_ls in linestrings_overlays for ls in overlay_ls], OVERLAY_STENCIL_CUT_DISTANCE / 2)
    logger.debug(f"Combination stencil time: {(datetime.datetime.now() - timer_start).total_seconds():5.2f}s")

    # cut each overlay from all underlying ones
    timer_start = datetime.datetime.now()
    combined_stencil = Point()
    for i in reversed(range(len(linestrings_overlays))):
        if not combined_stencil.is_empty:
            linestrings_overlays[i] = _cut(linestrings_overlays[i], [combined_stencil], OVERLAY_STENCIL_CUT_DISTANCE / 2)
        combined_stencil = shapely.unary_union([combined_stencil] + linestrings_overlays[i])
    logger.debug(f"Overlay cascade stencil time: {(datetime.datetime.now() - timer_start).total_seconds():5.2f}s")

    # Coloring
    palette = np.array(config.colors, dtype=int)
    palette = np.delete(palette, np.where(np.min(palette, axis=1) < 0), axis=0)  # remove invalid palette colors
    palette = palette.astype(np.uint8)

    if len(palette) != mapping_color.shape[2]:
        raise Exception(f"Palette size mismatch: {args.mapping_color} has {mapping_color.shape[2]} color(s), palette has {len(palette)}")

    svg = SvgWriter(args.output, config.dimensions)
    if config.invert_background:
        svg.background_color = "#000000"

    layer_styles: dict[str, dict[str, str]] = {}

    for i, color in enumerate(palette):
        layer_styles[f"lines_{i}"] = {
            "fill": "none",
            "stroke": f"rgb({color[0]},{color[1]},{color[2]})",
            "stroke-width": "0.30",
            "fill-opacity": "1.0",
        }

    for io, overlay_ls in enumerate(linestrings_overlays):
        if io < len(config.layer_colors) and len(config.layer_colors[io]) == 3:
            overlay_color = config.layer_colors[io]

            layer_styles[f"overlay_{io}"] = {
                "fill": "none",
                "stroke": f"rgb({overlay_color[0]},{overlay_color[1]},{overlay_color[2]})",
                "stroke-width": "0.30",
                "fill-opacity": "1.0",
            }

            svg.add(f"overlay_{io}", overlay_ls)
        else:
            for ic, color in enumerate(palette):
                layer_styles[f"overlay_color_{ic}"] = {
                    "fill": "none",
                    "stroke": f"rgb({color[0]},{color[1]},{color[2]})",
                    "stroke-width": "0.30",
                    "fill-opacity": "1.0",
                }

            linestrings_overlay_palette = _match_linestrings_to_palette(overlay_ls, mapping_color, palette, scaling_factor)
            for ic, colored_linestrings in enumerate(linestrings_overlay_palette):
                svg.add(f"overlay_color_{ic}", colored_linestrings)

    # add frame
    frame_color = [0, 0, 0] if not config.invert_background else [255, 255, 255]
    frame_length = 50
    width, height = config.dimensions
    linestrings_frame = [
        LineString([[0, height - frame_length], [0, height], [frame_length, height]]),
        LineString([[width - frame_length, height], [width, height], [width, height - frame_length]]),
        LineString([[width, frame_length], [width, 0], [width - frame_length, 0]]),
        LineString([[frame_length, 0], [0, 0], [0, frame_length]]),
        LineString([[width/2-frame_length/2, 0], [width/2+frame_length/2, 0]]),
        LineString([[width/2-frame_length/2, height], [width/2+frame_length/2, height]]),
        LineString([[0, height/2-frame_length/2], [0, height/2+frame_length/2]]),
        LineString([[width, height/2-frame_length/2], [width, height/2+frame_length/2]]),
    ]
    layer_styles["frame"] = {
        "fill": "none",
        "stroke": f"rgb({frame_color[0]},{frame_color[1]},{frame_color[2]})",
        "stroke-width": "0.50",
        "fill-opacity": "1.0",
    }
    svg.add("frame", linestrings_frame)

    # Build SVG

    for k, v in layer_styles.items():
        svg.add_style(k, v)

    linestrings_palette = _match_linestrings_to_palette(linestrings, mapping_color, palette, scaling_factor)
    for i, colored_linestrings in enumerate(linestrings_palette):
        svg.add(f"lines_{i}", colored_linestrings)

    svg.write()


if __name__ == "__main__":
    main()
