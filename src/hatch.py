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
from shapely import LineString, MultiLineString

from loguru import logger

from util.misc import write_linestrings_to_npz
from util import flowlines
from svgwriter import SvgWriter

DIR_DEBUG = Path("debug")

PSEUDO_RANDOM_SEED = "littleplanets"

OVERLAY_STENCIL_CUT_DISTANCE = 4
CUTOUT_STENCIL_CUT_DISTANCE = 1


class HatchConfig(BaseModel):
    dimensions: tuple[int, int] = (1000, 1000)
    invert_background: bool = False

    colors: list[list[int]] = [[255, 255, 255]]

    # Blurring kernel size, percentage of raster size(float)
    blur_angle_kernel_size_perc: float = Field(0, ge=0)
    blur_distance_kernel_size_perc: float = Field(0, ge=0)

    flowlines_line_distance: tuple[float, float] = (0.8, 10)
    flowlines_line_max_length: tuple[float, float] = (5, 25)
    flowlines_line_distance_end_factor: float = Field(0.25, ge=0, le=1.0)
    flowlines_max_angle_discontinuity: float = Field(math.pi / 12, gt=0, lt=math.tau)


def _blur_raster(raster: np.ndarray, perc: float) -> np.ndarray:
    kernel_size = int(max(*raster.shape) * (perc / 100.0))
    if kernel_size > 1:
        return cv2.blur(raster, (kernel_size, kernel_size))
    else:
        return raster


def _check_linestrings_within_bounds(linestrings: list[LineString], xmin: float, ymin: float, xmax: float, ymax: float) -> list[LineString]:
    checked_linestrings = []
    box = shapely.box(xmin, ymin, xmax - 1, ymax - 1)

    for ls in linestrings:
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

    return checked_linestrings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mapping-angle", type=Path, help="Mapping angle (PNG)")
    parser.add_argument("--mapping-distance", type=Path, help="Mapping distance (PNG)")
    parser.add_argument("--mapping-line-length", type=Path, help="Mapping line length (PNG)")
    parser.add_argument("--mapping-background", type=Path, help="Mapping background (PNG)")

    parser.add_argument("--config", type=Path, default=None, help="Config file (TOML)")

    parser.add_argument("--output", type=Path, default="hatchlines.npz", help="Output filename (NPZ)")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug (image) output")
    parser.add_argument("--suffix", type=str, default="", help="Filename suffix to be appended to all (debug) output")

    args = parser.parse_args()

    config = HatchConfig()
    if args.config is not None:
        with open(args.config, "rb") as f:
            data = tomllib.load(f)
            config = HatchConfig.model_validate(data)

    mapping_angle = cv2.imread(args.mapping_angle, cv2.IMREAD_GRAYSCALE)  # uint8 image must be centered around 128 to deal with negative values
    mapping_distance = cv2.imread(args.mapping_distance, cv2.IMREAD_GRAYSCALE) if args.mapping_distance is not None else np.zeros_like(mapping_angle)
    mapping_line_length = (
        cv2.imread(args.mapping_line_length, cv2.IMREAD_GRAYSCALE) if args.mapping_line_length is not None else np.zeros_like(mapping_angle)
    )
    mapping_background = (
        cv2.imread(args.mapping_background, cv2.IMREAD_GRAYSCALE) if args.mapping_background is not None else np.zeros_like(mapping_angle)
    )

    mapping_angle = _blur_raster(mapping_angle, config.blur_angle_kernel_size_perc)
    mapping_distance = _blur_raster(mapping_distance, config.blur_distance_kernel_size_perc)

    scaling_factor = config.dimensions[0] / mapping_angle.shape[1]

    mask = mapping_background == 0
    mapping_distance = ((mapping_distance - np.min(mapping_distance[mask])) / np.ptp(mapping_distance[mask]) * 255).astype(np.uint8)
    mapping_distance[~mask] = 0

    if config.invert_background:  # white ink on black paper, invert grayscale image
        mapping_distance = ~mapping_distance

    if args.debug:
        # cv2.imwrite(str(DIR_DEBUG / f"hatch_mapping_color{args.suffix}.png"), mapping_color) # doesn't make sense to export mapping_color if it's palette colors
        cv2.imwrite(str(DIR_DEBUG / f"hatch_mapping_angle{args.suffix}.png"), mapping_angle)
        cv2.imwrite(str(DIR_DEBUG / f"hatch_mapping_distance{args.suffix}.png"), mapping_distance)

    mappings = [
        mapping_distance,
        mapping_angle,
        mapping_line_length,
        mapping_background,
    ]

    flowlines_config = flowlines.FlowlineHatcherConfig()
    flowlines_config.LINE_DISTANCE = config.flowlines_line_distance
    flowlines_config.LINE_MAX_LENGTH = config.flowlines_line_max_length
    flowlines_config.LINE_DISTANCE_END_FACTOR = config.flowlines_line_distance_end_factor
    flowlines_config.MAX_ANGLE_DISCONTINUITY = config.flowlines_max_angle_discontinuity

    # hatcher = flowlines.FlowlineHatcher(dimensions, config, *mappings, exclusion_points=exclusion_points + contour_points)
    # hatcher = flowlines.FlowlineHatcher(dimensions, config, *mappings, exclusion_points=exclusion_points, initial_seed_points=contour_points)
    # hatcher = flowlines.FlowlineHatcher(config.dimensions, flowlines_config, *mappings, exclusion_points=exclusion_points)
    hatcher = flowlines.FlowlineHatcher(config.dimensions, flowlines_config, *mappings)
    linestrings: list[LineString] = hatcher.hatch()
    linestrings = [shapely.simplify(l, 0.01) for l in linestrings]

    print(f"num linestrings: {len(linestrings)}")

    write_linestrings_to_npz(args.output, linestrings, include_z=False)
