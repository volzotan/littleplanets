import argparse
import itertools
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import rasterio
import toml
from pydantic import BaseModel
import numpy as np
import math
from shapely.geometry import LineString
from loguru import logger

from util.misc import visualize_linestrings, write_linestrings_to_npz, rotate_linestrings, split_linestring

VISUALIZE = False
DEBUG = True
DIR_DEBUG = Path("debug")

MIN_LINE_LENGTH = 0.05

class OverlayCoastlinesConfig(BaseModel):
    rotX: float = 0
    rotY: float = 0
    rotZ: float = 0

    radius: float = 1.0
    morph_kernel_size: int = 4
    simplify: float = 2
    segmentize_length: float = 0.01


def _read(input_path: Path) -> tuple[np.ndarray, Any, Any]:
    with rasterio.open(input_path) as src:
        return src.read(), src.crs, src.transform


def _remove_zero_meridian_lines(linestrings: list[LineString], raster: np.ndarray) -> list[LineString]:
    output = []

    for ls in linestrings:
        coords = np.array(ls.coords)
        mask = (coords[:, 0] == 0) | (coords[:, 0] == raster.shape[1] - 1)

        if np.any(mask):
            sep_indices = np.where(mask)[0]
            subarrays = np.split(coords, sep_indices + 1)
            output += [LineString(sub) for sub in subarrays if len(sub) >= 2]
        else:
            output.append(ls)

    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dem", type=Path, help="Elevation raster data in the (Geo)Tiff format (TIF)")
    parser.add_argument("--output", type=Path, default="overlay_coastlines.npz", help="Output filename (NPZ)")
    parser.add_argument("--config", type=Path, help="Configuration file (TOML)")
    parser.add_argument("--debug", action="store_true", default=DEBUG, help="Enable debug output")
    parser.add_argument("--visualize", action="store_true", default=VISUALIZE, help="Enable interactive visualization")
    args = parser.parse_args()

    config = OverlayCoastlinesConfig()
    if args.config is not None:
        if args.config.exists():
            with open(args.config, "r") as f:
                data = toml.load(f)
                config = OverlayCoastlinesConfig.model_validate(data)
        else:
            logger.warning("No config found, writing empty file")
            write_linestrings_to_npz(args.output, [])
            return

    blender_rotation = np.array([np.radians(c % 360) for c in [config.rotX, config.rotY, config.rotZ]])

    timer_start = datetime.now()

    dem, _, _ = _read(args.dem)
    dem = np.transpose(dem, (1, 2, 0))

    mask = np.zeros_like(dem, dtype=np.uint8)
    mask[dem > 1] = 255

    if config.morph_kernel_size is not None and config.morph_kernel_size >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.morph_kernel_size, config.morph_kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # shift center
    mask = np.roll(mask, -int(mask.shape[1] / 2), axis=1)  # by 1/2 (adjust for 0 meridian in center)

    linestrings = []
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        linestrings.append(LineString(contour[:, 0, :]))

    linestrings = _remove_zero_meridian_lines(linestrings, mask)
    linestrings = [ls.simplify(config.simplify) for ls in linestrings]

    def _project_to_sphere(ls: LineString, width: float, height: float, radius: float) -> LineString:
        coords = np.array(ls.coords)

        lats = coords[:, 1] / height * math.pi + math.pi / 2
        lons = coords[:, 0] / width * math.tau + math.pi / 2

        c = np.zeros([coords.shape[0], 3])
        c[:, 0] = radius * np.cos(lats) * np.cos(lons)
        c[:, 1] = radius * np.cos(lats) * np.sin(lons)
        c[:, 2] = radius * np.sin(lats)

        return LineString(c)

    linestrings = [_project_to_sphere(ls, mask.shape[1], mask.shape[0], radius=config.radius) for ls in linestrings]

    # FILTER

    linestrings = [ls for ls in linestrings if ls.length > MIN_LINE_LENGTH]

    # SPLIT

    linestrings = itertools.chain.from_iterable([split_linestring(ls, config.segmentize_length) for ls in linestrings])

    # ROTATE

    linestrings = rotate_linestrings(linestrings, *blender_rotation)

    # VISUALIZE

    if args.debug:
        cv2.imwrite(str(DIR_DEBUG / "overlay_coastlines_threshold.png"), mask)

    if args.visualize:
        plotter = visualize_linestrings(linestrings)
        plotter.camera_position = "xy"
        plotter.show()

    # EXPORT

    write_linestrings_to_npz(args.output, linestrings)
    print(f"generated overlay_coastlines in {(datetime.now() - timer_start).total_seconds():5.2f}s")


if __name__ == "__main__":
    main()
