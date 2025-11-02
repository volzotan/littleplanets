import argparse
from pathlib import Path

import toml
from pydantic import BaseModel
from shapely.geometry import LineString
import numpy as np
import math
from util.hershey import HersheyFont
from util.misc import write_linestrings_to_npz, rotate_linestrings, visualize_linestrings

VISUALIZE = False

GRID_NUM_SEGMENTS = 360


class OverlayGridConfig(BaseModel):
    rotX: float = 0
    rotY: float = 0
    rotZ: float = 0
    grid_lines_lat: int = 0
    grid_lines_lon: int = 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default="overlay.npz", help="Output filename [NPZ]")
    parser.add_argument("--config", type=Path, help="Configuration file [TOML]")

    args = parser.parse_args()

    config = OverlayGridConfig()
    if args.config is not None:
        with open(args.config, "r") as f:
            data = toml.load(f)
            config = OverlayGridConfig.model_validate(data)

    # PLY exported from Blender is already correctly rotated with regard to Z axis up
    # but Lat/Lon needs to be adjusted for any additional rotation
    BLENDER_ROTATION = np.array([np.radians(c) for c in [config.rotX, config.rotY, config.rotZ]])

    font = HersheyFont(font_file=Path(HersheyFont.DEFAULT_FONT))
    linestrings = []

    # CREATE GRID

    points = [math.tau * i / GRID_NUM_SEGMENTS for i in range(GRID_NUM_SEGMENTS)]
    points = [[0, math.cos(angle), math.sin(angle)] for angle in points]
    points = points + [points[0]]
    for i in range(config.grid_lines_lat):
        linestrings += rotate_linestrings([LineString(points)], 0, 0, math.pi * i / config.grid_lines_lat)

    lons = [1.0 / (config.grid_lines_lon + 1) * (i + 1) for i in range(config.grid_lines_lon)]
    for lon in lons:
        z = lon * 2 - 1
        y = math.sqrt(1 - z**2)
        points = [math.tau * i / GRID_NUM_SEGMENTS for i in range(GRID_NUM_SEGMENTS)]
        points = [[math.cos(angle) * y, math.sin(angle) * y, z] for angle in points]
        points = points + [points[0]]
        linestrings.append(LineString(points))

    # ROTATE

    linestrings = rotate_linestrings(linestrings, *BLENDER_ROTATION)

    # VISUALIZE

    if VISUALIZE:
        visualize_linestrings(linestrings).show()
        exit()

    # EXPORT

    write_linestrings_to_npz(args.output, linestrings)


if __name__ == "__main__":
    main()
