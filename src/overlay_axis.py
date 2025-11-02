import argparse
from pathlib import Path

import shapely
import toml
from pydantic import BaseModel
from shapely.geometry import LineString
import numpy as np
import math

from util.misc import dash_linestring
from util.hershey import HersheyFont
from util.misc import write_linestrings_to_npz, rotate_linestrings, visualize_linestrings


def _segmentize_z(start_z: float, end_z: float, max_length_segment: float) -> np.ndarray:
    return np.linspace(start_z, end_z, num=math.ceil((end_z - start_z) / max_length_segment), endpoint=True)


class OverlayAxisConfig(BaseModel):
    rotX: float = 0
    rotY: float = 0
    rotZ: float = 0
    max_length_segment: float = 1e-3

    axis_extent: float = 1.3
    dash_length: float = 0.02
    pause_length: float = 0.02


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output", type=Path, default="overlay.npz", help="Output filename [NPZ]")
    parser.add_argument("--config", type=Path, help="Configuration file [TOML]")
    parser.add_argument("--visualize", action="store_true", default=False, help="Enable interactive visualization")

    args = parser.parse_args()

    config = OverlayAxisConfig()
    if args.config is not None:
        with open(args.config, "r") as f:
            data = toml.load(f)
            config = OverlayAxisConfig.model_validate(data)

    blender_rotation = np.array([np.radians(c) for c in [config.rotX, config.rotY, config.rotZ]])

    font = HersheyFont(font_file=Path(HersheyFont.DEFAULT_FONT))
    linestrings = []

    # CREATE AXIS

    # shapely's segmentize uses GEOS, which fails on LineStrings that are parallel to the Z axis
    zs = _segmentize_z(-config.axis_extent, config.axis_extent, config.max_length_segment)
    line_coords = np.zeros([zs.shape[0], 3])
    line_coords[:, 2] = zs

    ls = LineString(line_coords)
    linestrings += dash_linestring(ls, config.dash_length, config.pause_length)

    # ROTATE

    linestrings = rotate_linestrings(linestrings, *blender_rotation)

    # VISUALIZE

    if args.visualize:
        visualize_linestrings(linestrings).show()
        exit()

    # EXPORT

    write_linestrings_to_npz(args.output, linestrings)
