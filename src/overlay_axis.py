import argparse
from pathlib import Path

import shapely
from shapely.geometry import LineString
import numpy as np
import math
from util.hershey import HersheyFont
from util.misc import write_linestrings_to_npz, _rotate_linestrings, visualize_linestrings

VISUALIZE = False

AXIS_EXTENT = 1.3

def _segmentize_z(start_z: float, end_z: float, max_length_segment: float) -> np.ndarray:
    return np.linspace(start_z, end_z, num=math.ceil((end_z-start_z)/max_length_segment), endpoint=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rotX", type=float, default=0, help="rotation X in degrees [float]")
    parser.add_argument("--rotY", type=float, default=0, help="rotation Y in degrees [float]")
    parser.add_argument("--rotZ", type=float, default=0, help="rotation Z in degrees [float]")
    parser.add_argument("--output", type=Path, default="overlay.npz", help="Output filename [NPZ]")
    args = parser.parse_args()

    blender_rotation = np.array([np.radians(c) for c in [args.rotX, args.rotY, args.rotZ]])

    font = HersheyFont(font_file=Path(HersheyFont.DEFAULT_FONT))
    linestrings = []

    # CREATE AXIS

    # shapely's segmentize uses GEOS, which fails on LineStrings that are parallel to the Z axis
    zs = _segmentize_z(-AXIS_EXTENT, AXIS_EXTENT, 1e-2)
    line_coords = np.zeros([zs.shape[0], 3])
    line_coords[:, 2] = zs

    linestrings.append(LineString(line_coords))

    # ROTATE

    linestrings = _rotate_linestrings(linestrings, *blender_rotation)

    # VISUALIZE

    if VISUALIZE:
        visualize_linestrings(linestrings).show()
        exit()

    # EXPORT

    write_linestrings_to_npz(args.output, linestrings)
