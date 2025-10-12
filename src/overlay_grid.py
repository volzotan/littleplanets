import argparse
from pathlib import Path

from shapely.geometry import LineString
import numpy as np
import math
from hershey import HersheyFont
from util.misc import write_linestrings_to_npz, _rotate_linestrings, visualize_linestrings

VISUALIZE = False

GRID_NUM_SEGMENTS = 360


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rotX", type=float, default=0, help="rotation X in degrees [float]")
    parser.add_argument("--rotY", type=float, default=0, help="rotation Y in degrees [float]")
    parser.add_argument("--rotZ", type=float, default=0, help="rotation Z in degrees [float]")
    parser.add_argument("--grid-num-lat", type=int, default=0, help="number of latitude grid lines")
    parser.add_argument("--grid-num-lon", type=int, default=0, help="number of longitude grid lines")
    parser.add_argument("--output", type=Path, default="overlay.npz", help="Output filename [NPZ]")
    args = parser.parse_args()

    # PLY exported from Blender is already correctly rotated with regard to Z axis up
    # but Lat/Lon needs to be adjusted for any additional rotation
    BLENDER_ROTATION = np.array([np.radians(c) for c in [args.rotX, args.rotY, args.rotZ]])

    font = HersheyFont(font_file=Path(HersheyFont.DEFAULT_FONT))
    linestrings = []

    # CREATE GRID

    grid_num_lat_lines = args.grid_num_lat
    grid_num_lon_lines = args.grid_num_lon

    points = [math.tau * i / GRID_NUM_SEGMENTS for i in range(GRID_NUM_SEGMENTS)]
    points = [[0, math.cos(angle), math.sin(angle)] for angle in points]
    points = points + [points[0]]
    for i in range(grid_num_lat_lines):
        linestrings += _rotate_linestrings([LineString(points)], 0, 0, math.pi * i / grid_num_lat_lines)

    lons = [1.0 / (grid_num_lon_lines + 1) * (i + 1) for i in range(grid_num_lon_lines)]
    for lon in lons:
        z = lon * 2 - 1
        y = math.sqrt(1 - z**2)
        points = [math.tau * i / GRID_NUM_SEGMENTS for i in range(GRID_NUM_SEGMENTS)]
        points = [[math.cos(angle) * y, math.sin(angle) * y, z] for angle in points]
        points = points + [points[0]]
        linestrings.append(LineString(points))

    # ROTATE

    linestrings = _rotate_linestrings(linestrings, *BLENDER_ROTATION)

    # VISUALIZE

    if VISUALIZE:
        visualize_linestrings(linestrings).show()
        exit()

    # EXPORT

    write_linestrings_to_npz(args.output, linestrings)
