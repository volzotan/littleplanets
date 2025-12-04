import argparse
from pathlib import Path

import numpy as np
from shapely.geometry import LineString

from src.util.misc import visualize, visualize_linestrings

RESIZE_FACTOR = 0.05

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path, help="Linestrings (NPZ)")
    args = parser.parse_args()

    npz = np.load(args.npz)
    linestrings = [LineString(e) for e in npz.values()]

    print(f"loaded {len(linestrings)} LineStrings")
    print(linestrings[0])

    visualize_linestrings(linestrings).show()
