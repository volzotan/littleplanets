import argparse
from pathlib import Path

import numpy as np
from shapely import LineString

from util.misc import write_linestrings_to_npz

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("overlay", default="overlay.npz", type=Path, help="Input overlay lines [NPZ]")
    parser.add_argument("overlay_visible", default="overlay_visible.npz", type=Path, help="Input overlay visiblity info [NPZ]")
    parser.add_argument("--output", type=Path, default="overlay_cropped.npz", help="Output filename [NPZ]")
    args = parser.parse_args()

    overlay_npz = list(np.load(args.overlay).values())
    overlay_visible_npz = list(np.load(args.overlay_visible).values())

    linestrings = []
    for i in range(len(overlay_npz)):
        points = overlay_npz[i]
        visibility = overlay_visible_npz[i]

        line = []
        for j in range(len(points)):
            if visibility[j] == True:
                line.append(points[j])
            else:
                if len(line) >= 2:
                    linestrings.append(LineString(line))
                line = []

        if len(line) >= 2:
            linestrings.append(LineString(line))

    print(f"Input Linestrings: {len(overlay_npz)}")
    print(f"Output Linestrings: {len(linestrings)}")

    write_linestrings_to_npz(args.output, linestrings)
