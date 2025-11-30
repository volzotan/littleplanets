import argparse
from pathlib import Path

import numpy as np
import cv2

from src.util.misc import visualize

RESIZE_FACTOR = 0.05

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("raytrace", type=Path, default="raytrace.npy", help="Raytracing distance raster (NPY)")
    args = parser.parse_args()

    img_pxpos = np.load(args.raytrace)

    new_size = (int(img_pxpos.shape[1] * RESIZE_FACTOR), int(img_pxpos.shape[0] * RESIZE_FACTOR))
    img_pxpos = cv2.resize(img_pxpos, new_size)

    img_pxpos = np.reshape(img_pxpos, [-1, 3])

    print(f"shape: {img_pxpos.shape}")

    visualize([], [], [img_pxpos]).show()