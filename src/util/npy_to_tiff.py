import os
from pathlib import Path

import numpy as np
import tifffile

BUILD_DIR = Path("build_earth")
DEBUG_DIR = Path("debug")


def convert(data):
    mask = np.isnan(np.sum(data, axis=2))

    png = np.full([data.shape[0], data.shape[1], 3], 255, dtype=np.uint8)
    png[mask] = [0, 0, 0]

    return png


data = np.load(BUILD_DIR / "raytrace_clouds.npy")
tifffile.imwrite(DEBUG_DIR / "raytrace.tif", data)

data = np.load(BUILD_DIR / "raytrace_clouds.npy")
tifffile.imwrite(DEBUG_DIR / "raytrace.png", convert(data))

data = np.load(BUILD_DIR / "raytrace_clouds_backface.npy")
tifffile.imwrite(DEBUG_DIR / "raytrace_backface.png", convert(data))
