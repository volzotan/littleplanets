import os
from pathlib import Path

import numpy as np
import tifffile

BUILD_DIR = Path("build")

data = np.load(BUILD_DIR / "raytrace.npy")
tifffile.imwrite(BUILD_DIR / "raytrace.tif", data)
