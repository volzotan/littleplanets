import numpy as np
import tifffile

data = np.load("raytracing.npy")
tifffile.imwrite("raytracing.tif", data)
