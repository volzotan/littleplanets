
import os
from pathlib import Path

import Imath
import OpenEXR as exr
import rasterio
import openexr_numpy

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import cv2
import numpy as np

def load_raster(filename: Path) -> np.ndarray:
    with rasterio.open(filename) as dataset:
        data = dataset.read()
        return data


# img_depth = cv2.imread("blender_output_depth.tif", cv2.IMREAD_GRAYSCALE)
# img_depth = cv2.imread("blender_output_depth.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
# img_depth = cv2.imread("blender_output_depth.exr", cv2.IMREAD_GRAYSCALE)
# img_depth = cv2.imread("/tmp/Image0001.tif", cv2.IMREAD_GRAYSCALE)

# img_depth = cv2.imread("/tmp/Depth0001.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
# print(img_depth)
# img_depth = cv2.imread("/tmp/Depth0001.exr", cv2.IMREAD_ANYCOLOR)
# print(img_depth)
# img_depth = cv2.imread("/tmp/Depth0001.exr", cv2.IMREAD_ANYDEPTH)
# print(img_depth)
# img_depth = cv2.imread("/tmp/Depth0001.exr")
# print(img_depth)


# img_depth = cv2.imread("/tmp/Depth0001.tif")
# print(np.min(img_depth))
# print(np.max(img_depth))
# print(img_depth.shape)
# print(img_depth[500, 20])

# def read_depth_exr_file(filepath: Path):
#     exrfile = exr.InputFile(filepath.as_posix())
#     raw_bytes = exrfile.channel(0, Imath.PixelType(Imath.PixelType.FLOAT))
#     depth_vector = np.frombuffer(raw_bytes, dtype=np.float32)
#     height = exrfile.header()['displayWindow'].max.y + 1 - exrfile.header()['displayWindow'].min.y
#     width = exrfile.header()['displayWindow'].max.x + 1 - exrfile.header()['displayWindow'].min.x
#     depth_map = np.reshape(depth_vector, (height, width))
#     return depth_map

# print(read_depth_exr_file(Path("/tmp/Depth0001.exr")))


# (R,G,B) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B") ]

# print(img_depth)

# img_depth = load_raster("/tmp/Image0001.tif")

# print(np.min(img_depth))
# print(np.max(img_depth))
# print(img_depth.shape)


img_depth = openexr_numpy.imread("/tmp/Depth0001.exr", "V")
print(np.min(img_depth))
print(np.max(img_depth))
print(img_depth.shape)