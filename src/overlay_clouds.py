import argparse
from pathlib import Path

import cv2
import shapely
import toml
from pydantic import BaseModel, Field
from shapely.geometry import LineString
import numpy as np
import math

from util.misc import dash_linestring, rotate_points, rotate_points_inv
from util.hershey import HersheyFont
from util.misc import write_linestrings_to_npz, rotate_linestrings, visualize_linestrings

DEBUG = True
DIR_DEBUG = Path("debug")

class OverlayCloudsConfig(BaseModel):
    rotX: float = 0
    rotY: float = 0
    rotZ: float = 0

    # image_filename: Path
    #
    # flowlines_line_distance: tuple[float, float] = (0.8, 10)
    # flowlines_line_max_length: tuple[float, float] = (5, 25)
    # flowlines_line_distance_end_factor: float = Field(0.25, ge=0, le=1.0)
    # flowlines_max_angle_discontinuity: float = Field(math.pi / 12, gt=0, lt=math.tau)

def _map(raster: np.ndarray, lat: float, lon: float) -> np.ndarray:
    height, width = raster.shape

    y = int((lat / math.pi) * height)
    x = int((lon / math.tau) * width)

    return raster[y, x]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("normals", type=Path, default="normals.exr", help="Normals (EXR)")
    parser.add_argument("raytrace", type=Path, default="raytrace.npy", help="Raytracing distance raster (NPY)")
    parser.add_argument("cloud-cover", type=Path, default="image.tif", help="Cloud grayscale coverage map (TIFF)")
    parser.add_argument("--output", type=Path, default="build", help="Output directory")
    parser.add_argument("--config", type=Path, help="Configuration file [TOML]")
    parser.add_argument("--visualize", action="store_true", default=False, help="Enable interactive visualization")
    args = parser.parse_args()

    config = OverlayCloudsConfig()
    if args.config is not None:
        with open(args.config, "r") as f:
            data = toml.load(f)
            config = OverlayCloudsConfig.model_validate(data)

    blender_rotation = np.array([np.radians(c % 360) for c in [config.rotX, config.rotY, config.rotZ]])

    linestrings = []

    # ROTATE CLOUD COVERAGE RASTER

    img_pxpos = np.load(args.raytrace)
    map_clouds = cv2.imread(args.cloud_cover, cv2.IMREAD_GRAYSCALE)

    # img_pxpos = cv2.resize(img_pxpos, (500, 500))  # TODO
    # map_clouds = cv2.resize(map_clouds, (500, 500))  # TODO

    map_clouds = np.roll(map_clouds, int(map_clouds.shape[0] / 2), axis=1) # shift center to origin

    image_rotated = np.zeros_like(img_pxpos, dtype=np.uint8)

    for y in range(img_pxpos.shape[0]):
        for x in range(img_pxpos.shape[1]):
            p = img_pxpos[y, x]

            if np.isnan(np.sum(p)):
                continue

            p = rotate_points_inv([p], *(blender_rotation))[0]

            d = math.sqrt(np.sum(np.power(p, 2)))
            lat = math.acos(p[2] / d)
            lon = math.atan2(p[1] / d, p[0] / d) # - math.pi/2

            image_rotated[y, x] = _map(map_clouds, lat, lon).astype(np.uint8),

    if DEBUG:
        cv2.imwrite(str(DIR_DEBUG / "overlay_cloud.png"), image_rotated)

    exit()

    # CREATE HATCHLINES

    # ROTATE

    # map_clouds[0:10, :] = 0
    # # map_clouds = np.fliplr(map_clouds)
    # ps = rotate_points(
    #     [
    #         # np.array([1, 0, 0]),
    #         np.array([0, 1, 0]),
    #         # np.array([0, 0, 1])
    #     ],
    #     *blender_rotation,
    # )
    # print(ps)
    #
    # p = ps[0]
    # d = math.sqrt(np.sum(np.power(p, 2)))
    # colat = math.acos(p[2] / d)
    # lon = math.atan2(p[1] / d, p[0] / d) - math.pi/2
    #
    # print(math.degrees(colat), math.degrees(lon))

    # VISUALIZE

    if args.visualize:
        visualize_linestrings([LineString([[0, 0, 0], p]) for p in ps]).show()

    # EXPORT

    write_linestrings_to_npz(args.output, linestrings)


if __name__ == "__main__":
    main()