import argparse
from pathlib import Path

import cv2
import netCDF4
import toml
from pydantic import BaseModel
import numpy as np
import math

from util.misc import rotate_points_inv

DEBUG = True
DIR_DEBUG = Path("debug")


class OverlayCloudsConfig(BaseModel):
    rotX: float = 0
    rotY: float = 0
    rotZ: float = 0

    blur_kernel_size: int | None = 6
    morph_kernel_size: int | None = 5

    threshold: int = 150


def _map(raster: np.ndarray, lat: float, lon: float) -> np.ndarray:
    height, width = raster.shape

    y = int((lat / math.pi) * height)
    x = int((lon / math.tau) * width)

    return raster[y, x]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("normals", type=Path, default="normals.exr", help="Normals (EXR)")
    parser.add_argument("raytrace", type=Path, default="raytrace.npy", help="Raytracing distance raster (NPY)")
    parser.add_argument("cloud_cover", type=Path, default="cloud.tif", help="Cloud grayscale coverage map (TIFF)")
    parser.add_argument("wind_direction", type=Path, default="wind_direction.nc", help="Wind direction data (netCDF4)")
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

    # CLOUD COVERAGE RASTER

    img_pxpos = np.load(args.raytrace)
    map_clouds = cv2.imread(args.cloud_cover, cv2.IMREAD_GRAYSCALE)
    map_clouds = np.roll(map_clouds, int(map_clouds.shape[1] / 4), axis=1)  # shift center to 1/4 (adjust for X/Y axis change)

    if config.blur_kernel_size is not None and config.blur_kernel_size >= 3:
        map_clouds = cv2.blur(map_clouds, (config.blur_kernel_size, config.blur_kernel_size))

    # WRITE WIND DIRECTION RASTER

    mapping_angle = None
    with netCDF4.Dataset(args.wind_direction, "r", format="NETCDF4") as data:
        # convert xarray to numpy ndarray, fill empty pixels with zeros
        u = data.variables["u10"][:].filled(0)[0, :, :]
        v = data.variables["v10"][:].filled(0)[0, :, :]

        mapping_angle = np.atan2(u, v)
        mapping_angle = (mapping_angle + math.pi / 2) % math.tau

        # center on 255/2
        mapping_angle = (mapping_angle + np.pi) / (np.pi * 2)
        mapping_angle = (mapping_angle * 255).astype(np.uint8)

        mapping_angle = cv2.resize(mapping_angle, (img_pxpos.shape[0], img_pxpos.shape[1]))

    mapping_angle = np.roll(mapping_angle, int(mapping_angle.shape[1] / 4), axis=1)  # shift center to 1/4 (adjust for X/Y axis change)

    if DEBUG:
        cv2.imwrite(str(DIR_DEBUG / "clouds_mapping_angle.png"), mapping_angle)

    # cv2.imwrite(str(args.output / "clouds_mapping_angle.png"), mapping_angle)
    # exit()

    # TODO: convert angles from map space to image space

    # ROTATE

    image_rotated = np.zeros(img_pxpos.shape[0:2], dtype=np.uint8)
    angles_rotated = np.zeros(img_pxpos.shape[0:2], dtype=np.uint8)
    for y in range(img_pxpos.shape[0]):
        for x in range(img_pxpos.shape[1]):
            p = img_pxpos[y, x]

            if np.isnan(np.sum(p)):
                continue

            p = rotate_points_inv([p], *(blender_rotation))[0]

            d = math.sqrt(np.sum(np.power(p, 2)))
            lat = math.acos(p[2] / d)
            lon = math.atan2(p[1] / d, p[0] / d)  # - math.pi/2

            image_rotated[y, x] = _map(map_clouds, lat, lon).astype(np.uint8)
            angles_rotated[y, x] = _map(mapping_angle, lat, lon).astype(np.uint8)

    # BACKGROUND

    mapping_background = np.zeros(img_pxpos.shape[0:2], dtype=np.uint8)
    mapping_background[np.isnan(np.sum(img_pxpos, axis=2))] = 255

    mask = image_rotated >= config.threshold
    if config.morph_kernel_size is not None and config.morph_kernel_size >= 3:
        mask_uint8 = np.zeros_like(mask, dtype=np.uint8)
        mask_uint8[mask] = 255

        # cv2.imwrite(str(DIR_DEBUG / "overlay_clouds_mask.png"), mask_uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.morph_kernel_size, config.morph_kernel_size))
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)

        # cv2.imwrite(str(DIR_DEBUG / "overlay_clouds_mask_morph.png"), mask_uint8)

        mapping_background[mask_uint8 == 0] = 255
    else:
        mapping_background[~mask] = 255

    # image_rotated[mask] = 255
    # image_rotated[~mask] = 150
    # image_rotated = np.clip(image_rotated, 150)

    # image_rotated = ~image_rotated # default: white paper, colored ink

    if DEBUG:
        cv2.imwrite(str(DIR_DEBUG / "overlay_clouds.png"), image_rotated)

    # exit()

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

    # if args.visualize:
    #     visualize_linestrings([LineString([[0, 0, 0], p]) for p in ps]).show()

    # EXPORT

    # write_linestrings_to_npz(args.output, linestrings)

    # bgmask = np.isnan(np.sum(img_pxpos, axis=2))
    # image_rotated = np.zeros_like(image_rotated)
    # image_rotated[~bgmask] = 230
    # image_rotated[500, 500] = 0
    # image_rotated[501, 500] = 255
    # mapping_background = np.zeros_like(mapping_background)
    # mapping_background[bgmask] = 255

    cv2.imwrite(str(args.output / "clouds_mapping_angle.png"), angles_rotated)
    cv2.imwrite(str(args.output / "clouds_mapping_distance.png"), image_rotated)
    cv2.imwrite(str(args.output / "clouds_mapping_background.png"), mapping_background)


if __name__ == "__main__":
    main()
