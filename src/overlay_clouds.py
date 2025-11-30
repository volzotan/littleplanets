import argparse
from pathlib import Path

import cv2
import netCDF4
import toml
from pydantic import BaseModel
import numpy as np
import math
import matplotlib.pyplot as plt

from process_blender import project_vectors_to_image_space
from util.misc import rotate_points_inv, export_angles, visualize

VISUALIZE = False
DEBUG = True
DIR_DEBUG = Path("debug")


class OverlayCloudsConfig(BaseModel):
    rotX: float = 0
    rotY: float = 0
    rotZ: float = 0

    light_angle_xy: float = 0
    light_angle_z: float = 0

    blur_kernel_size: int | None = 6
    morph_kernel_size: int | None = 5

    threshold: int = 150


def _visualize(u: np.ndarray, v: np.ndarray, path_base_image: Path = Path("output.png")) -> None:
    size = (150, 75)

    u = cv2.resize(u, size)
    v = cv2.resize(v, size)

    x = np.arange(u.shape[1])
    y = np.arange(u.shape[0])
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(30, 15))

    if path_base_image.exists():
        base_image = cv2.imread(str(path_base_image))
        base_image = cv2.resize(base_image, (int(size[0] * 10), int(size[1] * 10))).astype(float) / 255
        plt.imshow(base_image, cmap="gray", origin="upper")

    plt.quiver(X * 10, Y * 10, u, v, color="red", headwidth=1)
    plt.show()


def _map(raster: np.ndarray, lat: float, lon: float) -> np.ndarray:
    height, width = raster.shape

    y = int((lat / math.pi) * height)
    x = int((lon / math.tau) * width)

    return raster[y, x]


def _rotate_unit_vectors(vectors: np.ndarray, angles_2d: np.ndarray, step_distance: float = 0.01) -> np.ndarray:
    """
    :param vectors: (M, N, 3)
    :param angles_2d: (M, N), in radians
    :return: (M, N, 3)
    """

    v = vectors.reshape([-1, 3])

    ys = step_distance * np.cos(angles_2d)
    xs = step_distance * np.sin(angles_2d)

    ys = ys.reshape([-1])
    xs = xs.reshape([-1])

    Ry = np.zeros((len(ys), 3, 3))
    Ry[:, 0, 0] = np.cos(ys)
    Ry[:, 0, 2] = np.sin(ys)
    Ry[:, 1, 1] = 1
    Ry[:, 2, 0] = -np.sin(ys)
    Ry[:, 2, 2] = np.cos(ys)

    Rx = np.zeros((len(xs), 3, 3))
    Rx[:, 0, 0] = 1
    Rx[:, 1, 1] = np.cos(xs)
    Rx[:, 1, 2] = -np.sin(xs)
    Rx[:, 2, 1] = np.sin(xs)
    Rx[:, 2, 2] = np.cos(xs)

    R = Ry @ Rx

    rotated_v = np.einsum("nij,nj->ni", R, v)
    return rotated_v.reshape(vectors.shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("normals", type=Path, default="normals.exr", help="Normals (EXR)")
    parser.add_argument("raytrace", type=Path, default="raytrace.npy", help="Raytracing distance raster (NPY)")
    parser.add_argument("raytrace_backface", type=Path, default="raytrace_backface.npy", help="Raytracing backface distance raster (NPY)")
    # parser.add_argument("cloud_cover", type=Path, default="cloud.tif", help="Cloud grayscale coverage map (TIFF)")
    parser.add_argument("netcdf", type=Path, default="data.nc", help="Cloud coverage and wind direction data (netCDF4)")
    parser.add_argument("projection_matrix", type=Path, help="3x4 projection matrix (NPY)")
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

    img_pxpos_front = np.load(args.raytrace)
    img_pxpos_back = np.load(args.raytrace_backface)
    projection_matrix = np.load(args.projection_matrix)

    # READ CLOUD COVERAGE / WIND DIRECTION RASTER

    with netCDF4.Dataset(args.netcdf, "r", format="NETCDF4") as data:
        # convert xarray to numpy ndarray, fill empty pixels with zeros
        lsm = data.variables["lsm"][:].filled(0)[0, :, :]  # lsm - land sea mask

        # cbh = data.variables["cbh"][:].filled(0)[0, :, :] # cbh - cloud base height, [0-meters_above_sea_level]
        # hcc = data.variables["hcc"][:].filled(0)[0, :, :] # hcc - high cloud cover, 0 or 1.0
        # lcc = data.variables["lcc"][:].filled(0)[0, :, :] # lcc - low cloud cover, 0 or 1.0
        # mcc = data.variables["mcc"][:].filled(0)[0, :, :] # mcc - medium cloud cover, 0 or 1.0
        tcc = data.variables["tcc"][:].filled(0)[0, :, :]  # tcc - total cloud cover, 0 or 1.0
        mapping_clouds = (tcc * 255).astype(np.uint8)

        u = data.variables["u10"][:].filled(0)[0, :, :]
        v = data.variables["v10"][:].filled(0)[0, :, :]
        mapping_angle = np.atan2(v, u)

        if VISUALIZE:
            _visualize(u, v)

    if config.blur_kernel_size is not None and config.blur_kernel_size >= 3:
        mapping_clouds = cv2.blur(mapping_clouds, (config.blur_kernel_size, config.blur_kernel_size))

    if DEBUG:
        cv2.imwrite(str(DIR_DEBUG / "land_sea_mask.png"), (lsm * 255).astype(np.uint8))
        cv2.imwrite(str(DIR_DEBUG / "clouds_mapping_clouds.png"), mapping_clouds)
        cv2.imwrite(str(DIR_DEBUG / "clouds_mapping_angle.png"), (mapping_angle / math.tau * 255).astype(np.uint8))

    # shift center by 1/4 (adjust for X/Y axis change)
    mapping_clouds = np.roll(mapping_clouds, -int(mapping_clouds.shape[1] / 4), axis=1)
    mapping_angle = np.roll(mapping_angle, -int(mapping_angle.shape[1] / 4), axis=1)
    lsm = np.roll(lsm, -int(lsm.shape[1] / 4), axis=1)

    # ROTATE

    clouds_rotated_front = np.zeros(img_pxpos_front.shape[0:2], dtype=np.uint8)
    angles_rotated_front = np.zeros(img_pxpos_front.shape[0:2], dtype=float)  # in radians
    lsm_rotated_front = np.zeros(img_pxpos_front.shape[0:2])

    clouds_rotated_back = np.zeros(img_pxpos_front.shape[0:2], dtype=np.uint8)
    angles_rotated_back = np.zeros(img_pxpos_front.shape[0:2], dtype=float)  # in radians
    lsm_rotated_back = np.zeros(img_pxpos_front.shape[0:2])

    for y in range(img_pxpos_front.shape[0]):
        for x in range(img_pxpos_front.shape[1]):
            pf = img_pxpos_front[y, x]
            if not np.isnan(np.sum(pf)):
                pf = rotate_points_inv([pf], *(blender_rotation))[0]

                d = math.sqrt(np.sum(np.power(pf, 2)))
                lat = math.acos(pf[2] / d)
                lon = math.atan2(pf[1] / d, pf[0] / d)  # - math.pi/2

                clouds_rotated_front[y, x] = _map(mapping_clouds, lat, lon)
                angles_rotated_front[y, x] = _map(mapping_angle, lat, lon)
                lsm_rotated_front[y, x] = _map(lsm, lat, lon)

            pb = img_pxpos_back[y, x]
            if not np.isnan(np.sum(pb)):
                pb = rotate_points_inv([pb], *(blender_rotation))[0]

                d = math.sqrt(np.sum(np.power(pb, 2)))
                lat = math.acos(pb[2] / d)
                lon = math.atan2(pb[1] / d, pb[0] / d)  # - math.pi/2

                clouds_rotated_back[y, x] = _map(mapping_clouds, lat, lon)
                angles_rotated_back[y, x] = _map(mapping_angle, lat, lon)
                lsm_rotated_back[y, x] = _map(lsm, lat, lon)

    # PROJECT TO IMAGE SPACE

    direction_is_front = project_vectors_to_image_space(
        img_pxpos_front, _rotate_unit_vectors(img_pxpos_front, angles_rotated_front) - img_pxpos_front, projection_matrix
    )

    direction_is_back = project_vectors_to_image_space(
        img_pxpos_back, _rotate_unit_vectors(img_pxpos_back, angles_rotated_back) - img_pxpos_back, projection_matrix
    )

    if DEBUG:
        cv2.imwrite(str(DIR_DEBUG / "overlay_clouds.png"), clouds_rotated_front)
        cv2.imwrite(str(DIR_DEBUG / "overlay_clouds_back.png"), clouds_rotated_back)

        cv2.imwrite(str(DIR_DEBUG / "clouds_angles_rotated.png"), (angles_rotated_front / math.tau * 255).astype(np.uint8))
        cv2.imwrite(str(DIR_DEBUG / "clouds_angles_rotated_is.png"), export_angles(direction_is_front, adjust_y_axis=True))
        cv2.imwrite(str(DIR_DEBUG / "clouds_landseamask_rotated.png"), (lsm_rotated_front * 255).astype(np.uint8))

        cv2.imwrite(str(DIR_DEBUG / "clouds_angles_rotated_back.png"), (angles_rotated_back / math.tau * 255).astype(np.uint8))
        cv2.imwrite(str(DIR_DEBUG / "clouds_angles_rotated_is_back.png"), export_angles(direction_is_back, adjust_y_axis=True))
        cv2.imwrite(str(DIR_DEBUG / "clouds_landseamask_rotated_back.png"), (lsm_rotated_back * 255).astype(np.uint8))

    # BACKGROUND

    def create_background(raycast: np.ndarray, clouds_mapping: np.ndarray, config: OverlayCloudsConfig) -> np.ndarray:
        mapping_background = np.zeros(raycast.shape[0:2], dtype=np.uint8)
        mapping_background[np.isnan(np.sum(raycast, axis=2))] = 255

        mask = clouds_mapping >= config.threshold
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

        return mapping_background

    mapping_background_front = create_background(img_pxpos_front, clouds_rotated_front, config)
    mapping_background_back = create_background(img_pxpos_back, clouds_rotated_back, config)

    # EXPORT

    def _clip_and_rescale(raster: np.ndarray, min_value: float | int) -> np.ndarray:
        return (np.clip(raster, min_value, 255) - min_value) / (255 - min_value) * 255

    cv2.imwrite(str(args.output / "clouds_mapping_front_angle.png"), export_angles(direction_is_front, adjust_y_axis=True))
    cv2.imwrite(str(args.output / "clouds_mapping_front_distance.png"), _clip_and_rescale(clouds_rotated_front, config.threshold))
    cv2.imwrite(str(args.output / "clouds_mapping_front_background.png"), mapping_background_front)

    cv2.imwrite(str(args.output / "clouds_mapping_back_angle.png"), export_angles(direction_is_back, adjust_y_axis=True))
    cv2.imwrite(str(args.output / "clouds_mapping_back_distance.png"), _clip_and_rescale(clouds_rotated_back, config.threshold))
    cv2.imwrite(str(args.output / "clouds_mapping_back_background.png"), mapping_background_back)


if __name__ == "__main__":
    main()
