import argparse
from pathlib import Path

import cv2
import netCDF4
import toml
from pydantic import BaseModel
import numpy as np
import math
import matplotlib.pyplot as plt

from loguru import logger

from process_blender import project_vectors_to_image_space
from util.misc import export_angles, visualize, normalize_vectors, normalize_vector, rotate_points

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


def _map(raster: np.ndarray, lat: float, lon: float) -> np.ndarray:
    height, width = raster.shape

    y = int((lat / math.pi) * height)
    x = int((lon / math.tau) * width)

    return raster[y, x]


def _rotate_vectors_2d(vectors: np.ndarray, euler_rotations: np.ndarray) -> np.ndarray:
    V = vectors
    A = euler_rotations

    ax, ay, az = A[..., 0], A[..., 1], A[..., 2]
    sx, sy, sz = np.sin(ax), np.sin(ay), np.sin(az)
    cx, cy, cz = np.cos(ax), np.cos(ay), np.cos(az)

    x, y, z = V[..., 0], V[..., 1], V[..., 2]

    # Rx
    y1 = cx * y - sx * z
    z1 = sx * y + cx * z
    x1 = x

    # Ry
    z2 = cy * z1 - sy * x1
    x2 = sy * z1 + cy * x1
    y2 = y1

    # Rz
    x3 = cz * x2 - sz * y2
    y3 = sz * x2 + cz * y2
    z3 = z2

    return np.stack([x3, y3, z3], axis=-1)


def _rotate_vectors(vectors: np.ndarray, euler_rotation: np.ndarray, backwards: bool = False) -> np.ndarray:
    x, y, z = euler_rotation
    v = vectors.reshape([-1, 3])

    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(x), -np.sin(x)],
            [0, np.sin(x), np.cos(x)],
        ]
    )
    R_y = np.array(
        [
            [np.cos(y), 0, np.sin(y)],
            [0, 1, 0],
            [-np.sin(y), 0, np.cos(y)],
        ]
    )
    R_z = np.array(
        [
            [np.cos(z), -np.sin(z), 0],
            [np.sin(z), np.cos(z), 0],
            [0, 0, 1],
        ]
    )

    R = R_z @ R_y @ R_x
    if backwards:
        R = R.T
    v_rot = v @ R.T  # multiplying from the right inverses the rotation, prior transpose required

    return v_rot.reshape(vectors.shape)


def _rotate_and_map_nonvectorized(vectors: np.ndarray, raster: np.ndarray, euler_rotation: np.ndarray) -> np.ndarray:
    output = np.zeros(vectors.shape[0:2], dtype=raster.dtype)

    for y in range(vectors.shape[0]):
        for x in range(vectors.shape[1]):
            p = vectors[y, x]
            if not np.isnan(np.sum(p)):
                p = rotate_points([p], *euler_rotation, backwards=True)[0]

                d = math.sqrt(np.sum(np.power(p, 2)))
                lat = math.acos(p[2] / d)
                lon = math.atan2(p[1] / d, p[0] / d)

                output[y, x] = _map(raster, lat, lon)

    return output


def _rotate_and_map(vectors: np.ndarray, raster: np.ndarray, euler_rotation: np.ndarray) -> np.ndarray:
    """rotates an array of vectors and maps it against a raster representing the surface of a sphere"""

    output = np.full(vectors.shape[0:2], 0, dtype=raster.dtype)

    v_rot = _rotate_vectors(vectors, euler_rotation, backwards=True)
    v_rot_norm = normalize_vectors(v_rot)
    mask_nan = np.isnan(np.sum(v_rot, axis=2))

    # caveat: NaNs vectors will be mapped to 0, 0 indices in the raster image

    lats = np.acos(v_rot_norm[:, :, 2])
    lons = np.atan2(v_rot_norm[:, :, 1], v_rot_norm[:, :, 0])

    height, width = raster.shape
    I = ((lats / math.pi) * height).astype(int)
    J = ((lons / math.tau) * width).astype(int)

    I[mask_nan] = 0
    J[mask_nan] = 0

    output[:, :] = raster[I, J]

    # set NaN vector pixels to zero
    output[mask_nan] = 0

    return output


def _visualize_uv(u: np.ndarray, v: np.ndarray, path_base_image: Path = Path("output.png")) -> None:
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


def _visualize_vectors(centers: np.ndarray, vectors: list[np.ndarray]) -> None:
    visualize(centers.reshape([-1, 3]), [v.reshape([-1, 3]) for v in vectors], []).show()


def _visualize_points(points: np.ndarray | list[np.ndarray], scaling_factor: float | None = None) -> None:
    if type(points) is np.ndarray:
        point_sets = [points]
    else:
        point_sets = points

    for i, point_set in enumerate(point_sets):
        if scaling_factor is not None:
            point_sets[i] = cv2.resize(point_set, dsize=None, fx=scaling_factor, fy=scaling_factor)

        if point_set.shape[0] > 1:
            point_sets[i] = point_set.reshape([-1, 3])

    visualize([], [], point_sets).show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raytrace", type=Path, help="Raytracing distance raster (NPY)")
    parser.add_argument("raytrace_backface", type=Path, help="Raytracing backface distance raster (NPY)")
    parser.add_argument("netcdf", type=Path, help="Cloud coverage and wind direction data (netCDF4)")
    parser.add_argument("projection_matrix", type=Path, help="3x4 projection matrix (NPY)")
    parser.add_argument("--output", type=Path, default="build", help="Output directory")
    parser.add_argument("--config", type=Path, help="Configuration file [TOML]")
    parser.add_argument("--visualize", action="store_true", default=VISUALIZE, help="Enable interactive visualization")
    args = parser.parse_args()

    config = OverlayCloudsConfig()
    if args.config is not None:
        if args.config.exists():
            with open(args.config, "r") as f:
                data = toml.load(f)
                config = OverlayCloudsConfig.model_validate(data)
        else:
            logger.warning("No config found, writing empty file(s)")

            dummy_array = np.full([1, 1], 255, dtype=np.uint8)

            cv2.imwrite(str(args.output / "clouds_mapping_front_angle.png"), dummy_array)
            cv2.imwrite(str(args.output / "clouds_mapping_front_distance.png"), dummy_array)
            cv2.imwrite(str(args.output / "clouds_mapping_front_background.png"), dummy_array)

            cv2.imwrite(str(args.output / "clouds_mapping_back_angle.png"), dummy_array)
            cv2.imwrite(str(args.output / "clouds_mapping_back_distance.png"), dummy_array)
            cv2.imwrite(str(args.output / "clouds_mapping_back_background.png"), dummy_array)

            return

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

        if args.visualize:
            _visualize_uv(u, v, path_base_image=DIR_DEBUG / "land_sea_mask.png")

    if config.blur_kernel_size is not None and config.blur_kernel_size >= 3:
        mapping_clouds = cv2.blur(mapping_clouds, (config.blur_kernel_size, config.blur_kernel_size))

    if DEBUG:
        cv2.imwrite(str(DIR_DEBUG / "land_sea_mask.png"), (lsm * 255).astype(np.uint8))
        cv2.imwrite(str(DIR_DEBUG / "clouds_mapping_clouds.png"), mapping_clouds)
        cv2.imwrite(str(DIR_DEBUG / "clouds_mapping_angle.png"), (mapping_angle / math.tau * 255).astype(np.uint8))

    if args.visualize:
        resize_size = (40, 40)
        img_pxpos_front = cv2.resize(img_pxpos_front, resize_size)
        img_pxpos_back = cv2.resize(img_pxpos_back, resize_size)

    # shift center by 1/4 (adjust for X/Y axis change)
    mapping_clouds = np.roll(mapping_clouds, -int(mapping_clouds.shape[1] / 4), axis=1)
    mapping_angle = np.roll(mapping_angle, -int(mapping_angle.shape[1] / 4), axis=1)
    lsm = np.roll(lsm, -int(lsm.shape[1] / 4), axis=1)

    # ROTATE

    """
    Procedure:
    
    img_pxpos_front is a 2d array of the world space coordinates of every pixel in the image. 
    These world space coordinates are lying on the surface of a sphere, already rotated by `blender_rotation`.
    
    1) Rotate the world space img_pxpos coordinates back to their original position (unrotated sphere)
    2) Calculate the lat/lon coordinates of each point on a (unit-)sphere
    3) Map this point to the mapping_angles
    4) Rotate the vector a tiny distance according to this angle, resulting in a direction vector
    5) Rotate this direction vector back to the position from 1)
    6) Project the direction vector into image space
    """

    # 1)
    img_pxpos_front_orig = _rotate_vectors(img_pxpos_front, blender_rotation, backwards=True)
    img_pxpos_back_orig = _rotate_vectors(img_pxpos_back, blender_rotation, backwards=True)

    # 2, 3)
    clouds_rotated_front = _rotate_and_map(img_pxpos_front, mapping_clouds, blender_rotation)
    angles_rotated_front = _rotate_and_map(img_pxpos_front, mapping_angle, blender_rotation)
    lsm_rotated_front = _rotate_and_map(img_pxpos_front, lsm, blender_rotation)

    clouds_rotated_back = _rotate_and_map(img_pxpos_back, mapping_clouds, blender_rotation)
    angles_rotated_back = _rotate_and_map(img_pxpos_back, mapping_angle, blender_rotation)
    lsm_rotated_back = _rotate_and_map(img_pxpos_back, lsm, blender_rotation)

    if args.visualize:
        _visualize_points([img_pxpos_front, img_pxpos_front_orig])

    def calculate_direction_vector(vectors, mapping, step_distance: float = 0.01):
        deg_z = step_distance * np.cos(mapping)
        deg_x = step_distance * -np.sin(mapping)

        rotations = np.dstack((deg_x, np.zeros_like(deg_x), deg_z))

        p1 = vectors
        p2 = _rotate_vectors_2d(p1, rotations)

        return normalize_vectors(p2 - p1)

    def calculate_direction_vector_nonvectorized(vectors, mapping, step_distance: float = 0.01):
        output = np.zeros_like(vectors)
        for y in range(vectors.shape[0]):
            for x in range(vectors.shape[1]):
                p = vectors[y, x]
                if not np.isnan(np.sum(p)):
                    # 2, 3)
                    angle = mapping[y, x]

                    # 4
                    deg_z = step_distance * math.cos(angle)
                    deg_x = step_distance * -math.sin(angle)
                    rot = np.array([deg_x, 0, deg_z])
                    p2 = _rotate_vectors(p, rot)

                    output[y, x] = normalize_vector(p2 - p)

        return output

    # 4)
    direction_front_orig = calculate_direction_vector(img_pxpos_front_orig, angles_rotated_front)
    direction_back_orig = calculate_direction_vector(img_pxpos_back_orig, angles_rotated_back)

    # 5)
    direction_front = _rotate_vectors(direction_front_orig, blender_rotation)
    direction_back = _rotate_vectors(direction_back_orig, blender_rotation)

    # 6) after projection, the Y axis is flipped
    direction_is_front = project_vectors_to_image_space(img_pxpos_front, direction_front, projection_matrix)
    direction_is_back = project_vectors_to_image_space(img_pxpos_back, direction_back, projection_matrix)

    if args.visualize:
        _visualize_vectors(img_pxpos_front_orig, [direction_front_orig])
        _visualize_vectors(img_pxpos_front, [direction_front])
        # _visualize_vectors(img_pxpos_front, [direction_front, direction_is_front])

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
        # debug: enforce full cover
        # tmp = np.full_like(raster, 255)
        # tmp[500, 500] = 0
        # tmp[501, 500] = 255
        # return tmp

        return (np.clip(raster, min_value, 255) - min_value) / (255 - min_value) * 255

    cv2.imwrite(str(args.output / "clouds_mapping_front_angle.png"), export_angles(direction_is_front, adjust_y_axis=True))
    cv2.imwrite(str(args.output / "clouds_mapping_front_distance.png"), _clip_and_rescale(clouds_rotated_front, config.threshold))
    cv2.imwrite(str(args.output / "clouds_mapping_front_background.png"), mapping_background_front)

    cv2.imwrite(str(args.output / "clouds_mapping_back_angle.png"), export_angles(direction_is_back, adjust_y_axis=True))
    cv2.imwrite(str(args.output / "clouds_mapping_back_distance.png"), _clip_and_rescale(clouds_rotated_back, config.threshold))
    cv2.imwrite(str(args.output / "clouds_mapping_back_background.png"), mapping_background_back)


if __name__ == "__main__":
    main()
