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
from util.misc import rotate_points_inv, export_angles, visualize, normalize_vector

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


def _rotate_raster(vectors: np.ndarray, raster: np.ndarray, euler_rotation: np.ndarray) -> np.ndarray:

    output = np.zeros(vectors.shape[0:2], dtype=raster.dtype)

    for y in range(vectors.shape[0]):
        for x in range(vectors.shape[1]):
            p = vectors[y, x]
            if not np.isnan(np.sum(p)):
                p = rotate_points_inv([p], *(euler_rotation))[0]

                d = math.sqrt(np.sum(np.power(p, 2)))
                lat = math.acos(p[2] / d)
                lon = math.atan2(p[1] / d, p[0] / d)

                output[y, x] = _map(raster, lat, lon)

    return output



def _rotate_vector_backwards(vectors: np.ndarray, euler_rotation: np.ndarray) -> np.ndarray:

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

    v_rot = v @ (R_z @ R_y @ R_x) # multiplying from the right inverses the rotation, no transpose required

    return v_rot.reshape(vectors.shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("normals", type=Path, default="normals.exr", help="Normals (EXR)")
    parser.add_argument("raytrace", type=Path, default="raytrace.npy", help="Raytracing distance raster (NPY)")
    parser.add_argument("raytrace_backface", type=Path, default="raytrace_backface.npy", help="Raytracing backface distance raster (NPY)")
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

    azimuthal_angle = config.light_angle_xy
    polar_angle = config.light_angle_z
    light_pos = [
        math.sin(math.radians(polar_angle)) * math.cos(math.radians(azimuthal_angle)),
        math.sin(math.radians(polar_angle)) * math.sin(math.radians(azimuthal_angle)),
        math.cos(math.radians(polar_angle)),
    ]
    light_axis = normalize_vector(np.array(light_pos))

    # READ CLOUD COVERAGE / WIND DIRECTION RASTER

    with netCDF4.Dataset(args.netcdf, "r", format="NETCDF4") as data:
        # convert xarray to numpy ndarray, fill empty pixels with zeros
        lsm = data.variables["lsm"][:].filled(0)[0, :, :]  # lsm - land sea mask
        tcc = data.variables["tcc"][:].filled(0)[0, :, :]  # tcc - total cloud cover, 0 or 1.0
        mapping_clouds = (tcc * 255).astype(np.uint8)

        u = data.variables["u10"][:].filled(0)[0, :, :]
        v = data.variables["v10"][:].filled(0)[0, :, :]
        mapping_angle = np.atan2(v, u)

        if VISUALIZE:
            _visualize(u, v)



    new_size = (30,30)
    img_pxpos_front = cv2.resize(img_pxpos_front, new_size)
    img_pxpos_front_orig = _rotate_vector_backwards(img_pxpos_front, blender_rotation)
    # img_pxpos_front_orig = rotate_points_inv(img_pxpos_front.reshape([-1, 3]).tolist(), *blender_rotation)

    points = [img_pxpos_front.reshape([-1, 3]), img_pxpos_front_orig.reshape([-1, 3])]
    visualize([], [], points, light_axis=light_axis).show()


    # 1) ich nehme den raygecasteten Punkt in Weltkoordinaten (img_pxpos_front)
    # 2) rotiere ihn rückwärts auf seine ursprüngliche Weltkoordinate (Frontal Blick auf die Y-Achse, world coordinates)
    # 3) berechne die Lat/Lon pos dieses Punktes auf der Sphäre und mache das Mapping zum Winkel (Winkel ist relativ zur Kugeloberfläche mit oben als Z-Achse)
    # 4) Umrechnung dieses Winkels in einen World Coordinate Vector -> Resultat: Winkelvektor
    # 5) rotiere Winkelvektor vorwärts auf seine endgültige Weltkoordinate
    # 6) Rechne nun den Winkelvektor in Image Space um (startpunkt im IS ist bereits bekannt (der index im ndarray ist der pixel),
    #       dann kleinen schritt in richtung des winkels gehen (im WS), neuen Punkt als endpunkt in IS konvertieren, winkel berechnen




if __name__ == "__main__":
    main()
