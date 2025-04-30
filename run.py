import math
import os
from pathlib import Path
import textwrap
from typing import NamedTuple

import rasterio
from stl import mesh
import numpy as np


class Mesh(NamedTuple):
    points: list[np.array]
    triangles: list[tuple[int, int, int]]
    colors: list[np.array]


def tetrahedron() -> Mesh:
    a = 1.0
    h_d = (math.sqrt(3) / 2.0) * a
    h_p = a * math.sqrt(2.0 / 3.0)
    center = np.array([0, h_d / 3.0, h_p / 4.0])

    p1 = np.array([-a / 2.0, 0, 0]) - center
    p2 = np.array([a / 2.0, 0, 0]) - center
    p3 = np.array([0, h_d, 0]) - center
    p4 = np.array([0, h_d / 3.0, h_p]) - center

    points = [p1, p2, p3, p4]

    triangles = [
        (0, 2, 1),
        (0, 1, 3),
        (2, 0, 3),
        (1, 2, 3),
    ]

    return Mesh(points, triangles, [])


def write_stl(m: Mesh, filename: Path) -> None:
    obj = mesh.Mesh(np.zeros(len(m.triangles), dtype=mesh.Mesh.dtype))

    for i, t in enumerate(m.triangles):
        obj.vectors[i] = np.array([m.points[i] for i in t])

    obj.save(str(filename))


def write_ply(m: Mesh, filename: Path) -> None:
    with open(filename, "w") as f:
        data = f"""
                ply
                format ascii 1.0
                element vertex {len(m.points)}
                property float x
                property float y
                property float z
                property uchar red
                property uchar green
                property uchar blue
                element face {len(m.triangles)}
                property list uchar int vertex_indices
                end_header
                """

        data = textwrap.dedent(data[1:])  # remove first newline (for dedent to work)
        f.write(data)

        for i in range(len(m.points)):
            p0, p1, p2 = m.points[i]
            c0, c1, c2 = m.colors[i]

            f.write(f"{p0:.4f} {p1:.4f} {p2:.4f} {c0:d} {c1:d} {c2:d}\n")

        for i0, i1, i2 in m.triangles:
            f.write(f"3 {i0} {i1} {i2}\n")

        f.write("\n")


def subdivide(m: Mesh, n: int) -> Mesh:
    if n <= 0:
        return m

    output_points = m.points
    output_triangles = []

    map_points = {}

    for i0, i1, i2 in m.triangles:
        p0 = output_points[i0]
        p1 = output_points[i1]
        p2 = output_points[i2]

        alpha = p0 + (p1 - p0) / 2
        beta = p1 + (p2 - p1) / 2
        gamma = p2 + (p0 - p2) / 2

        k_alpha = "{}|{}".format(*sorted([i0, i1]))
        k_beta = "{}|{}".format(*sorted([i1, i2]))
        k_gamma = "{}|{}".format(*sorted([i2, i0]))

        i_alpha = -1
        i_beta = -1
        i_gamma = -1

        if k_alpha in map_points:
            i_alpha = map_points[k_alpha]
        else:
            output_points.append(alpha)
            i_alpha = len(output_points) - 1

        if k_beta in map_points:
            i_beta = map_points[k_beta]
        else:
            output_points.append(beta)
            i_beta = len(output_points) - 1

        if k_gamma in map_points:
            i_gamma = map_points[k_gamma]
        else:
            output_points.append(gamma)
            i_gamma = len(output_points) - 1

        output_triangles.append([i0, i_alpha, i_gamma])
        output_triangles.append([i_alpha, i1, i_beta])
        output_triangles.append([i_beta, i2, i_gamma])
        output_triangles.append([i_gamma, i_alpha, i_beta])

    return subdivide(Mesh(output_points, output_triangles, []), n - 1)


def project_to_sphere(m: Mesh) -> Mesh:
    for i in range(len(m.points)):
        p = m.points[i]

        r = math.sqrt(np.sum(np.power(p, 2)))
        lat = math.acos(p[2] / r)
        lon = math.atan2(p[1], p[0])

        r = 1.0

        x = r * math.sin(lat) * math.cos(lon)
        y = r * math.sin(lat) * math.sin(lon)
        z = r * math.cos(lat)

        cartesian_projected = np.array([x, y, z])
        m.points[i] = cartesian_projected

    return m


def map(raster: np.ndarray, lat: float, lon: float) -> np.ndarray:
    height, width = raster.shape[1:3]

    y = int(((lat / math.pi) + 0.0) * height)
    x = int(((lon / math.tau) + 0.0) * width)

    return raster[:, y, x]


def project(
    m: Mesh, raster_height: np.ndarray, raster_color: np.ndarray, scale: float = 0.1
) -> Mesh:
    for i in range(len(m.points)):
        p = m.points[i]

        d = math.sqrt(np.sum(np.power(p, 2)))
        lat = math.acos(p[2] / d)
        lon = math.atan2(p[1], p[0])

        r = 1.0 + map(raster_height, lat, lon)[0] * scale

        x = r * math.sin(lat) * math.cos(lon)
        y = r * math.sin(lat) * math.sin(lon)
        z = r * math.cos(lat)

        cartesian_projected = np.array([x, y, z])
        m.points[i] = cartesian_projected

        c = map(raster_color, lat, lon).astype(np.uint8)
        m.colors.append(c)

    return m


def load_raster(filename: Path) -> np.ndarray:
    with rasterio.open(filename) as dataset:
        data = dataset.read()
        return data


def normalize(data: np.ndarray) -> np.ndarray:
    """Normalize to [-1, 1]"""
    return data / max(abs(np.min(data)), abs(np.max(data)))


if __name__ == "__main__":

    # color_mapping = load_raster(Path("BasicHapke_wbhs_blend_b137_IOF.55deg_nearcenter_1kmpp_20140925_134515.tif"))
    # empty = np.zeros([color_mapping.shape[0], int(color_mapping.shape[1] * 25/90), color_mapping.shape[2]], dtype=np.uint8)
    # color_mapping = np.concatenate([empty, color_mapping, empty], axis=1)

    poly = project(
        subdivide(tetrahedron(), n=10),
        normalize(load_raster(Path("Lunar_DEM_resized.tif"))),
        load_raster(Path("lroc_color_poles.tif"))[0:3, :, :],
        scale=3e-2,
    )

    asset_dir = Path("assets")
    os.makedirs(asset_dir)

    write_ply(poly, Path(asset_dir, "output.ply"))
    write_stl(poly, Path(asset_dir, "output.stl"))
