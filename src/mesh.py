import math
import os
from dataclasses import dataclass
from pathlib import Path
import textwrap
import datetime
import argparse

import rasterio
from scipy.spatial import cKDTree
from stl import mesh
import numpy as np
import pyvista as pv
import cv2
import toml
from pydantic import BaseModel, Field

from loguru import logger

# since the magnitude of the elevation direction vector is so small
# a strong weight is required in order to have _any_ noticeable effect
# compared to the light axis direction unit vector
ELEVATION_VECTOR_WEIGHT = 0.6  # 0.95

DIR_DEBUG = Path("debug")


class MeshConfig(BaseModel):
    scale: float = Field(default=0.10, description="Scaling factor")
    fixed_elevation_scale: float | None = Field(
        default=None, description="A fixed scale value [-1, +1] that uniformly overrides elevation raster data"
    )
    blur: int = Field(default=100, description="Elevation raster blurring kernel size")
    subdivision: int = Field(default=10, description="Number of subdivision steps")


@dataclass
class Mesh:
    points: np.ndarray
    faces: np.ndarray
    colors: np.ndarray | None = None
    centers: np.ndarray | None = None
    normals: np.ndarray | None = None
    field_vectors: np.ndarray | None = None
    field_elevation_vectors: np.ndarray | None = None

    def __repr__(self) -> str:
        points_len = len(self.points) if self.points is not None else 0
        faces_len = len(self.faces) if self.faces is not None else 0
        colors_len = len(self.colors) if self.colors is not None else 0
        return f"Mesh[points={points_len}, faces={faces_len}, colors={colors_len}]"


def triangle() -> Mesh:
    points = np.array(
        [
            [-1.0, -1.0, 1.5],
            [1.0, -1.0, 1.5],
            [0.0, 1.0, 1.5],
        ],
        dtype=np.float32,
    )

    faces = np.array([[0, 1, 2]], dtype=np.int32)

    return Mesh(points, faces)


def tetrahedron() -> Mesh:
    a = 1.0
    height_base = math.sqrt(3) / 2.0 * a
    height_pyramid = a * math.sqrt(2.0 / 3.0)
    center = np.array([0.0, height_base / 3.0, height_pyramid / 4.0], dtype=np.float32)

    points = (
        np.array(
            [
                [-a / 2.0, 0.0, 0.0],
                [a / 2.0, 0.0, 0.0],
                [0.0, height_base, 0.0],
                [0.0, height_base / 3.0, height_pyramid],
            ],
            dtype=np.float32,
        )
        - center
    )

    faces = np.array(
        [
            [0, 2, 1],
            [0, 1, 3],
            [2, 0, 3],
            [1, 2, 3],
        ],
        dtype=np.int32,
    )

    return Mesh(points, faces)


def cube() -> Mesh:
    points = np.array(
        [
            [-1.0, -1.0, -1.0],  # 0: bottom-left-back
            [1.0, -1.0, -1.0],  # 1: bottom-right-back
            [1.0, 1.0, -1.0],  # 2: bottom-right-front
            [-1.0, 1.0, -1.0],  # 3: bottom-left-front
            [-1.0, -1.0, 1.0],  # 4: top-left-back
            [1.0, -1.0, 1.0],  # 5: top-right-back
            [1.0, 1.0, 1.0],  # 6: top-right-front
            [-1.0, 1.0, 1.0],  # 7: top-left-front
        ],
        dtype=np.float32,
    )

    faces = np.array(
        [
            # Bottom face (z = -1)
            [0, 3, 1],
            [1, 3, 2],
            # Top face (z = 1)
            [4, 5, 7],
            [5, 6, 7],
            # Front face (y = -1)
            [0, 1, 4],
            [1, 5, 4],
            # Back face (y = 1)
            [2, 3, 6],
            [3, 7, 6],
            # Right face (x = 1)
            [1, 2, 5],
            [2, 6, 5],
            # Left face (x = -1)
            [3, 0, 7],
            [0, 4, 7],
        ],
        dtype=np.int32,
    )

    return Mesh(points, faces)


def write_stl(m: Mesh, filename: Path) -> None:
    obj = mesh.Mesh(np.zeros(len(m.faces), dtype=mesh.Mesh.dtype))

    for i, face in enumerate(m.faces):
        obj.vectors[i] = m.points[face]

    filename.parent.mkdir(parents=True, exist_ok=True)
    obj.save(str(filename))


def write_ply(m: Mesh, filename: Path, color_vertices=False) -> None:
    if not color_vertices:  # faces
        with open(filename, "w") as f:
            data = f"""
                    ply
                    format ascii 1.0
                    element vertex {len(m.points)}
                    property float x
                    property float y
                    property float z
                    element face {len(m.faces)}
                    property list uchar int vertex_indices
                    property uchar red
                    property uchar green
                    property uchar blue
                    end_header
                    """

            # remove first newline (for dedent to work)
            data = textwrap.dedent(data[1:])
            f.write(data)

            for i in range(len(m.points)):
                p0, p1, p2 = m.points[i]

                f.write(f"{p0:.4f} {p1:.4f} {p2:.4f}\n")

            for i in range(len(m.faces)):
                i0, i1, i2 = m.faces[i]
                c0, c1, c2 = m.colors[i]

                f.write(f"3 {i0} {i1} {i2} {c0:d} {c1:d} {c2:d}\n")

            f.write("\n")

    else:  # vertices
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
                    element face {len(m.faces)}
                    property list uchar int vertex_indices
                    end_header
                    """

            # remove first newline (for dedent to work)
            data = textwrap.dedent(data[1:])
            f.write(data)

            for i in range(len(m.points)):
                p0, p1, p2 = m.points[i]
                c0, c1, c2 = m.colors[i]

                f.write(f"{p0:.4f} {p1:.4f} {p2:.4f} {c0:d} {c1:d} {c2:d}\n")

            for i0, i1, i2 in m.faces:
                f.write(f"3 {i0} {i1} {i2}\n")

            f.write("\n")


def write_ply_pointcloud(m: Mesh, filename: Path) -> None:
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
                end_header
                """

        # remove first newline (for dedent to work)
        data = textwrap.dedent(data[1:])
        f.write(data)

        for i in range(len(m.points)):
            p0, p1, p2 = m.points[i]
            c0, c1, c2 = m.colors[i] if m.colors is not None else [255, 255, 255]

            f.write(f"{p0:.4f} {p1:.4f} {p2:.4f} {c0:d} {c1:d} {c2:d}\n")

        f.write("\n")


def subdivide(mesh: Mesh, n: int) -> Mesh:
    if n <= 0:
        return mesh

    points = mesh.points.copy()
    faces = mesh.faces.copy()

    for level in range(n):
        num_points = len(points)
        num_faces = len(faces)

        all_edges = np.empty((num_faces * 3, 2), dtype=np.int32)
        all_edges[0::3] = np.stack([faces[:, 0], faces[:, 1]], axis=1)  # edge 0-1
        all_edges[1::3] = np.stack([faces[:, 1], faces[:, 2]], axis=1)  # edge 1-2
        all_edges[2::3] = np.stack([faces[:, 2], faces[:, 0]], axis=1)  # edge 2-0

        all_edges.sort(axis=1)

        unique_edges, edge_inverse = np.unique(all_edges, axis=0, return_inverse=True)
        num_unique_edges = len(unique_edges)

        edge_midpoints = (points[unique_edges[:, 0]] + points[unique_edges[:, 1]]) * 0.5
        new_points = np.vstack([points, edge_midpoints])
        edge_to_new_point = num_points + np.arange(num_unique_edges)
        midpoint_indices = edge_to_new_point[edge_inverse].reshape(num_faces, 3)
        new_faces = np.empty((num_faces * 4, 3), dtype=np.int32)

        v0, v1, v2 = faces[:, 0], faces[:, 1], faces[:, 2]
        m01, m12, m20 = midpoint_indices[:, 0], midpoint_indices[:, 1], midpoint_indices[:, 2]

        new_faces[0::4] = np.stack([v0, m01, m20], axis=1)
        new_faces[1::4] = np.stack([m01, v1, m12], axis=1)
        new_faces[2::4] = np.stack([m12, v2, m20], axis=1)
        new_faces[3::4] = np.stack([m20, m01, m12], axis=1)

        points = new_points
        faces = new_faces

    return Mesh(points, faces)


def project(m: Mesh, raster: np.ndarray | None, scale: float = 0.1) -> Mesh:
    points = m.points

    # cartesian to spherical
    r = np.linalg.norm(points, axis=1)
    lat = np.arccos(points[:, 2] / r)
    lon = np.arctan2(points[:, 1], points[:, 0])

    r = np.ones(len(points))

    if raster is not None:
        height, width = raster.shape
        y_indices = ((lat / math.pi) * height).astype(int)
        x_indices = ((lon / math.tau) * width).astype(int)

        # Clamp latitude indices only (longitude can wrap around with negative indexing)
        y_indices = np.clip(y_indices, 0, height - 1)

        r += raster[y_indices, x_indices] * scale

    # spherical to cartesian
    m.points[:, 0] = r * np.sin(lat) * np.cos(lon)
    m.points[:, 1] = r * np.sin(lat) * np.sin(lon)
    m.points[:, 2] = r * np.cos(lat)

    return m


def add_color(m: Mesh, raster: np.ndarray, color_vertices=False) -> Mesh:
    if not color_vertices:
        # Get first vertex of each face for face coloring
        points_array = m.points[m.faces[:, 0]]
    else:
        points_array = m.points

    # cartesian to spherical
    d = np.linalg.norm(points_array, axis=1)
    lat = np.arccos(points_array[:, 2] / d)
    lon = np.arctan2(points_array[:, 1], points_array[:, 0])

    height, width, channels = raster.shape
    y_indices = ((lat / math.pi) * height).astype(int)
    x_indices = ((lon / math.tau) * width).astype(int)

    # Clamp latitude indices only (longitude can wrap around with negative indexing)
    y_indices = np.clip(y_indices, 0, height - 1)

    m.colors = raster[y_indices, x_indices].astype(np.uint8)

    return m


def fill_color(m: Mesh, value: list[int] = [255, 255, 255]) -> Mesh:
    num_faces = len(m.faces)
    m.colors = np.tile(np.array(value, dtype=np.uint8), (num_faces, 1))
    return m


def normalize_elevation(data: np.ndarray) -> np.ndarray:
    """Normalize to [-1, 1]"""
    return data / max(abs(np.min(data)), abs(np.max(data)))


def load_raster(filename: Path) -> np.ndarray:
    if not filename.exists():
        raise FileNotFoundError(f"Raster file not found: {filename}")

    with rasterio.open(filename) as dataset:
        return dataset.read()


def _compute_normals(m: Mesh) -> tuple[np.ndarray, np.ndarray]:
    points = m.points
    faces = m.faces

    v0 = points[faces[:, 0]]
    v1 = points[faces[:, 1]]
    v2 = points[faces[:, 2]]

    normals = np.cross(v1 - v0, v2 - v0)
    normalized_normals = _normalize_vectors(normals)

    centers = (v0 + v1 + v2) / 3.0

    return centers, normalized_normals


def _normalize_vector(v: np.array) -> np.array:
    return v / np.linalg.norm(v)


def _normalize_vectors(v: np.ndarray) -> np.array:
    return v / np.linalg.norm(v, axis=1).reshape(-1, 1)


def shift_center_to_origin(raster: np.ndarray) -> np.ndarray:
    """
    shift center of raster image to 1/4. This adjusts for the X/Y axis change.
    atan2 returns angles relative to the X axis, while the default blender orientation
    should have the zero meridian face the Y axis.

    """
    return np.roll(raster, int(raster.shape[1] / 4), axis=1)


def write_obj(plotter: pv.Plotter, filename: Path) -> None:
    if not filename.suffix == ".obj":
        filename = filename.parent / (filename.name + ".obj")
    plotter.export_obj(filename)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elevation", type=Path, help="Elevation raster data in the (Geo)Tiff format (TIF)")
    parser.add_argument("--color", type=Path, help="Surface color raster data in the Tiff format (TIF)")
    parser.add_argument("--output", type=Path, default="mesh.ply", help="Output filename (PLY)")
    parser.add_argument("--config", type=Path, help="Configuration file [TOML]")
    parser.add_argument("--debug", action="store_true", default=False, help="Write debug output")

    args = parser.parse_args()

    config = MeshConfig()
    if args.config is not None:
        if args.config.exists():
            with open(args.config, "r") as f:
                data = toml.load(f)
                config = MeshConfig.model_validate(data)
        else:
            logger.info("No config supplied, writing empty file")
            write_ply(Mesh([], []), args.output)
            return

    os.makedirs(args.output if args.output.is_dir() else args.output.parent, exist_ok=True)

    # WRITE

    timer_start = datetime.datetime.now()

    if args.elevation is not None:
        dem_raster = normalize_elevation(load_raster(args.elevation))
        dem_raster = dem_raster[0, :, :]

    if config.fixed_elevation_scale is not None:
        dem_raster = np.full([1, 1, 1], config.fixed_elevation_scale)

    if args.color is not None:
        color_raster = load_raster(args.color)
        color_raster = np.transpose(color_raster[0:3, :, :], (1, 2, 0))  # from [3, rows, cols] to [rows, cols, 3]
    else:
        color_raster = np.full([1, 1, 3], 255, dtype=np.uint8)

    if color_raster.shape[2] == 1:  # grayscale color image, i.e. cloud cover map
        color_raster = np.dstack([color_raster, color_raster, color_raster])

    dem_raster = shift_center_to_origin(dem_raster)
    color_raster = shift_center_to_origin(color_raster)

    # use a partial DEM (lat not from -90 to 90)
    # color_raster = load_raster(Path("data", "BasicHapke_wbhs_blend_b137_IOF.55deg_nearcenter_1kmpp_20140925_134515.tif")) # [3, rows, cols]
    # empty = np.zeros([color_raster.shape[0], int(color_raster.shape[1] * 25/90), color_raster.shape[2]], dtype=np.uint8)
    # color_raster = np.concatenate([empty, color_raster, empty], axis=1)

    if args.debug:
        cv2.imwrite(str(DIR_DEBUG / "color_raster.png"), color_raster)

    if config.blur is not None and config.blur > 1:
        dem_raster = cv2.blur(dem_raster, (config.blur, config.blur))

    poly = project(subdivide(tetrahedron(), n=config.subdivision), dem_raster, scale=config.scale)
    write_ply(
        add_color(poly, color_raster, color_vertices=True),
        args.output,
        color_vertices=True,
    )

    print("Completed in: {:.3f}s".format((datetime.datetime.now() - timer_start).total_seconds()))


if __name__ == "__main__":
    main()
