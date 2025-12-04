import math
import os
from dataclasses import dataclass
from pathlib import Path
import textwrap
import datetime
import argparse

import rasterio
import rtree
from stl import mesh
import numpy as np
import pyvista as pv
from shapely.geometry import LineString
import cv2
import toml
from pydantic import BaseModel, Field

from loguru import logger

# since the magnitude of the elevation direction vector is so small
# a strong weight is required in order to have _any_ noticeable effect
# compared to the light axis direction unit vector
ELEVATION_VECTOR_WEIGHT = 0.6  # 0.95

DIR_DEBUG = Path("debug")


@dataclass
class Mesh:
    points: list[np.array]
    faces: list[np.array]
    colors: list[np.array] | None = None
    centers: list[np.array] | None = None
    normals: list[np.array] | None = None
    field_vectors: list[np.array] | None = None
    field_elevation_vectors: list[np.array] | None = None

    def __repr__(self):
        return f"Mesh [ points: {len(self.points)} / faces: {len(self.faces)} / colors: {len(self.colors)} ]"


def triangle() -> Mesh:
    p1 = np.array([-1, -1, 1.5], dtype=np.float32)
    p2 = np.array([+1, -1, 1.5], dtype=np.float32)
    p3 = np.array([+0, +1, 1.5], dtype=np.float32)

    points = [p1, p2, p3]

    triangles = [
        (0, 1, 2),
    ]

    return Mesh(points, [np.array(t) for t in triangles])


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

    return Mesh(points, [np.array(t) for t in triangles])


def cube() -> Mesh:
    p1 = np.array([-1, -1, -1])
    p2 = np.array([1, -1, -1])
    p3 = np.array([1, 1, -1])
    p4 = np.array([-1, 1, -1])
    p5 = np.array([-1, -1, 1])
    p6 = np.array([1, -1, 1])
    p7 = np.array([1, 1, 1])
    p8 = np.array([-1, 1, 1])

    points = [p1, p2, p3, p4, p5, p6, p7, p8]

    triangles = [
        (0, 3, 1),  # Bottom
        (1, 3, 2),
        (4, 5, 7),  # Top
        (5, 6, 7),
        (0, 1, 4),  # Front
        (1, 5, 4),
        (2, 3, 6),  # Back
        (3, 7, 6),
        (1, 2, 5),  # Right
        (2, 6, 5),
        (3, 0, 7),  # Left
        (0, 4, 7),
    ]

    return Mesh(points, [np.array(t) for t in triangles])


def write_stl(m: Mesh, filename: Path) -> None:
    obj = mesh.Mesh(np.zeros(len(m.faces), dtype=mesh.Mesh.dtype))

    for i, t in enumerate(m.faces):
        obj.vectors[i] = np.array([m.points[i] for i in t])

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


def subdivide(m: Mesh, n: int) -> Mesh:
    if n <= 0:
        return m

    output_points = m.points
    output_triangles = []

    map_points: dict[str, int] = {}

    for i0, i1, i2 in m.faces:
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
            map_points[k_alpha] = i_alpha

        if k_beta in map_points:
            i_beta = map_points[k_beta]
        else:
            output_points.append(beta)
            i_beta = len(output_points) - 1
            map_points[k_beta] = i_beta

        if k_gamma in map_points:
            i_gamma = map_points[k_gamma]
        else:
            output_points.append(gamma)
            i_gamma = len(output_points) - 1
            map_points[k_gamma] = i_gamma

        output_triangles.append(np.array([i0, i_alpha, i_gamma]))
        output_triangles.append(np.array([i_alpha, i1, i_beta]))
        output_triangles.append(np.array([i_beta, i2, i_gamma]))
        output_triangles.append(np.array([i_gamma, i_alpha, i_beta]))

    return subdivide(Mesh(output_points, output_triangles), n - 1)


def _map(raster: np.ndarray, lat: float, lon: float) -> np.ndarray:
    height, width = raster.shape

    y = int(((lat / math.pi) + 0.0) * height)
    x = int(((lon / math.tau) + 0.0) * width)

    return raster[y, x]


def project(m: Mesh, raster: np.ndarray | None, scale: float = 0.1) -> Mesh:
    for i in range(len(m.points)):
        p = m.points[i]

        r = math.sqrt(np.sum(np.power(p, 2)))
        lat = math.acos(p[2] / r)
        lon = math.atan2(p[1], p[0])

        r = 1.0
        if raster is not None:
            r += _map(raster, lat, lon) * scale

        x = r * math.sin(lat) * math.cos(lon)
        y = r * math.sin(lat) * math.sin(lon)
        z = r * math.cos(lat)

        m.points[i] = np.array([x, y, z])

    return m


def add_color(m: Mesh, raster: np.ndarray, color_vertices=False) -> Mesh:
    m.colors = []

    points = []
    if not color_vertices:
        points = [m.points[f[0]] for f in m.faces]
    else:
        points = m.points

    for p in points:
        d = math.sqrt(np.sum(np.power(p, 2)))
        lat = math.acos(p[2] / d)
        lon = math.atan2(p[1], p[0])

        c = [
            _map(raster[:, :, 0], lat, lon).astype(np.uint8),
            _map(raster[:, :, 1], lat, lon).astype(np.uint8),
            _map(raster[:, :, 2], lat, lon).astype(np.uint8),
        ]
        m.colors.append(c)

    return m


def fill_color(m: Mesh, value: list[int] = [255, 255, 255]) -> Mesh:
    m.colors = []

    for _ in range(len(m.faces)):
        m.colors.append(np.array(value, dtype=np.uint8))

    return m


def get_seedpoints(num: int) -> list[np.array]:
    # equal distance point placement on a sphere:
    # https://stackoverflow.com/a/44164075

    indices = np.arange(0, num, dtype=float) + 0.5

    phi = np.arccos(1 - 2 * indices / num)
    theta = np.pi * (1 + 5**0.5) * indices

    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)

    points = np.stack([x, y, z]).T
    return list(points)


def add_seedpoints(m: Mesh, num: int) -> Mesh:
    m.colors = [np.array([255, 255, 255], dtype=np.uint8)] * len(m.faces)

    seedpoints = get_seedpoints(num)

    index = rtree.index
    p = index.Property()
    p.dimension = 3
    tree = index.Index(properties=p)

    for i, t in enumerate(m.faces):
        center_point = np.mean(
            np.array(
                [
                    m.points[t[0]],
                    m.points[t[1]],
                    m.points[t[2]],
                ]
            ),
            axis=0,
        )
        tree.insert(i, center_point.tolist())

    for p in seedpoints:
        neighbour = tree.nearest(p, 1)
        nearest_triangle_index = list(neighbour)[0]
        m.colors[nearest_triangle_index] = np.array([255, 0, 0], dtype=np.uint8)

    return m


def normalize_elevation(data: np.ndarray) -> np.ndarray:
    """Normalize to [-1, 1]"""
    return data / max(abs(np.min(data)), abs(np.max(data)))


def load_raster(filename: Path) -> np.ndarray:
    with rasterio.open(filename) as dataset:
        data = dataset.read()
        return data


def _compute_normals(m: Mesh) -> tuple[np.ndarray, np.ndarray]:
    centers = np.zeros([len(m.faces), 3], dtype=np.float32)
    normals = np.zeros([len(m.faces), 3], dtype=np.float32)

    for i in range(len(m.faces)):
        t = m.faces[i]
        a, b, c = m.points[t[0]], m.points[t[1]], m.points[t[2]]
        normals[i, :] = np.cross(b - a, c - a)
        centers[i, :] = np.mean(
            np.array(
                [
                    m.points[t[0]],
                    m.points[t[1]],
                    m.points[t[2]],
                ]
            ),
            axis=0,
        )

    return centers, _normalize_vectors(normals)


def _normalize_vector(v: np.array) -> np.array:
    return v / np.linalg.norm(v)


def _normalize_vectors(v: np.ndarray) -> np.array:
    return v / np.linalg.norm(v, axis=1).reshape(-1, 1)


def _line_plane_intersection(
    plane_normal: np.array,
    plane_point: np.array,
    line_point: np.array,
    line_direction: np.array,
    tol: float = 1e-6,
) -> np.array:
    denom = np.dot(plane_normal, line_direction)

    if abs(denom) < tol:  # the line is parallel to the plane
        if abs(np.dot(plane_normal, line_point - plane_point)) < tol:
            return line_point  # the line lies on the plane
        else:
            return line_direction  # no intersection

    t = -np.dot(plane_normal, line_point - plane_point) / denom
    intersection = line_point + t * line_direction

    return intersection


def _compute_intersections(m: Mesh, centers: np.ndarray, normals: np.ndarray, axis: np.array, tol=1e-6) -> np.ndarray:
    intersections = np.zeros_like(centers)

    line_point = np.array([0, 0, 0], dtype=np.float32)
    for i in range(len(m.faces)):
        intersections[i, :] = _line_plane_intersection(normals[i], centers[i], line_point, axis)

    return intersections


def add_field_vectors(m: Mesh, axis: np.array) -> Mesh:
    centers, normals = _compute_normals(m)
    intersections = _compute_intersections(m, centers, normals, axis)
    directions = _normalize_vectors(intersections - centers)

    # flip direction if intersection point is on the opposite end of the axis
    # necessary to avoid a flipping of direction signs when moving from one face to the next one
    opposite_directions = np.full_like(directions, 1, dtype=np.float32)
    opposite_directions[np.dot(intersections, axis) < 0] = -1
    directions *= opposite_directions

    m.centers = [np.array(e) for e in centers.tolist()]
    m.normals = [np.array(e) for e in normals.tolist()]
    m.field_vectors = [np.array(e) for e in directions.tolist()]

    elevation_vectors = _normalize_vectors(centers)
    field_elevation_vectors = []
    for i in range(len(elevation_vectors)):
        projected = elevation_vectors[i] - (np.dot(elevation_vectors[i], normals[i])) * normals[i]
        magnitude = np.arccos(np.dot(elevation_vectors[i], normals[i]))

        combined = _normalize_vector(
            directions[i] * (1 - magnitude) * (1 - ELEVATION_VECTOR_WEIGHT) + projected * magnitude * ELEVATION_VECTOR_WEIGHT
        )
        field_elevation_vectors.append(combined)

    m.field_elevation_vectors = field_elevation_vectors

    return m


def field_vectors_to_image(m: Mesh, num: int = 360) -> np.ndarray:
    image = np.zeros([num, num, 3], dtype=float)

    index = rtree.index
    p = index.Property()
    p.dimension = 3
    tree = index.Index(properties=p)

    for i, t in enumerate(m.faces):
        center_point = np.mean(
            np.array(
                [
                    m.points[t[0]],
                    m.points[t[1]],
                    m.points[t[2]],
                ]
            ),
            axis=0,
        )
        tree.insert(i, center_point.tolist())

    lats = np.linspace(-90, 90, num, endpoint=False)
    lons = np.linspace(-180, 180, num, endpoint=False)

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            latr = math.radians(lat)
            lonr = math.radians(lon)

            x = 1 * math.cos(latr) * math.cos(lonr)
            y = 1 * math.cos(latr) * math.sin(lonr)
            z = 1 * math.sin(latr)

            neighbour = tree.nearest(np.array([x, y, z]), 1)
            ind = list(neighbour)[0]
            dir = np.array(m.field_vectors[ind])

            image[i, j, :] = dir

    return image


def shift_center_to_origin(raster: np.ndarray) -> np.ndarray:
    """
    shift center of raster image to 1/4. This adjusts for the X/Y axis change.
    atan2 returns angles relative to the X axis, while the default blender orientation
    should have the zero meridian face the Y axis.

    """
    return np.roll(raster, int(raster.shape[1] / 4), axis=1)


def display(m: Mesh) -> None:
    import pyvista as pv

    plotter = pv.Plotter()

    # X, Y, Z axes
    plotter.add_mesh(
        pv.Spline(np.array([[0, 0, 0], [1, 0, 0]]), 10).tube(radius=0.01),
        color=[255, 0, 0],
    )
    plotter.add_mesh(
        pv.Spline(np.array([[0, 0, 0], [0, 1, 0]]), 10).tube(radius=0.01),
        color=[0, 255, 0],
    )
    plotter.add_mesh(
        pv.Spline(np.array([[0, 0, 0], [0, 0, 1]]), 10).tube(radius=0.01),
        color=[0, 0, 255],
    )

    # mesh
    points_pv = np.stack(m.points)
    faces_pv = np.hstack([[3, *face] for face in m.faces])
    pvmesh = pv.PolyData(points_pv, faces_pv)
    plotter.add_mesh(pvmesh, show_edges=True, opacity=0.5)

    # light axis
    light_axis = _normalize_vector(np.array([0, 1, 1]))
    spline = pv.Spline(np.array([[0, 0, 0], light_axis], dtype=np.float32)).tube(radius=0.01)
    plotter.add_mesh(spline, color=[255, 255, 0])

    # normals
    centers, normals = _compute_normals(m)
    # for i in range(len(normals)):
    #     arrow = pv.Arrow(
    #         centers[i],
    #         normals[i],
    #         scale=0.5
    #     )
    #     plotter.add_mesh(arrow, color=[255, 0, 0])

    # intersections
    intersections = _compute_intersections(m, centers, normals, light_axis)
    # for i in range(len(intersections)):
    #     plotter.add_mesh(pv.Sphere(0.05, intersections[i]), color=[0, 0, 255])

    # directions
    directions = _normalize_vectors(intersections - centers)
    for i in range(len(directions)):
        arrow = pv.Arrow(centers[i], directions[i], scale=0.1)
        plotter.add_mesh(arrow, color=[0, 255, 0])

    plotter.show()


def visualize(m: Mesh, lines: list[LineString]) -> pv.Plotter:
    plotter = pv.Plotter()

    # X, Y, Z axes
    plotter.add_mesh(
        pv.Spline(np.array([[0, 0, 0], [1, 0, 0]]), 10).tube(radius=0.02),
        color=[255, 0, 0],
    )
    plotter.add_mesh(
        pv.Spline(np.array([[0, 0, 0], [0, 1, 0]]), 10).tube(radius=0.02),
        color=[0, 255, 0],
    )
    plotter.add_mesh(
        pv.Spline(np.array([[0, 0, 0], [0, 0, 1]]), 10).tube(radius=0.02),
        color=[0, 0, 255],
    )

    # mesh
    points_pv = np.stack(m.points)
    faces_pv = np.hstack([[3, *face] for face in m.faces])
    pvmesh = pv.PolyData(points_pv, faces_pv)
    plotter.add_mesh(pvmesh, opacity=0.5, show_edges=True)  # ), opacity=0.5)

    # light axis
    light_axis = _normalize_vector(np.array([0, 1, 1]))
    spline = pv.Spline(np.array([[0, 0, 0], light_axis], dtype=np.float32), 10).tube(radius=0.02)
    plotter.add_mesh(spline, color=[255, 255, 0])

    for line in lines:
        spline = pv.Spline(np.array(list(line.coords))).tube(radius=0.005)
        plotter.add_mesh(spline, color=[125, 125, 125])

    # normals
    for i in range(len(m.normals)):
        arrow = pv.Arrow(m.centers[i], m.normals[i], scale=0.10)
        plotter.add_mesh(arrow, color=[255, 0, 0])

    # field vectors
    for i in range(len(m.field_vectors)):
        arrow = pv.Arrow(m.centers[i], m.field_vectors[i], scale=0.10)
        plotter.add_mesh(arrow, color=[255, 255, 0])

    # elevation vectors
    for i in range(len(m.centers)):
        arrow = pv.Arrow(m.centers[i], _normalize_vector(m.centers[i]), scale=0.10)
        plotter.add_mesh(arrow, color=[0, 255, 0])

    for i in range(len(m.centers)):
        elevation_vector = _normalize_vector(m.centers[i])
        projected = elevation_vector - (np.dot(elevation_vector, m.normals[i])) * m.normals[i]
        magnitude = np.arccos(np.dot(elevation_vector, m.normals[i]))
        # spline = pv.Spline(np.array([m.centers[i], m.centers[i] + m.field_elevation_vectors[i]], dtype=np.float32), 10).tube(radius=0.005)
        spline = pv.Spline(
            np.array([m.centers[i], m.centers[i] + projected * magnitude], dtype=np.float32),
            10,
        ).tube(radius=0.005)
        plotter.add_mesh(spline, color=[0, 0, 255])

        arrow = pv.Arrow(m.centers[i], m.field_elevation_vectors[i], scale=0.10)
        plotter.add_mesh(arrow, color=[0, 0, 255])

    # centers
    # for i in range(len(m.centers)):
    #     plotter.add_mesh(pv.Sphere(0.01, m.centers[i]), color=[0, 0, 255])

    return plotter


def visualize_lines(m: Mesh, lines: list[LineString]) -> pv.Plotter:
    plotter = pv.Plotter()

    # X, Y, Z axes
    plotter.add_mesh(
        pv.Spline(np.array([[0, 0, 0], [0.5, 0, 0]]), 10).tube(radius=0.01),
        color=[255, 0, 0],
    )
    plotter.add_mesh(
        pv.Spline(np.array([[0, 0, 0], [0, 0.5, 0]]), 10).tube(radius=0.01),
        color=[0, 255, 0],
    )
    plotter.add_mesh(
        pv.Spline(np.array([[0, 0, 0], [0, 0, 0.5]]), 10).tube(radius=0.01),
        color=[0, 0, 255],
    )

    # mesh
    points_pv = np.stack(m.points)
    faces_pv = np.hstack([[3, *face] for face in m.faces])
    pvmesh = pv.PolyData(points_pv, faces_pv)
    plotter.add_mesh(pvmesh)  # , opacity=0.2, show_edges=True)

    # light axis
    light_axis = _normalize_vector(np.array([0, 1, 1])) * 0.5
    spline = pv.Spline(np.array([[0, 0, 0], light_axis], dtype=np.float32)).tube(radius=0.01)
    plotter.add_mesh(spline, color=[255, 255, 0])

    for line in lines:
        plotter.add_lines(np.array(list(line.coords)), color="purple", width=3, connected=True)

    return plotter


def write_obj(plotter: pv.Plotter, filename: Path) -> None:
    if not filename.suffix == ".obj":
        filename = filename.parent / (filename.name + ".obj")
    plotter.export_obj(filename)


class MeshConfig(BaseModel):
    scale: float = Field(default=0.10, description="Scaling factor")
    fixed_elevation_scale: float | None = Field(
        default=None, description="A fixed scale value [-1, +1] that uniformly overrides elevation raster data"
    )
    blur: int = Field(default=100, description="Elevation raster blurring kernel size")
    subdivision: int = Field(default=10, description="Number of subdivision steps")


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
    # color_raster2 = load_raster(Path("data", "BasicHapke_wbhs_blend_b137_IOF.55deg_nearcenter_1kmpp_20140925_134515.tif")) # [3, rows, cols]
    # empty = np.zeros([color_raster2.shape[0], int(color_raster2.shape[1] * 25/90), color_raster2.shape[2]], dtype=np.uint8)
    # color_raster2 = np.concatenate([empty, color_raster2, empty], axis=1)
    # color_raster = color_raster2 # TODO

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
    exit()

    # 3D HATCHING / TODO: remove

    # poly = subdivide(tetrahedron(), n=args.subdivision)
    #
    # dem_raster = normalize_elevation(load_raster(args.elevation_raster))[0, :, :]
    # size = (np.array([dem_raster.shape[1], dem_raster.shape[0]]) * 0.25).astype(int).tolist()
    # dem_raster = cv2.resize(dem_raster, size)
    #
    # poly = project(poly, dem_raster, scale=args.scale)
    # poly = add_field_vectors(poly, _normalize_vector(np.array([0, 1, 1])))
    #
    # color_raster = load_raster(args.color_raster)[0:3, :, :]
    # add_color(poly, color_raster)
    # write_ply(poly, args.output)
    #
    # import flowlines3
    #
    # config = flowlines3.FlowlineHatcherConfig()
    # # mapping_line_distance = np.mean(color_raster[0:3, :, :], axis=0)
    # lines = flowlines3.FlowlineHatcher(
    #     poly,
    #     1 + dem_raster * SCALE,
    #     np.zeros([1, 1], dtype=np.uint8),  # mapping_line_distance,
    #     np.zeros([1, 1], dtype=np.uint8),
    #     np.zeros([1, 1], dtype=np.uint8),
    #     config,
    #     initial_seed_points=poly.centers,
    # ).hatch()
    #
    # print(f"lines: {len(lines)}")
    # print("Completed in: {:.3f}s".format((datetime.datetime.now() - timer_start).total_seconds()))
    #
    # # points = []
    # # for line in lines:
    # #     points += [np.array(p) for p in line.coords]
    # # pointcloud = Mesh(points, [])
    # # write_ply_pointcloud(pointcloud, Path("pointcloud.ply"))
    #
    # # plotter = visualize(poly, lines)
    # # write_obj(plotter, Path("scene.obj"))
    # # plotter.show()
    #
    # # visualize(poly, []).show()
    #
    # visualize_lines(poly, lines).show()


if __name__ == "__main__":
    main()
