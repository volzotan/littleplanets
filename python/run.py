import math
import os
from dataclasses import dataclass
from pathlib import Path
import textwrap
from typing import Any

import rasterio
import rtree
from stl import mesh
import numpy as np

import datetime


@dataclass
class Mesh:
    points: list[np.array]
    triangles: list[tuple[int, int, int]]
    colors: list[np.array]

    def __repr__(self):
        return f"Mesh [ points: {len(self.points)} / triangles: {len(self.triangles)} / colors: {len(self.colors)} ]"


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

    return Mesh(points, triangles, [])


def write_stl(m: Mesh, filename: Path) -> None:
    obj = mesh.Mesh(np.zeros(len(m.triangles), dtype=mesh.Mesh.dtype))

    for i, t in enumerate(m.triangles):
        obj.vectors[i] = np.array([m.points[i] for i in t])

    obj.save(str(filename))


def write_ply(m: Mesh, filename: Path, color_vertices=False) -> None:
    if not color_vertices:  # triangles
        with open(filename, "w") as f:
            data = f"""
                    ply
                    format ascii 1.0
                    element vertex {len(m.points)}
                    property float x
                    property float y
                    property float z
                    element face {len(m.triangles)}
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

            for i in range(len(m.triangles)):
                i0, i1, i2 = m.triangles[i]
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
                    element face {len(m.triangles)}
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

            for i0, i1, i2 in m.triangles:
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
            c0, c1, c2 = m.colors[i]

            f.write(f"{p0:.4f} {p1:.4f} {p2:.4f} {c0:d} {c1:d} {c2:d}\n")

        f.write("\n")


def subdivide(m: Mesh, n: int) -> Mesh:
    if n <= 0:
        return m

    output_points = m.points
    output_triangles = []

    map_points: dict[str, int] = {}

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

        output_triangles.append((i0, i_alpha, i_gamma))
        output_triangles.append((i_alpha, i1, i_beta))
        output_triangles.append((i_beta, i2, i_gamma))
        output_triangles.append((i_gamma, i_alpha, i_beta))

    return subdivide(Mesh(output_points, output_triangles, []), n - 1)


def _map(raster: np.ndarray, lat: float, lon: float) -> np.ndarray:
    height, width = raster.shape[1:3]

    y = int(((lat / math.pi) + 0.0) * height)
    x = int(((lon / math.tau) + 0.0) * width)

    return raster[:, y, x]


def project(m: Mesh, raster: np.ndarray | None, scale: float = 0.1) -> Mesh:
    for i in range(len(m.points)):
        p = m.points[i]

        d = math.sqrt(np.sum(np.power(p, 2)))
        lat = math.acos(p[2] / d)
        lon = math.atan2(p[1], p[0])

        r = 1.0
        if raster is not None:
            r += _map(raster, lat, lon)[0] * scale

        x = r * math.sin(lat) * math.cos(lon)
        y = r * math.sin(lat) * math.sin(lon)
        z = r * math.cos(lat)

        m.points[i] = np.array([x, y, z])

    return m


def add_color(m: Mesh, raster: np.ndarray) -> Mesh:
    m.colors = []
    for i in range(len(m.triangles)):  # color matched to triangles, not vertices
        p = m.points[m.triangles[i][0]]

        d = math.sqrt(np.sum(np.power(p, 2)))
        lat = math.acos(p[2] / d)
        lon = math.atan2(p[1], p[0])

        c = _map(raster, lat, lon).astype(np.uint8)
        m.colors.append(c)

    return m


def fill_color(m: Mesh, value: list[int] = [255, 255, 255]) -> Mesh:
    m.colors = []

    for _ in range(len(m.triangles)):
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
    m.colors = [np.array([255, 255, 255], dtype=np.uint8)] * len(m.triangles)

    seedpoints = get_seedpoints(num)

    index = rtree.index
    p = index.Property()
    p.dimension = 3
    tree = index.Index(properties=p)

    for i, t in enumerate(m.triangles):
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


def add_normal_vectors(m: Mesh) -> Mesh:
    m.colors = []

    for i in range(len(m.triangles)):
        t = m.triangles[i]
        a, b, c = m.points[t[0]], m.points[t[1]], m.points[t[2]]

        normal_dir = np.cross(b - a, c - a)
        normal_vector = normal_dir / np.linalg.norm(normal_dir)

        vector_rgb = (normal_vector + 1) * 255 / 2
        m.colors.append(vector_rgb.astype(np.uint8))

    return m


def normalize_elevation(data: np.ndarray) -> np.ndarray:
    """Normalize to [-1, 1]"""
    return data / max(abs(np.min(data)), abs(np.max(data)))


def _normalize_vector(x) -> np.array:
    return x / np.linalg.norm(x)


def _initial_bearing(n, v1) -> Any:
    n = _normalize_vector(n)
    v1 = _normalize_vector(v1)

    v3 = v1 - (np.dot(v1, n) * n)

    # u muss orthogonal zu n sein UND in der Ebene liegen die n und 0, 0, 1 (Z-Achse) aufspannt
    z = np.array([0, 0, 1])
    u = z - (np.dot(n, z) / np.dot(n, n)) * n
    u = _normalize_vector(u)

    w = np.cross(n, u)

    x = np.dot(v3, u)
    y = np.dot(v3, w)

    return np.degrees(np.arctan2(y, x)) + 90


def add_field_vectors(m: Mesh) -> Mesh:
    m.colors = []

    for i in range(len(m.triangles)):
        t = m.triangles[i]
        a, b, c = m.points[t[0]], m.points[t[1]], m.points[t[2]]
        # normal_dir = np.cross(b-a, c-a)
        # normal_vector = normalize(normal_dir)

        angle = _initial_bearing(a, np.array([1, 1, 1]))
        angle = ((angle) / 180) * 255

        vector_rgb = np.array([angle, angle, angle])
        m.colors.append(vector_rgb.astype(np.uint8))

    return m


def add_field_vectors_new(m: Mesh) -> Mesh:
    m.colors = []

    for i in range(len(m.triangles)):
        t = m.triangles[i]
        a, b, c = m.points[t[0]], m.points[t[1]], m.points[t[2]]
        # normal_dir = np.cross(b-a, c-a)
        # normal_vector = normalize(normal_dir)

        vector_rgb = np.array([angle, angle, angle])
        m.colors.append(vector_rgb.astype(np.uint8))

    return m


def load_raster(filename: Path) -> np.ndarray:
    with rasterio.open(filename) as dataset:
        data = dataset.read()
        return data


def _compute_normals(m: mesh) -> tuple[np.ndarray, np.ndarray]:
    centers = np.zeros([len(m.triangles), 3], dtype=np.float32)
    normals = np.zeros([len(m.triangles), 3], dtype=np.float32)

    for i in range(len(m.triangles)):
        t = m.triangles[i]
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

    return centers, normals


import numpy as np


def _line_plane_intersection(plane_normal, plane_point, line_point, line_direction, tol=1e-6):
    n = np.array(plane_normal)
    p0 = np.array(plane_point)
    l0 = np.array(line_point)
    d = np.array(line_direction)

    denom = np.dot(n, d)

    if abs(denom) < tol:
        # The line is parallel to the plane
        if abs(np.dot(n, l0 - p0)) < tol:
            return l0  # The line lies on the plane
        else:
            return None  # No intersection

    t = -np.dot(n, l0 - p0) / denom
    intersection = l0 + t * d
    return intersection


def _compute_intersections(m: mesh, centers: np.ndarray, normals: np.ndarray, axis: np.array, tol=1e-6) -> np.ndarray:
    intersections = np.zeros_like(centers)

    line_point = np.array([0, 0, 0], dtype=np.float32)
    for i in range(len(m.triangles)):
        n = normals[i]
        p0 = centers[i]
        l0 = line_point
        d = axis

        denom = np.dot(n, d)

        if abs(denom) < tol:  # the line is parallel to the plane
            if abs(np.dot(n, l0 - p0)) < tol:
                return l0  # the line lies on the plane
            else:
                return (
                    None  # no intersection # TODO: would it make sense to use the direction of the line in this case?
                )

        t = -np.dot(n, l0 - p0) / denom
        intersection = l0 + t * d

        intersections[i, :] = intersection if intersection is not None else [np.nan, np.nan, np.nan]

    return intersections


def display(m: Mesh) -> None:
    import pyvista as pv

    plotter = pv.Plotter()

    # X, Y, Z axes
    plotter.add_mesh(
        pv.Spline(np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32), 10).tube(radius=0.01), color=[255, 0, 0]
    )
    plotter.add_mesh(
        pv.Spline(np.array([[0, 0, 0], [0, 1, 0]], dtype=np.float32), 10).tube(radius=0.01), color=[0, 255, 0]
    )
    plotter.add_mesh(
        pv.Spline(np.array([[0, 0, 0], [0, 0, 1]], dtype=np.float32), 10).tube(radius=0.01), color=[0, 0, 255]
    )

    # mesh
    points_pv = np.stack(m.points)
    faces_pv = np.hstack([[3, *face] for face in m.triangles])
    pvmesh = pv.PolyData(points_pv, faces_pv)
    plotter.add_mesh(pvmesh, show_edges=True, opacity=0.5)

    # light axis
    light_axis = _normalize_vector([0, 1, 1])
    spline = pv.Spline(np.array([[0, 0, 0], light_axis], dtype=np.float32), 10).tube(radius=0.01)
    plotter.add_mesh(spline, color=[255, 255, 0])

    # normals
    centers, normals = _compute_normals(m)
    # for i in range(len(normals)):
    #     arrow = pv.Arrow(
    #         centers[i],
    #         centers[i] + normals[i],
    #         scale=0.5
    #     )
    #     plotter.add_mesh(arrow, color=[255, 0, 0])

    # intersections
    intersections = _compute_intersections(m, centers, normals, light_axis)
    # for i in range(len(intersections)):
    #     plotter.add_mesh(pv.Sphere(0.05, intersections[i]), color=[0, 0, 255])

    # directions
    directions = _normalize_vector(intersections - centers)
    for i in range(len(directions)):
        arrow = pv.Arrow(centers[i], directions[i], scale=0.1)
        plotter.add_mesh(arrow, color=[0, 255, 0])

    plotter.show()


if __name__ == "__main__":
    asset_dir = Path("..", "assets")
    os.makedirs(asset_dir, exist_ok=True)

    timer_start = datetime.datetime.now()

    # use a partial DEM (lat not from -90 to 90)
    # color_mapping = load_raster(Path("BasicHapke_wbhs_blend_b137_IOF.55deg_nearcenter_1kmpp_20140925_134515.tif"))
    # empty = np.zeros([color_mapping.shape[0], int(color_mapping.shape[1] * 25/90), color_mapping.shape[2]], dtype=np.uint8)
    # color_mapping = np.concatenate([empty, color_mapping, empty], axis=1)

    # poly = project(
    #     subdivide(tetrahedron(), n=10),
    #     normalize_elevation(load_raster(Path(asset_dir, "Lunar_DEM_resized.tif"))),
    #     load_raster(Path(asset_dir, "lroc_color_poles.tif"))[0:3, :, :],
    #     scale=3e-2,
    # )
    #
    # write_ply_with_vertex_colors(poly, Path(asset_dir, "output.ply"))
    # write_stl(poly, Path(asset_dir, "output.stl"))

    # write_stl(project_to_sphere(subdivide(tetrahedron(), n=1)), Path(asset_dir, "unit_sphere_n1.stl"))

    # poly = set_direction_vectors(project_to_sphere(subdivide(cube(), n=1)))
    # write_ply_with_triangle_colors(poly, Path(asset_dir, "direction_sphere.ply"))

    # poly = project(
    #     subdivide(tetrahedron(), n=10),
    #     normalize_elevation(load_raster(Path(asset_dir, "Lunar_DEM_resized.tif"))),
    #     load_raster(Path(asset_dir, "lroc_color_poles.tif"))[0:3, :, :],
    #     scale=3e-2,
    # )
    # write_ply_with_triangle_colors(set_direction_vectors(poly), Path(asset_dir, "direction_sphere.ply"))

    # poly = add_normal_vectors(project_to_sphere(subdivide(cube(), n=5)))
    # write_ply_with_triangle_colors(poly, Path(asset_dir, "direction_sphere.ply"))

    # poly = add_field_vectors(project_to_sphere(subdivide(cube(), n=5)))
    # write_ply_with_triangle_colors(poly, Path(asset_dir, "direction_sphere.ply"))

    # poly = project(
    #     subdivide(tetrahedron(), n=10),
    #     normalize_elevation(load_raster(Path(asset_dir, "Lunar_DEM_resized.tif"))),
    #     load_raster(Path(asset_dir, "lroc_color_poles.tif"))[0:3, :, :],
    #     scale=1e-1,
    # )
    #
    # write_ply_with_triangle_colors(
    #     add_field_vectors(poly), Path(asset_dir, "direction_sphere.ply")
    # )

    # poly = subdivide(tetrahedron(), n=10)
    # poly = add_color(poly)
    # write_ply_with_triangle_colors(poly, Path(asset_dir, "test.ply"))

    # poly = subdivide(tetrahedron(), n=10)
    # poly = project(
    #     poly,
    #     normalize_elevation(load_raster(Path(asset_dir, "Lunar_DEM_resized.tif"))),
    #     scale=1e-1,
    # )
    # # poly = add_color(poly, load_raster(Path(asset_dir, "lroc_color_poles.tif"))[0:3, :, :])
    #
    # poly = add_seedpoints(poly, 1000)
    #
    # write_ply(
    #     poly,
    #     Path(asset_dir, "flow.ply")
    # )
    #
    # print("Completed in: {:.3f}s".format((datetime.datetime.now() - timer_start).total_seconds()))

    # points = get_seedpoints(1000)
    # mesh = Mesh(points, [], [np.array([255, 255, 255])] * len(points))
    # write_ply_pointcloud(mesh, Path("pointcloud.ply"))

    # poly = subdivide(tetrahedron(), n=8)
    # poly = project(
    #     poly,
    #     normalize_elevation(load_raster(Path(asset_dir, "Lunar_DEM_resized.tif"))),
    #     scale=1e-1,
    # )
    # # poly = add_color(poly, load_raster(Path(asset_dir, "lroc_color_poles.tif"))[0:3, :, :])
    #
    # poly = add_seedpoints(poly, 1000)
    #
    # write_ply(
    #     poly,
    #     Path(asset_dir, "flow.ply")
    # )
    #
    # print("Completed in: {:.3f}s".format((datetime.datetime.now() - timer_start).total_seconds()))

    poly = subdivide(tetrahedron(), n=2)
    poly = project(poly, None)
    poly = add_field_vectors(poly)
    display(poly)
    # write_ply(poly, Path(asset_dir, "flow.ply"))

    print("Completed in: {:.3f}s".format((datetime.datetime.now() - timer_start).total_seconds()))
