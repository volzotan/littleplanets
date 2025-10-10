import argparse
import csv
import json
from pathlib import Path
import trimesh
import shapely
from shapely.geometry import LineString, Point, MultiLineString
import numpy as np
import math
import cv2
import rasterio
import pyvista as pv
from hershey import HersheyFont, Align

import rtree

from util.misc import linestring_to_coordinate_pairs

VISUALIZE = False
PROJECT_ONTO_SURFACE = True

GRID_NUM_SEGMENTS = 100


def _rotate_linestrings(lines: list[LineString], x: float, y: float, z: float) -> list[LineString]:
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

    lines_rotated = []
    for line in lines:
        lines_rotated.append(shapely.ops.transform(lambda x, y, z: R_z @ R_y @ R_x @ np.array([x, y, z]), line))

    return lines_rotated


def _map(raster: np.ndarray, lat: float, lon: float) -> np.ndarray:
    height, width = raster.shape

    y = int(((lat / math.pi) + 0.0) * height)
    x = int(((lon / math.tau) + 0.0) * width)

    return raster[y, x]


def project(raster: np.ndarray, points: np.ndarray, scale: float) -> np.ndarray:
    proj = np.zeros_like(points)

    r = np.sqrt(np.sum(np.power(points, 2), axis=1))
    lats = np.acos(points[:, 2] / r)
    lons = np.atan2(points[:, 1], points[:, 0])

    for i in range(proj.shape[0]):
        dist = 1 + _map(raster, lats[i], lons[i]) * scale

        x = dist * math.sin(lats[i]) * math.cos(lons[i])
        y = dist * math.sin(lats[i]) * math.sin(lons[i])
        z = dist * math.cos(lats[i])

        proj[i, :] = [x, y, z]

    return proj


def _map_vertices(raster: np.ndarray, lat: float, lon: float) -> np.ndarray:
    height, width = raster.shape

    y = int(((lat / math.pi) + 0.0) * height)
    x = int(((lon / math.tau) + 0.0) * width)

    return raster[y, x]


def project_vertices(tree: rtree.Index, points: np.ndarray, scale: float) -> np.ndarray:
    proj = np.zeros_like(points)

    r = np.sqrt(np.sum(np.power(points, 2), axis=1))
    lats = np.acos(points[:, 2] / r)
    lons = np.atan2(points[:, 1], points[:, 0])

    for i in range(proj.shape[0]):
        nearest_neighbor = list(tree.nearest(points[i, :], 1, objects="raw"))[0]
        dist = np.linalg.norm(nearest_neighbor)

        x = dist * math.sin(lats[i]) * math.cos(lons[i])
        y = dist * math.sin(lats[i]) * math.sin(lons[i])
        z = dist * math.cos(lats[i])

        proj[i, :] = [x, y, z]

    return proj


def normalize_elevation(data: np.ndarray) -> np.ndarray:
    """Normalize to [-1, 1]"""
    return data / max(abs(np.min(data)), abs(np.max(data)))


def load_raster(filename: Path) -> np.ndarray:
    with rasterio.open(filename) as dataset:
        data = dataset.read()
        return data


def visualize(linestrings: list[LineString]) -> pv.Plotter:
    plotter = pv.Plotter()

    plotter.camera.position = (0.0, 0.0, 5.0)
    plotter.camera.focal_point = (0.0, 0.0, 0.0)
    plotter.camera.up = (0, 1, 0)

    # X, Y, Z axes
    plotter.add_mesh(
        pv.Spline(np.array([[0, 0, 0], [1, 0, 0]]), 10).tube(radius=0.005),
        color=[255, 0, 0],
    )
    plotter.add_mesh(
        pv.Spline(np.array([[0, 0, 0], [0, 1, 0]]), 10).tube(radius=0.005),
        color=[0, 255, 0],
    )
    plotter.add_mesh(
        pv.Spline(np.array([[0, 0, 0], [0, 0, 1]]), 10).tube(radius=0.005),
        color=[0, 0, 255],
    )

    colors = ["red", "green", "blue"]

    for li, linestring in enumerate(linestrings):
        for pair in linestring_to_coordinate_pairs(linestring):
            spline = pv.Spline(np.array(pair, dtype=np.float32)).tube(radius=0.005)
            plotter.add_mesh(spline, color=colors[li % 3])

    return plotter


def write_csv(filename: Path, linestrings: list[LineString]) -> None:
    with open(filename, "w") as csvfile:
        writer = csv.writer(csvfile, delimiter=" ")

        for ls in linestrings:
            writer.writerow(shapely.get_coordinates(ls, include_z=True).tolist())


def write_npz(filename: Path, linestrings: list[LineString]) -> None:
    arrays = [shapely.get_coordinates(l, include_z=True) for l in linestrings]
    np.savez(filename, *arrays)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", type=Path, help="Input mesh path [PLY]")
    parser.add_argument("pois", type=Path, help="Position of interest data [JSON]")
    parser.add_argument("--rotX", type=float, default=0, help="rotation X in degrees [float]")
    parser.add_argument("--rotY", type=float, default=0, help="rotation Y in degrees [float]")
    parser.add_argument("--rotZ", type=float, default=0, help="rotation Z in degrees [float]")
    parser.add_argument("--grid-num-lat", type=int, default=0, help="number of latitude grid lines")
    parser.add_argument("--grid-num-lon", type=int, default=0, help="number of longitude grid lines")
    parser.add_argument("--output", type=Path, default="overlay.npz", help="Output filename [NPZ]")
    parser.add_argument("--circle-radius", type=float, default=0.02, help="POI circle radius [float]")
    parser.add_argument("--font-size", type=float, default=0.03, help="Label font size [float]")
    parser.add_argument("--subdivision", type=int, default=10, help="Number of subdivision steps [int]")
    args = parser.parse_args()

    DEFAULT_ROTATION = np.array([-math.pi / 2, 0, 0])  # Blender Camera is aligned with the Z axis

    # PLY exported from Blender is already correctly rotated with regard to Z axis up
    # but Lat/Lon needs to be adjusted for any additional rotation
    BLENDER_ROTATION = np.array([np.radians(c) for c in [args.rotX, args.rotY, args.rotZ]])

    linestrings = []

    # DRAW POINTS OF INTEREST

    pois = []
    with open(args.pois) as f:
        pois = json.load(f)

    font = HersheyFont(font_file=Path(HersheyFont.DEFAULT_FONT))

    for poi in pois:
        circle = Point([0, 0]).buffer(args.circle_radius).boundary.segmentize(0.05)

        # circular text
        # path_text_baseline = Point([0, 0]).buffer(CIRCLE_RADIUS + 0.01).boundary.segmentize(0.01)
        # linestrings_along_path = font.lines_for_text(
        #     poi["name"], FONT_SIZE, path=path_text_baseline, align=Align.CENTER, reverse_path=True
        # )

        # linear text
        linestrings_along_path = font.lines_for_text(poi["name"], args.font_size)

        angle = poi.get("label_angle", 0.0)
        dist = args.circle_radius * 1.30
        geom = MultiLineString(linestrings_along_path)

        x = 0
        y = geom.bounds[3] / 2  # vertical align

        if math.isclose(angle, 90):
            x += -geom.bounds[2] / 2
        elif math.isclose(angle, 270):
            x += -geom.bounds[2] / 2
        elif angle > 90 and angle < 270:
            x += -geom.bounds[2]
        else:
            pass

        linestrings_along_path = [shapely.affinity.translate(ls, xoff=x, yoff=y) for ls in linestrings_along_path]
        linestrings_along_path = [
            shapely.affinity.translate(ls, xoff=dist * math.cos(math.radians(angle)), yoff=dist * math.sin(math.radians(angle)))
            for ls in linestrings_along_path
        ]
        text = [LineString(shapely.get_coordinates(l) * np.array([1, -1])) for l in linestrings_along_path]
        ls_poi = [circle] + text

        for i in range(len(ls_poi)):  # add Z
            ls = ls_poi[i]
            coords = shapely.get_coordinates(ls)
            new_col = np.full([coords.shape[0], 1], 1.0)
            coords_with_z = np.concatenate((coords, new_col), axis=1)
            ls_poi[i] = LineString(coords_with_z)

        poi_rot_x = (poi["lat"] * -1 + 90.0) / 180 * math.pi
        poi_rot_z = (poi["lon"]) / 360 * math.tau

        ls_rotated = ls_poi
        ls_rotated = _rotate_linestrings(ls_poi, *[poi_rot_x, 0, poi_rot_z])

        linestrings += ls_rotated

    #  GRID

    grid_linestrings = []

    grid_num_lat_lines = args.grid_num_lat
    grid_num_lon_lines = args.grid_num_lon

    points = [math.tau * i / GRID_NUM_SEGMENTS for i in range(GRID_NUM_SEGMENTS)]
    points = [[0, math.cos(angle), math.sin(angle)] for angle in points]
    points = points + [points[0]]
    for i in range(grid_num_lat_lines):
        grid_linestrings += _rotate_linestrings([LineString(points)], 0, 0, math.pi * i / grid_num_lat_lines)

    lons = [1.0 / (grid_num_lon_lines + 1) * (i + 1) for i in range(grid_num_lon_lines)]
    for lon in lons:
        z = lon * 2 - 1
        y = math.sqrt(1 - z**2)
        points = [math.tau * i / GRID_NUM_SEGMENTS for i in range(GRID_NUM_SEGMENTS)]
        points = [[math.cos(angle) * y, math.sin(angle) * y, z] for angle in points]
        points = points + [points[0]]
        grid_linestrings.append(LineString(points))

    linestrings += grid_linestrings

    # ROTATE

    linestrings = _rotate_linestrings(linestrings, *BLENDER_ROTATION)

    # VISUALIZE

    if VISUALIZE:
        visualize(linestrings).show()
        exit()

    # PROJECT ONTO SURFACE

    # TODO: PLY mesh should already be correctly rotated. Does this work as expected?

    if PROJECT_ONTO_SURFACE:
        mesh = trimesh.load(args.mesh)
        index = rtree.index
        p = index.Property()
        p.dimension = 3
        tree = index.Index(properties=p)
        for i, v in enumerate(mesh.vertices.tolist()):
            tree.insert(i, v, obj=v)

        linestrings = [LineString(project_vertices(tree, shapely.get_coordinates(l, include_z=True), 0.1)) for l in linestrings]

    # TODO: occlusion check

    # EXPORT

    write_npz(args.output, linestrings)
