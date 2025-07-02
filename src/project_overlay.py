import csv
import json
from pathlib import Path
import trimesh
import shapely
from shapely.geometry import LineString, Point
import numpy as np
import math
import cv2
import rasterio
import pyvista as pv
from hershey import HersheyFont, Align

import rtree


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


def _linestring_to_coordinate_pairs(
    linestring: LineString,
) -> list[list[tuple[float, float]]]:
    pairs = []

    for i in range(len(linestring.coords) - 1):
        pairs.append([linestring.coords[i], linestring.coords[i + 1]])

    return pairs


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
        for pair in _linestring_to_coordinate_pairs(linestring):
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
    ASSETS_DIR = Path("..", "assets")
    ASSETS_LOWRES_DIR = Path("..", "assets_lowres")

    INPUT_MESH = ASSETS_DIR / "Moon_Z.ply"
    POI_DATA = ASSETS_DIR / "Moon_apollo_landing_sites.json"

    OUTPUT_DIR = ASSETS_DIR
    output_filename = "Moon_linestrings_overlay.npz"

    DEFAULT_ROTATION = np.array([-math.pi / 2, 0, 0])  # Blender Camera is aligned with the Z axis

    # PLY exported from Blender is already correctly rotated with regard to Z axis up
    # but Lat/Lon needs to be adjusted for any additional rotation
    BLENDER_ROTATION = np.array([np.radians(c) for c in [0, 0, -45]])

    CIRCLE_RADIUS = 0.03
    FONT_SIZE = 0.025

    linestrings = []
    # linestrings.append(Point([0, 0]).buffer(0.10).boundary.segmentize(0.1))
    # linestrings.append(Point([0, 0]).buffer(0.20).boundary.segmentize(0.1))
    # linestrings.append(Point([0, 0]).buffer(0.30).boundary.segmentize(0.1))

    # DRAW POINTS OF INTEREST

    pois = []
    with open(POI_DATA) as f:
        pois = json.load(f)

    font = HersheyFont(font_file=Path(HersheyFont.DEFAULT_FONT))

    for poi in pois:
        circle = Point([0, 0]).buffer(CIRCLE_RADIUS).boundary.segmentize(0.05)
        path_text_baseline = Point([0, 0]).buffer(CIRCLE_RADIUS + 0.01).boundary.segmentize(0.01)

        # linestrings_along_path1 = font.lines_for_text(
        #     poi["name"], FONT_SIZE, path=path_text_baseline, align=Align.CENTER, reverse_path=True
        # )
        linestrings_along_path1 = font.lines_for_text(poi["name"], FONT_SIZE)
        text = [LineString(shapely.get_coordinates(l) * np.array([1, -1])) for l in linestrings_along_path1]
        ls_poi = [circle] + text

        for i in range(len(ls_poi)):  # add Z
            ls = ls_poi[i]
            coords = shapely.get_coordinates(ls)
            new_col = np.full([coords.shape[0], 1], 1.0)
            coords_with_z = np.concatenate((coords, new_col), axis=1)
            ls_poi[i] = LineString(coords_with_z)

        rot_x = (poi["lat"] * -1 + 90.0) / 180 * math.pi
        rot_z = (poi["lon"]) / 360 * math.tau

        ls_rotated = ls_poi
        ls_rotated = _rotate_linestrings(ls_rotated, *[0, 0, -BLENDER_ROTATION[2]])
        ls_rotated = _rotate_linestrings(ls_rotated, *[rot_x, 0, rot_z])
        ls_rotated = _rotate_linestrings(ls_rotated, *DEFAULT_ROTATION)
        ls_rotated = _rotate_linestrings(ls_rotated, *BLENDER_ROTATION)

        linestrings += ls_rotated

    # visualize(linestrings).show()
    # exit()

    # PROJECT ONTO SURFACE
    # TODO: Caveat: height data derived from the DEM raster is missing blender mesh rotation info

    # dem_raster = normalize_elevation(load_raster(ASSETS_LOWRES_DIR / "Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif"))
    # dem_raster = dem_raster[0, :, :]
    # size = (np.array([dem_raster.shape[1], dem_raster.shape[0]]) * 0.25).astype(int).tolist()
    # dem_raster = cv2.resize(dem_raster, size)

    # linestrings_projected = [
    #     LineString(project(dem_raster, shapely.get_coordinates(l, include_z=True), 0.1)) for l in linestrings
    # ]

    mesh = trimesh.load(INPUT_MESH)
    index = rtree.index
    p = index.Property()
    p.dimension = 3
    tree = index.Index(properties=p)
    for i, v in enumerate(mesh.vertices.tolist()):
        tree.insert(i, v, obj=v)

    linestrings_projected = [
        LineString(project_vertices(tree, shapely.get_coordinates(l, include_z=True), 0.1))
        for l in linestrings
    ]

    # visualize(linestrings_projected + linestrings_projected2).show()
    # visualize(linestrings_projected).show()

    # EXPORT

    # write_npz(OUTPUT_DIR / output_filename, linestrings)
    write_npz(OUTPUT_DIR / output_filename, linestrings_projected)
    # write_npz(OUTPUT_DIR / output_filename, linestrings)
