import math
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import LineString, Point
from shapely.ops import transform
import pyvista as pv


def linestring_to_coordinate_pairs(linestring: LineString) -> list[list[tuple[float, float]]]:
    pairs = []

    for i in range(len(linestring.coords) - 1):
        pairs.append([linestring.coords[i], linestring.coords[i + 1]])

    return pairs


def write_linestrings_to_npz(filename: Path, linestrings: list[LineString], include_z=True) -> None:
    arrays = [shapely.get_coordinates(l, include_z=include_z) for l in linestrings]
    np.savez(filename, *arrays)


def _linestring_z_length(line: LineString) -> float:
    coords = list(line.coords)
    return sum(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) for (x1, y1, z1), (x2, y2, z2) in zip(coords[:-1], coords[1:]))


def _linestring_z_interpolate(line: LineString, distance: float) -> Point:
    coords = np.array(line.coords)

    # Compute 3D segment lengths
    seg_lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    total_length = seg_lengths.sum()

    if distance <= 0:
        return Point(*coords[0])
    if distance >= total_length:
        return Point(*coords[-1])

    # Find which segment contains the target distance
    cumdist = np.cumsum(seg_lengths)
    idx = np.searchsorted(cumdist, distance)
    prev_len = cumdist[idx - 1] if idx > 0 else 0.0

    # Fraction along this segment
    seg_frac = (distance - prev_len) / seg_lengths[idx]
    p1, p2 = coords[idx], coords[idx + 1]
    interp = p1 + seg_frac * (p2 - p1)

    return Point(*interp)


def dash_linestring(linestring: LineString, dash_length: float, pause_length: float, step_size: float = 1e-3) -> list[LineString]:
    if linestring.is_empty:
        return []
    if dash_length <= 0 or pause_length < 0:
        raise ValueError("dash_length must be > 0 and pause_length must be >= 0")

    total_length = _linestring_z_length(linestring)
    dash_segments = []
    position = 0.0

    while position < total_length:
        start_pos = position
        end_pos = min(position + dash_length, total_length)

        segment_points = [_linestring_z_interpolate(linestring, start_pos)]

        # Collect intermediate points for the dash
        current = start_pos + step_size
        while current < end_pos:
            segment_points.append(_linestring_z_interpolate(linestring, current))
            current += step_size
        segment_points.append(_linestring_z_interpolate(linestring, end_pos))

        # Create a new LineString for this dash
        dash_line = LineString(segment_points)
        dash_segments.append(dash_line)

        # Move to the next dash start (skip the pause)
        position += dash_length + pause_length

    return dash_segments


def rotate_linestrings(lines: list[LineString], x: float, y: float, z: float) -> list[LineString]:
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
        lines_rotated.append(transform(lambda x, y, z: R_z @ R_y @ R_x @ np.array([x, y, z]), line))

    return lines_rotated


def rotate_points(points: list[np.ndarray], x: float, y: float, z: float) -> list[np.ndarray]:
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

    return [R_z @ R_y @ R_x @ p for p in points]


def rotate_points_inv(points: list[np.ndarray], x: float, y: float, z: float) -> list[np.ndarray]:
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

    return [(R_z @ R_y @ R_x).T @ p for p in points]


def visualize_linestrings(linestrings: list[LineString]) -> pv.Plotter:
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
            # discard any linestrings with almost zero length
            if np.all(np.isclose(pair[0], pair[1])):
                continue

            spline = pv.Spline(np.array(pair, dtype=np.float32)).tube(radius=0.002)
            plotter.add_mesh(spline, color=colors[li % 3])

    return plotter
