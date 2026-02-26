import argparse
from pathlib import Path

import toml
from pydantic import BaseModel
from shapely.geometry import LineString, Polygon
import numpy as np
import math

from util.misc import write_linestrings_to_npz, visualize_linestrings, normalize_vectors, normalize_vector

import cv2

MAX_SEGMENT_LENGTH = 10.0


class OverlayContoursConfig(BaseModel):
    double_line_distance: float | None = None
    shrink: float | None = None
    simplify: float | None = None


class AdjustSceneConfig(BaseModel):
    horizontal_width: float = 2.2
    camera_focal_length: float = 50


def _calculate_z_distance_circle(focal_length: float, radius: float, sensor_size: float = 36.0) -> float:
    """duplicated code from blender/adjust_scene.py"""
    fov = 2 * math.atan(sensor_size / (2 * focal_length))

    # one of two tangent points (the x negative one) of a line with slope fov/2 and
    # a circle of a given radius at the origin
    tangent_slope = math.tan(fov / 2)
    tangent_point_x = -(tangent_slope * radius) / (math.sqrt(1 + math.pow(tangent_slope, 2)))
    tangent_point_y = (radius) / (math.sqrt(1 + math.pow(tangent_slope, 2)))

    # X axis intersection of a line going through the tangent point with slope fov/2
    return (tangent_point_x - tangent_point_y / tangent_slope) * -1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raytrace", type=Path, help="Raytracing distance raster (NPY)")
    parser.add_argument("--output", type=Path, default="overlay.npz", help="Output filename (NPZ)")
    parser.add_argument("--config-contours", type=Path, help="Configuration file (TOML)")
    parser.add_argument("--config-scene", type=Path, help="Configuration file (TOML)")
    parser.add_argument("--visualize", action="store_true", help="Display interactive visualization")

    args = parser.parse_args()

    config_contours = OverlayContoursConfig()
    if args.config_contours is not None:
        with open(args.config_contours, "r") as f:
            data = toml.load(f)
            config_contours = OverlayContoursConfig.model_validate(data)

    config_scene = OverlayContoursConfig()
    if args.config_scene is not None:
        with open(args.config_scene, "r") as f:
            data = toml.load(f)
            config_scene = AdjustSceneConfig.model_validate(data)

    img_pxpos = np.load(args.raytrace)

    mask_nan = np.isnan(np.sum(img_pxpos, axis=2))
    img_non_nan = np.full(mask_nan.shape, 255, dtype=np.uint8)
    img_non_nan[mask_nan] = 0
    contours, hierarchy = cv2.findContours(img_non_nan, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    polygons_silhouette = []
    for contour in contours:
        p = Polygon(contour[:, 0, :].tolist())
        if p.area > 10.0:
            polygons_silhouette.append(p)

    polygons_silhouette = [p.segmentize(MAX_SEGMENT_LENGTH) for p in polygons_silhouette]

    if config_contours.shrink is not None and config_contours.shrink > 0:
        polygons_silhouette = [p.buffer(-config_contours.shrink) for p in polygons_silhouette]

    if config_contours.simplify is not None and config_contours.simplify > 0:
        polygons_silhouette = [p.simplify(config_contours.simplify) for p in polygons_silhouette]

    linestrings = [LineString(p.exterior.coords) for p in polygons_silhouette]

    # 2D TO 3D

    camera_z = _calculate_z_distance_circle(config_scene.camera_focal_length, config_scene.horizontal_width / 2)
    camera_pos = np.array([0.0, 0.0, camera_z])

    linestrings_3d = []
    for ls in linestrings:
        new_coords = np.array([img_pxpos[int(p[1]), int(p[0])] for p in ls.coords])
        new_coords = new_coords[~np.isnan(np.sum(new_coords, axis=1))]

        mean_z = np.mean(new_coords[:, 2])
        vectors = new_coords - camera_pos
        # Calculate scaling factor to project vector onto plane at Z = mean_z
        # For vector from camera at (0,0,camera_z) to point (x,y,z), we need:
        # camera_z + d * (c - camera_z) = mean_z
        # Solving for d: c = (mean_z - camera_z) / (c - camera_z)
        dist = (mean_z - camera_pos[2]) / (new_coords[:, 2] - camera_pos[2])
        dist = dist.reshape(-1, 1)  # Reshape for broadcasting
        new_coords = camera_pos + dist * vectors

        if config_contours.double_line_distance is not None:
            # Convert to spherical coordinates
            x, y, z = new_coords[:, 0], new_coords[:, 1], new_coords[:, 2]
            r = np.sqrt(x**2 + y**2 + z**2)
            theta = np.arctan2(y, x)
            phi = np.arccos(z / r)

            r += config_contours.double_line_distance

            # Convert to Euclidean coordinates
            x_new = r * np.sin(phi) * np.cos(theta)
            y_new = r * np.sin(phi) * np.sin(theta)
            z_new = r * np.cos(phi)
            coords_double_line = np.column_stack((x_new, y_new, z_new))
            linestrings_3d.append(LineString(coords_double_line))

        linestrings_3d.append(LineString(new_coords))

    # VISUALIZE

    if args.visualize:
        visualize_linestrings(linestrings_3d).show()
        exit()

    # if args.debug:
    #     cv2.imwrite(str(DIR_DEBUG / "mask_nan.png"), img_non_nan)
    #
    #     img_silhouette = np.zeros([img_non_nan.shape[0], img_non_nan.shape[1], 3], dtype=np.uint8)
    #     for ls in linestrings_silhouette:
    #         color = random.choices(range(256), k=3)
    #         for pair in linestring_to_coordinate_pairs(ls):
    #             pt1 = [int(c) for c in pair[0]]
    #             pt2 = [int(c) for c in pair[1]]
    #             cv2.line(img_silhouette, pt1, pt2, color, 2)
    #     cv2.imwrite(str(DIR_DEBUG / "silhouette.png"), img_silhouette)

    # EXPORT

    write_linestrings_to_npz(args.output, linestrings_3d)


if __name__ == "__main__":
    main()
