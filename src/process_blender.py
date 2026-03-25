import argparse
import datetime
import math
import os
import shutil
from enum import Enum
from pathlib import Path

import numpy as np
import pyvista as pv
import openexr_numpy
import cv2
import toml
from pydantic import BaseModel, Field
from scipy import ndimage

import matplotlib.pyplot as plt

from util.misc import project_vectors_to_image_space, normalize_vectors, export_angles, rotate_vectors, normalize_vector, rotate_points

MAGNITUDE_THRESHOLD = 0.06

CUTOUT_THRESHOLD = 10

MAPPING_DISTANCE_CUTOFF = True

# EXPORT = False


class LightMode(Enum):
    IMPLICIT = "implicit"
    EXPLICIT = "explicit"
    AXIS = "axis"


class ProcessBlenderConfig(BaseModel):
    # light_angle_xy: float = Field(default=45, description="Azimuthal angle φ (around Z-axis) of the lighting vector in degrees")
    # light_angle_z: float = Field(default=45, description="Polar angle θ (to Z-axis) of the lighting vector in degrees")

    rotX: float = 0
    rotY: float = 0
    rotZ: float = 0

    light_pos_x: float = 0.0
    light_pos_y: float = 0.0
    light_pos_z: float = 2.0

    light_axis_pos_x: float = 0.0
    light_axis_pos_y: float = 0.0
    light_axis_pos_z: float = 2.0

    light_mode: LightMode = LightMode.EXPLICIT

    cross_flow_light: bool = True
    cross_flow_elevation: bool = True
    mixture: list[float] = [0.035, 0.06]

    contrast_increase: float | None = None

    # given the full range of the image, how much percent of this range should be clipped (i.e. [10, 210], 10%, 90% => clipping at [30, 190])
    clip_lower_percent_range: float = Field(default=0, ge=0, lt=100)
    clip_upper_percent_range: float = Field(default=100, gt=0, le=100)

    mode: int = 3

    def model_post_init(self, __context):
        match self.light_mode:
            case LightMode.IMPLICIT:
                self.light_axis_pos_x = self.light_pos_x
                self.light_axis_pos_y = self.light_pos_y
                self.light_axis_pos_z = self.light_pos_z

            case LightMode.EXPLICIT:
                pass

            case LightMode.AXIS:
                p_rot = rotate_points([np.array([0, 0, 1.0])], np.radians(self.rotX), np.radians(self.rotY), np.radians(self.rotZ))
                self.light_axis_pos_x, self.light_axis_pos_y, self.light_axis_pos_z = p_rot[0]

                print(p_rot)

            case _:
                raise NotImplementedError(f"Enum type {self.light_mode} of LightMode not implemented")


def _visualize(centers: np.ndarray, vectors: list[np.ndarray], points: list[np.ndarray], light_axis: np.array) -> pv.Plotter:
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

    # light axis
    if light_axis is not None:
        spline = pv.Spline(np.array([[0, 0, 0], light_axis], dtype=np.float32)).tube(radius=0.005)
        plotter.add_mesh(spline, color=[255, 255, 0])

    colors = ["red", "green", "blue"]

    for vi, vector in enumerate(vectors):
        for i in range(len(vector)):
            if np.isnan(np.sum(centers[i])):
                continue

            if np.isnan(np.sum(vector[i])):
                continue

            if np.sum(np.abs(vector[i])) == 0.0:
                continue

            arrow = pv.Arrow(centers[i], vector[i], scale=0.05)
            plotter.add_mesh(arrow, color=colors[vi])

            # line = pv.Line(centers[i], centers[i] + vector[i])
            # plotter.add_mesh(line)

    for pi, point_set in enumerate(points):
        for point in point_set:
            sphere = pv.Sphere(0.005, point)
            plotter.add_mesh(sphere, color=colors[pi])

    return plotter


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

    return line_point + (-np.dot(plane_normal, line_point - plane_point) / denom) * line_direction


def _compute_intersections(centers: np.ndarray, normals: np.ndarray, axis: np.array) -> np.ndarray:
    intersections = np.zeros_like(centers)
    line_point = np.array([0, 0, 0], dtype=np.float32)

    for i in range(centers.shape[0]):
        intersections[i, :] = _line_plane_intersection(normals[i], centers[i], line_point, axis)

    return intersections


def _apply_clipping(m: np.ndarray, start_percentile: float, end_percentile: float) -> np.ndarray:
    m_no_nan = np.nan_to_num(m)
    minval = np.percentile(m_no_nan, start_percentile)
    maxval = np.percentile(m_no_nan, 100 - end_percentile)
    return np.clip(m_no_nan, minval, maxval)


def _apply_colormap(img: np.ndarray, clip_bottom_percentile: float = 0, clip_top_percentile: float = 0) -> np.ndarray:
    if len(img.shape) > 2 and img.shape[2] > 1:
        raise Exception(f"Can not apply colormap to multi-dimensional image with shape {img.shape}")

    if clip_bottom_percentile > 0 or clip_top_percentile > 0:
        img = _apply_clipping(img, clip_bottom_percentile, clip_top_percentile)

    img_no_nan = np.nan_to_num(img)
    img_norm = cv2.normalize(img_no_nan.astype("float32"), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    colormap = plt.colormaps.get_cmap("viridis")
    img_colored = colormap(img_norm)  # shape: (H, W, 4) with RGBA
    img_rgb = (img_colored[:, :, :3] * 255).astype(np.uint8)  # Remove alpha channel
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr


def _apply_linear_slope(m: np.ndarray, slope_start: float, slope_end: float, clipping_start: float = 0, clipping_end: float = 0) -> np.ndarray:
    """
    Applies a transformation to the ndarray m. M is normalized to [0, 1.0], all values below `slope_start` are
    set to 0, all values above `slope_end` to 1.0.
    In between the start and end points the values are linearly interpolated.
    """

    if clipping_start > 0 or clipping_end > 0:
        m = _apply_clipping(m, clipping_start, clipping_end)

    m_no_nan = np.nan_to_num(m)
    m_norm = cv2.normalize(m_no_nan, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    m_norm[m_norm < slope_start] = 0
    mask = (m_norm >= slope_start) & (m_norm <= slope_end)
    m_norm[mask] = (m_norm[mask] - slope_start) * 1 / (slope_end - slope_start)
    m_norm[m_norm > slope_end] = 1.0

    return m_norm


def _apply_linear_transition(m: np.ndarray, min: float, max: float) -> np.ndarray:
    return (np.clip(m, min, max) - min) / (max - min)


def _segmentize(image: np.ndarray) -> np.ndarray:
    algo = cv2.ximgproc.SLIC  # Standard SLIC
    # algo = cv2.ximgproc.SLICO      # Zero-parameter SLIC
    # algo = cv2.ximgproc.MSLIC      # Multi-scale SLIC

    slic = cv2.ximgproc.createSuperpixelSLIC(
        image,
        algorithm=algo,
        region_size=10,
        ruler=10,
    )

    slic.iterate(10)
    slic.enforceLabelConnectivity(min_element_size=25)

    mask = slic.getLabelContourMask(thick_line=False)
    labels = slic.getLabels()

    return labels, mask


def _float_to_uint8(a: np.ndarray) -> np.ndarray:
    min_a = np.min(a[~np.isnan(a)])
    ptp_a = np.ptp(a[~np.isnan(a)])
    return (np.iinfo(np.uint8).max * ((a - min_a) / ptp_a)).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("normals", type=Path, default="normals.exr", help="Normals (EXR)")
    parser.add_argument("image", type=Path, default="image.tif", help="RGB image (TIFF)")
    parser.add_argument("raytrace", type=Path, default="raytrace.npy", help="Raytracing distance raster (NPY)")
    parser.add_argument("--projection-matrix", type=Path, default=None, help="3x4 projection matrix (NPY)")
    parser.add_argument("--scaling-factor", type=float, default=None, help="Scaling factor (float)")
    parser.add_argument("--output", type=Path, default="temp", help="Output directory")
    parser.add_argument("--config", type=Path, help="Configuration file (TOML)")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug output")
    parser.add_argument("--visualize", action="store_true", default=False, help="Enable interactive visualization")

    args = parser.parse_args()

    config = ProcessBlenderConfig()
    if args.config is not None:
        with open(args.config, "r") as f:
            data = toml.load(f)
            config = ProcessBlenderConfig.model_validate(data)

    dir_debug = args.output.parent / (str(args.output.stem) + "_debug")

    if args.debug:
        os.makedirs(dir_debug, exist_ok=True)

    timer_start = datetime.datetime.now()

    img_normals = openexr_numpy.imread(str(args.normals), "XYZ")
    img_color = cv2.imread(str(args.image))
    img_gray = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    img_pxpos = np.load(args.raytrace)

    # azimuthal_angle = config.light_angle_xy
    # polar_angle = config.light_angle_z
    # light_pos = [
    #     math.sin(math.radians(polar_angle)) * math.cos(math.radians(azimuthal_angle)),
    #     math.sin(math.radians(polar_angle)) * math.sin(math.radians(azimuthal_angle)),
    #     math.cos(math.radians(polar_angle)),
    # ]
    # light_axis = normalize_vector(np.array(light_pos))

    # align light_axis with the polar axis of the planet
    light_axis_rot = [math.radians(r) for r in [config.rotX, config.rotY, config.rotZ]]
    light_axis = np.array([0, 0, 1])
    light_axis = rotate_vectors(light_axis, np.array(light_axis_rot))

    # if light axis pos data is available, use this instead
    light_axis_pos = np.array([config.light_axis_pos_x, config.light_axis_pos_y, config.light_axis_pos_z], dtype=float)
    if not np.isnan(np.sum(light_axis_pos)):
        light_axis = normalize_vector(light_axis_pos)

    if args.scaling_factor is not None:
        resize_size = (int(img_normals.shape[1] * args.scaling_factor), int(img_normals.shape[0] * args.scaling_factor))
        img_normals = cv2.resize(img_normals, resize_size)
        img_color = cv2.resize(img_color, resize_size)
        img_gray = cv2.resize(img_gray, resize_size)
        img_pxpos = cv2.resize(img_pxpos, resize_size)

    if args.visualize:
        resize_size = [20, 20]
        img_normals = cv2.resize(img_normals, resize_size)
        img_color = cv2.resize(img_color, resize_size)
        img_gray = cv2.resize(img_gray, resize_size)
        img_pxpos = cv2.resize(img_pxpos, resize_size)

    # Mapping Color

    mapping_color = img_color

    # Mapping Angle

    intersections = np.zeros_like(img_normals)
    for x in range(img_normals.shape[1]):
        for y in range(img_normals.shape[0]):
            intersections[y, x, :] = _line_plane_intersection(img_normals[y, x], img_pxpos[y, x], np.array([0.0, 0.0, 0.0]), light_axis)

    img_direction = intersections - img_pxpos

    # flip direction if intersection point is on the opposite end of the axis
    # necessary to avoid a flipping of direction signs when moving from one face to the next one
    opposite_directions = np.full(img_direction.shape[0:2], 1, dtype=np.float32)
    opposite_directions[np.dot(img_direction, light_axis) < 0] = -1
    img_direction *= opposite_directions[:, :, np.newaxis]

    img_elevation_vector = normalize_vectors(img_pxpos)
    dot = np.sum(img_elevation_vector * img_normals, axis=2, keepdims=True)  # vectorized dot product
    img_elevation_direction = img_elevation_vector - dot * img_normals
    img_elevation_magnitude = np.arccos(dot)
    img_elevation_magnitude = img_elevation_magnitude[:, :, 0]  # [m, n, 1] to [m, n]

    # create mixture for elevation and magnitude
    mixture_elevation_magnitude = _apply_linear_slope(img_elevation_magnitude, 0.1, 0.15, clipping_end=1.0)

    # calculation of angles in world space and image space

    projection_matrix = np.load(args.projection_matrix)

    img_direction_ws = img_direction
    img_elevation_direction_ws = img_elevation_direction

    # cross product must be applied on the vectors in world space before projecting to image space
    if config.cross_flow_light:
        img_direction_ws = np.cross(img_direction_ws, img_normals)
    if config.cross_flow_elevation:
        img_elevation_direction_ws = np.cross(img_elevation_direction_ws, img_normals)

    img_direction_is = project_vectors_to_image_space(img_pxpos, img_direction_ws, projection_matrix)
    img_elevation_direction_is = project_vectors_to_image_space(img_pxpos, img_elevation_direction_ws, projection_matrix)

    # ---

    img_field_elevation_vectors_0 = np.zeros_like(img_direction)
    img_field_elevation_vectors_1 = np.zeros_like(img_direction)
    img_field_elevation_vectors_2 = np.zeros_like(img_direction)
    img_field_elevation_vectors_3 = np.zeros_like(img_direction)
    img_field_elevation_vectors_4 = np.zeros_like(img_direction)
    img_field_elevation_vectors_5 = np.zeros_like(img_direction)
    img_field_elevation_vectors_6 = np.zeros_like(img_direction)
    img_field_elevation_vectors_7 = np.zeros_like(img_direction)
    img_field_elevation_vectors_8 = np.zeros_like(img_direction_is)
    img_field_elevation_vectors_9 = np.zeros_like(img_direction_is)
    img_field_elevation_vectors_10 = np.zeros_like(img_direction_is)

    ELEVATION_VECTOR_WEIGHT = 0.5

    distance_point_to_light_axis = np.linalg.norm(light_axis - img_normals, axis=-1)
    distance_weight = (distance_point_to_light_axis - np.min(distance_point_to_light_axis)) / np.ptp(distance_point_to_light_axis)

    # 100% elevation
    img_field_elevation_vectors_0 = img_elevation_direction

    # 100% light
    img_field_elevation_vectors_1 = img_direction

    # fixed weights: the final vector is X% light and 1-X% elevation
    img_field_elevation_vectors_2 = img_direction * (1 - ELEVATION_VECTOR_WEIGHT) + img_elevation_direction * ELEVATION_VECTOR_WEIGHT

    # dynamic mixture based on magnitude
    mixture_magnitude = cv2.normalize(_apply_clipping(img_elevation_magnitude, 0, 1), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    mixture_magnitude = mixture_magnitude[:, :, np.newaxis]
    img_field_elevation_vectors_3 = img_direction * (1 - mixture_magnitude) + img_elevation_direction * mixture_magnitude

    # dynamic mixture based on magnitude with thresholds and a linear slope
    mixture_elevation_magnitude_newaxis = mixture_elevation_magnitude[:, :, np.newaxis]
    img_field_elevation_vectors_4 = (
        img_direction * (1 - mixture_elevation_magnitude_newaxis) + img_elevation_direction * mixture_elevation_magnitude_newaxis
    )

    # hard cut: below MAGNITUDE_THRESHOLD follow the light vector, above the elevation vector
    mask = img_elevation_magnitude > MAGNITUDE_THRESHOLD
    img_field_elevation_vectors_5 = img_direction.copy()
    img_field_elevation_vectors_5[mask] = img_elevation_direction[mask]

    img_elevation_direction_crossed = np.cross(img_elevation_direction, img_normals)
    img_direction_crossed = np.cross(img_direction, img_normals)

    # like _5 but with partial cross
    mask = mixture_elevation_magnitude > MAGNITUDE_THRESHOLD
    # img_field_elevation_vectors_6 = img_direction.copy()
    # img_field_elevation_vectors_6[mask] = img_elevation_direction_crossed[mask]

    img_field_elevation_vectors_6 = img_direction_crossed.copy()
    img_field_elevation_vectors_6[mask] = img_elevation_direction[mask]

    # like _4 but with partial cross
    # img_field_elevation_vectors_7 = (
    #     img_direction * (1 - mixture_elevation_magnitude_newaxis)
    #     + img_elevation_direction_crossed * mixture_elevation_magnitude_newaxis
    # )
    img_field_elevation_vectors_7 = (
        img_direction_crossed * (1 - mixture_elevation_magnitude_newaxis) + img_elevation_direction * mixture_elevation_magnitude_newaxis
    )

    # image space - 100% elevation

    img_field_elevation_vectors_8 = img_direction_is

    # image space - hard cut: below MAGNITUDE_THRESHOLD follow the light vector, above the elevation vector

    mask = img_elevation_magnitude > MAGNITUDE_THRESHOLD
    img_field_elevation_vectors_9 = img_direction_is.copy()
    img_field_elevation_vectors_9[mask] = img_elevation_direction_is[mask]

    # image space - blend:

    mixture = _apply_linear_transition(img_elevation_magnitude, config.mixture[0], config.mixture[1])
    img_field_elevation_vectors_10 = img_direction_is * (1 - mixture)[:, :, np.newaxis] + img_elevation_direction_is * mixture[:, :, np.newaxis]

    if args.debug:
        cv2.imwrite(str(dir_debug / "img_direction_is.png"), _apply_colormap(export_angles(img_direction_is)))
        cv2.imwrite(str(dir_debug / "img_elevation_direction_is.png"), _apply_colormap(export_angles(img_elevation_direction_is)))
        cv2.imwrite(str(dir_debug / "mixture.png"), _apply_colormap(mixture))

    # ---

    # Mapping Distance

    mapping_distance = img_gray
    # mapping_distance = ((mapping_distance - np.min(mapping_distance)) / np.ptp(mapping_distance) * 255).astype(np.uint8)

    # Mapping Line Length

    img_distance = np.linalg.norm(img_pxpos, axis=-1)  # distance from pixel location to origin

    # 0) angle | Flatness

    # dot product
    img_pxpos_normalized = img_pxpos / np.linalg.norm(img_pxpos, axis=-1, keepdims=True)
    img_normals_normalized = img_normals / np.linalg.norm(img_normals, axis=-1, keepdims=True)
    cos_angle = np.sum(img_pxpos_normalized * img_normals_normalized, axis=-1)
    cos_angle = np.clip(cos_angle, -1, 1)  # handle numerical errors

    img_angle = np.arccos(cos_angle)
    mapping_line_length = _float_to_uint8(img_angle)

    cv2.imwrite(str(args.output / "mapping_line_length_0_0.png"), mapping_line_length)
    cv2.imwrite(str(args.output / "mapping_line_length_0_1.png"), ~mapping_line_length)

    # 1) win var inverted | Roughness / rate of change of terrain

    WINDOW_SIZE = 10
    MAX_WIN_VAR = 1e-6
    img_dist = np.nan_to_num(img_distance)
    win_mean = ndimage.uniform_filter(img_dist, (WINDOW_SIZE, WINDOW_SIZE))
    win_sqr_mean = ndimage.uniform_filter(img_dist**2, (WINDOW_SIZE, WINDOW_SIZE))
    win_var = win_sqr_mean - win_mean**2

    win_var = np.clip(win_var, 0, MAX_WIN_VAR)
    win_var = win_var * -1 + MAX_WIN_VAR

    mapping_line_length = _float_to_uint8(win_var)

    cv2.imwrite(str(args.output / "mapping_line_length_1_0.png"), mapping_line_length)
    cv2.imwrite(str(args.output / "mapping_line_length_1_1.png"), ~mapping_line_length)

    # 2) line_distance | Brightness

    bg_mask = ~np.isnan(np.sum(img_pxpos, axis=2))

    minval = np.percentile(mapping_distance[bg_mask], 10)
    maxval = np.percentile(mapping_distance[bg_mask], 100 - 3)
    mapping_distance2 = np.clip(mapping_distance, minval, maxval)
    mapping_distance2 = _float_to_uint8(mapping_distance2)

    cv2.imwrite(str(args.output / "mapping_line_length_2_0.png"), mapping_distance2)
    cv2.imwrite(str(args.output / "mapping_line_length_2_1.png"), ~mapping_distance2)

    # 3) distance from origin | Altitude

    mapping_line_length = _float_to_uint8(img_distance)

    cv2.imwrite(str(args.output / "mapping_line_length_3_0.png"), mapping_line_length)
    cv2.imwrite(str(args.output / "mapping_line_length_3_1.png"), ~mapping_line_length)

    match config.mode:
        case 0:
            shutil.copyfile(args.output / "mapping_line_length_0_0.png", args.output / "mapping_line_length.png")
        case 1:
            shutil.copyfile(args.output / "mapping_line_length_0_1.png", args.output / "mapping_line_length.png")
        case 2:
            shutil.copyfile(args.output / "mapping_line_length_1_0.png", args.output / "mapping_line_length.png")
        case 3:
            shutil.copyfile(args.output / "mapping_line_length_1_1.png", args.output / "mapping_line_length.png")
        case 4:
            shutil.copyfile(args.output / "mapping_line_length_2_0.png", args.output / "mapping_line_length.png")
        case 5:
            shutil.copyfile(args.output / "mapping_line_length_2_1.png", args.output / "mapping_line_length.png")
        case 6:
            shutil.copyfile(args.output / "mapping_line_length_3_0.png", args.output / "mapping_line_length.png")
        case 7:
            shutil.copyfile(args.output / "mapping_line_length_3_1.png", args.output / "mapping_line_length.png")

    # Mapping Background

    mapping_background = np.zeros_like(img_pxpos, dtype=np.uint8)
    mapping_background[np.isnan(np.sum(img_pxpos, axis=2))] = [255, 255, 255]

    if config.cross_flow_elevation:
        img_field_elevation_vectors_0 = np.cross(img_field_elevation_vectors_0, img_normals)
        img_field_elevation_vectors_1 = np.cross(img_field_elevation_vectors_1, img_normals)
        img_field_elevation_vectors_2 = np.cross(img_field_elevation_vectors_2, img_normals)
        img_field_elevation_vectors_3 = np.cross(img_field_elevation_vectors_3, img_normals)
        img_field_elevation_vectors_4 = np.cross(img_field_elevation_vectors_4, img_normals)
        img_field_elevation_vectors_5 = np.cross(img_field_elevation_vectors_5, img_normals)
        # img_field_elevation_vectors_6 = np.cross(img_field_elevation_vectors_6, img_normals)
        # img_field_elevation_vectors_7 = np.cross(img_field_elevation_vectors_7, img_normals)

    if args.visualize:
        # in world space

        centers = img_pxpos.reshape([-1, 3])
        normals = img_normals.reshape([-1, 3])
        direction = img_direction.reshape([-1, 3])  # relative to (0, 0)
        elevation_direction = normalize_vectors(img_elevation_direction).reshape([-1, 3])
        elevation_direction = img_field_elevation_vectors_1.reshape([-1, 3])

        # _visualize(centers, [normals, direction, elevation_direction], [], light_axis).show()
        _visualize(centers, [normals, direction], [], None).show()

        # visualize(centers, [direction], [], light_axis).show()
        # visualize(centers, [normals, elevation_direction], [], light_axis).show()

        # field_elevation_vectors = img_field_elevation_vectors_4.reshape([-1, 3])
        # visualize(centers, [direction, field_elevation_vectors], light_axis).show()
        # visualize(centers, [direction], light_axis).show()

        # in image space

        # P = np.load(args.projection_matrix)
        # img_centers = img_pxpos.reshape([-1, 3])
        # img_centers = np.hstack([img_centers, np.full([img_centers.shape[0], 1], 1)])
        # img_centers = (P @ img_centers.T).T
        # img_centers = img_centers[:, 0:2] / img_centers[:, 2][:, np.newaxis] # divide by w
        #
        # img_centers /= 3000
        #
        # img_centers = img_centers[~np.isnan(np.sum(img_centers, axis=1))] # remove NaNs
        # img_centers = np.hstack((img_centers, np.full([img_centers.shape[0], 1], 0)))

        # # visualize(img_centers, [direction], light_axis).show()
        # visualize(img_centers, [np.full([img_centers.shape[0], 3], [0, 0, 1])], light_axis).show()

        # img_direction_reshaped = _normalize_vectors(img_direction).reshape([-1, 3]) * IMAGE_SPACE_DIRECTION_STEP_DISTANCE
        #
        # p_image_space = _project_to_image_space(centers, P)
        # p2_image_space = _project_to_image_space(centers + img_direction_reshaped, P)
        #
        # p_image_space /= 3000
        # p2_image_space /= 3000
        #
        # angles = p2_image_space - p_image_space
        # angles = np.atan2(angles[:, 1], angles[:, 0])
        #
        # cos_angles = np.cos(angles)
        # sin_angles = np.sin(angles)
        # R = np.stack([np.stack([cos_angles, -sin_angles], axis=-1), np.stack([sin_angles, cos_angles], axis=-1)], axis=-2)
        # rotated = np.einsum("nij,nj->ni", R, np.full([p_image_space.shape[0], 2], [IMAGE_SPACE_DIRECTION_STEP_DISTANCE, 0]))
        # p3_image_space = p_image_space + rotated
        #
        # p_image_space = np.hstack([p_image_space, np.full([p_image_space.shape[0], 1], 0)])
        # p2_image_space = np.hstack([p2_image_space, np.full([p2_image_space.shape[0], 1], 0)])
        # p3_image_space = np.hstack([p3_image_space, np.full([p3_image_space.shape[0], 1], 0)])
        #
        # # visualize(centers, [], [centers, centers + direction2], light_axis).show()
        # # visualize(centers, [], [centers, centers + direction, centers + direction2], light_axis).show()
        # visualize(centers, [], [p_image_space, p2_image_space, p3_image_space], light_axis).show()

        # foo = img_elevation_direction_is.reshape([-1, 3])
        # foo /= 3000
        # visualize(centers, [foo], [], light_axis).show()

    cv2.imwrite(str(args.output / "mapping_color.png"), mapping_color)

    if config.contrast_increase is not None and config.contrast_increase > 0:
        clahe = cv2.createCLAHE(clipLimit=config.contrast_increase, tileGridSize=(8, 8))
        mapping_distance = clahe.apply(mapping_distance)

    if args.debug:
        cv2.imwrite(str(dir_debug / "mapping_distance_noclip.png"), _apply_colormap(mapping_distance))

    bg_mask = ~np.isnan(np.sum(img_pxpos, axis=2))

    min_dist = np.min(mapping_distance[bg_mask])
    range_dist = np.ptp(mapping_distance[bg_mask])

    minval = min_dist + range_dist * (config.clip_lower_percent_range / 100.0) if config.clip_lower_percent_range > 0 else 0
    maxval = min_dist + range_dist * (config.clip_upper_percent_range / 100.0) if config.clip_upper_percent_range < 100 else 255

    mapping_distance = np.clip(mapping_distance, minval, maxval)
    mapping_distance = ((mapping_distance - minval) / (maxval - minval) * 255).astype(np.uint8)

    if args.debug:
        cv2.imwrite(str(dir_debug / "mapping_distance_clip.png"), _apply_colormap(mapping_distance))

    if MAPPING_DISTANCE_CUTOFF:
        # add the min regions of mapping_distance to background
        mapping_background[mapping_distance == 0] = 255

    cv2.imwrite(str(args.output / "mapping_distance.png"), mapping_distance)
    cv2.imwrite(str(args.output / "mapping_angle.png"), export_angles(img_field_elevation_vectors_10, adjust_y_axis=True))
    # cv2.imwrite(str(args.output / "mapping_line_length.png"), mapping_line_length)
    cv2.imwrite(str(args.output / "mapping_background.png"), mapping_background)

    print(f"total time: {(datetime.datetime.now() - timer_start).total_seconds():5.2f}s")

    # Segmentation Experiment

    # labels, _ = _segmentize(img_normals)
    # segmented_angles = img_field_elevation_vectors_10.copy()
    # for i in range(np.max(labels)):
    #     mask = labels == i
    #     avg = np.mean(segmented_angles[mask], axis=0)
    #     segmented_angles[mask] = avg
    #
    # cv2.imwrite(str(args.output / "mapping_angle.png"), export_angles(segmented_angles, adjust_y_axis=True))


if __name__ == "__main__":
    main()
