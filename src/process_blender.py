import argparse
import datetime
from pathlib import Path

import numpy as np
import pyvista as pv
import openexr_numpy
import cv2
from scipy import ndimage

import matplotlib.pyplot as plt

DIR_DEBUG = Path("debug")

CROSS_FLOW = True

VISUALIZE = False
EXPORT = True

# coordinate system: Y-UP
LIGHT_POS = [1.5, 0.3, 2]
LIGHT_POS = [0.22, 0.22, 1]
LIGHT_POS = [1, 1, 1]
LIGHT_POS = [1, 1, 0]
LIGHT_POS = [1, 0, 1]

CONTRAST_ENHANCEMENT = True
CONTRAST_VALUE = 1.60

CLIPPING = True
CLIPPING_CUTOFF_PERCENTILE = 1.00

MAGNITUDE_THRESHOLD = 0.1

CUTOUT_THRESHOLD = 10

# VISUALIZE = True
# EXPORT = False


def _normalize_vector(v: np.array) -> np.array:
    return v / np.linalg.norm(v)


def _normalize_vectors(v: np.ndarray) -> np.array:
    return v / np.linalg.norm(v, axis=2)[:, :, np.newaxis]


def visualize(centers: np.ndarray, vectors: list[np.ndarray], light_axis: np.array) -> pv.Plotter:
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
    spline = pv.Spline(np.array([[0, 0, 0], light_axis], dtype=np.float32)).tube(radius=0.005)
    plotter.add_mesh(spline, color=[255, 255, 0])

    colors = ["red", "green", "blue"]

    for vi, vector in enumerate(vectors):
        for i in range(len(vector)):
            if np.isnan(np.sum(vector[i])):
                continue
            if np.sum(np.abs(vector[i])) == 0.0:
                continue
            arrow = pv.Arrow(centers[i], vector[i], scale=0.05)
            plotter.add_mesh(arrow, color=colors[vi])

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


def apply_clipping(m: np.ndarray, start_percentile: float, end_percentile: float) -> np.ndarray:
    m_no_nan = np.nan_to_num(m)
    minval = np.percentile(m_no_nan, start_percentile)
    maxval = np.percentile(m_no_nan, 100 - end_percentile)
    return np.clip(m, minval, maxval)


def apply_colormap(img: np.ndarray, clip_bottom_percentile: float = 0, clip_top_percentile: float = 0) -> np.ndarray:
    if len(img.shape) > 2 and img.shape[2] > 1:
        raise Exception("Can not apply colormap to multi-dimensional image")

    if clip_bottom_percentile > 0 or clip_top_percentile > 0:
        img = apply_clipping(img, clip_bottom_percentile, clip_top_percentile)

    img_norm = cv2.normalize(img.astype("float32"), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    colormap = plt.colormaps.get_cmap("viridis")
    img_colored = colormap(img_norm)  # shape: (H, W, 4) with RGBA
    img_rgb = (img_colored[:, :, :3] * 255).astype(np.uint8)  # Remove alpha channel
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr


def apply_linear_slope(
    m: np.ndarray, slope_start: float, slope_end: float, clipping_start: float = 0, clipping_end: float = 0
) -> np.ndarray:
    """
    Applies a transformation to the ndarray m. M is normalized to [0, 1.0], all values below `slope_start` are
    set to 0, all values above `slope_end` to 1.0.
    In between the start and end points the values are linearly interpolated.
    """

    if clipping_start > 0 or clipping_end > 0:
        m = apply_clipping(m, clipping_start, clipping_end)

    m_norm = cv2.normalize(m, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    m_norm[m_norm < slope_start] = 0
    mask = (m_norm >= slope_start) & (m_norm <= slope_end)
    m_norm[mask] = (m_norm[mask] - slope_start) * 1 / (slope_end - slope_start)
    m_norm[m_norm > slope_end] = 1.0

    return m_norm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("normals", type=Path, default="normals.exr", help="Normals (EXR)")
    parser.add_argument("image", type=Path, default="image.tif", help="RGB image (TIFF)")
    parser.add_argument("raytrace", type=Path, default="raytrace.npy", help="Raytracing distance raster (NPY)")
    parser.add_argument("--scaling-factor", type=float, default=None, help="Scaling factor (float)")
    parser.add_argument("--output", type=Path, default="temp", help="Output directory")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug output")
    args = parser.parse_args()

    timer_start = datetime.datetime.now()

    img_normals = openexr_numpy.imread(str(args.normals), "XYZ")
    img_gray = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    img_pxpos = np.load(args.raytrace)

    light_axis = _normalize_vector(np.array(LIGHT_POS))

    if args.scaling_factor is not None:
        resize_size = (int(img_normals.shape[1] * args.scaling_factor), int(img_normals.shape[0] * args.scaling_factor))
        img_normals = cv2.resize(img_normals, resize_size)
        img_gray = cv2.resize(img_gray, resize_size)
        img_pxpos = cv2.resize(img_pxpos, resize_size)

    if VISUALIZE:
        resize_size = [50, 50]
        img_normals = cv2.resize(img_normals, resize_size)
        img_gray = cv2.resize(img_gray, resize_size)
        img_pxpos = cv2.resize(img_pxpos, resize_size)

    # EXPERIMENTS

    # centers = []
    # normals = []
    #
    # for x in range(img_normals.shape[1]):
    #     for y in range(img_normals.shape[0]):
    #         if abs(np.linalg.norm(img_normals[y, x, :])) < 0.01:
    #             continue
    #
    #         # centers.append(np.array([
    #         #     (x / img_normals.shape[1] - 0.5) * 2,
    #         #     (y / img_normals.shape[0] - 0.5) * 2 * -1, # Y axis coordinate flip: numpy origin top-left, blender bottom-left
    #         #     img_pxpos[y, x, 2]
    #         # ]))
    #
    #         centers.append(np.nan_to_num(img_pxpos[y, x, :]))
    #         normals.append(img_normals[y, x, :])
    #
    # centers = np.array(centers)
    # normals = np.array(normals)
    #
    # intersections = _compute_intersections(centers, normals, light_axis)
    # directions = _normalize_vectors(intersections - centers)
    #
    # # flip direction if intersection point is on the opposite end of the axis
    # # necessary to avoid a flipping of direction signs when moving from one face to the next one
    # opposite_directions = np.full_like(directions, 1, dtype=np.float32)
    # opposite_directions[np.dot(intersections, light_axis) < 0] = -1
    # directions *= opposite_directions
    #
    # ELEVATION_VECTOR_WEIGHT = 0.6
    # elevation_vectors = _normalize_vectors(centers)
    # field_elevation_vectors = []
    # for i in range(len(elevation_vectors)):
    #     projected = elevation_vectors[i] - (np.dot(elevation_vectors[i], normals[i])) * normals[i]
    #     magnitude = np.arccos(np.dot(elevation_vectors[i], normals[i]))
    #     combined = _normalize_vector(
    #         directions[i] * (1 - magnitude) * (1 - ELEVATION_VECTOR_WEIGHT)
    #         + projected * magnitude * ELEVATION_VECTOR_WEIGHT
    #     )
    #     field_elevation_vectors.append(combined)
    #
    # visualize(centers, [normals, directions], light_axis).show()
    # visualize(centers, [directions, field_elevation_vectors], light_axis).show()
    # visualize(centers, [directions], light_axis).show()

    # Mapping Angle

    intersections = np.zeros_like(img_normals)
    for x in range(img_normals.shape[1]):
        for y in range(img_normals.shape[0]):
            intersections[y, x, :] = _line_plane_intersection(
                img_normals[y, x], img_pxpos[y, x], np.array([0.0, 0.0, 0.0]), light_axis
            )

    img_direction = intersections - img_pxpos

    # flip direction if intersection point is on the opposite end of the axis
    # necessary to avoid a flipping of direction signs when moving from one face to the next one
    opposite_directions = np.full_like(img_direction, 1, dtype=np.float32)
    opposite_directions[np.dot(img_direction, light_axis) < 0] = -1
    img_direction *= opposite_directions

    img_elevation_vector = _normalize_vectors(img_pxpos)
    dot = np.sum(img_elevation_vector * img_normals, axis=2, keepdims=True)  # vectorized dot product
    img_elevation_direction = img_elevation_vector - (dot) * img_normals
    img_elevation_magnitude = np.arccos(dot)

    # create mixture for elevation and magnitude
    mixture_elevation_magnitude = apply_linear_slope(img_elevation_magnitude, 0.1, 0.15, clipping_end=1.0)

    if args.debug:
        cv2.imwrite(
            str(DIR_DEBUG / "img_elevation_magnitude.png"),
            apply_colormap(
                img_elevation_magnitude,
                # clip_bottom_percentile=0,
                # clip_top_percentile=0.5
            ),
        )

        cv2.imwrite(str(DIR_DEBUG / "mixture_elevation_magnitude.png"), apply_colormap(mixture_elevation_magnitude))

    img_field_elevation_vectors_0 = np.zeros_like(img_direction)
    img_field_elevation_vectors_1 = np.zeros_like(img_direction)
    img_field_elevation_vectors_2 = np.zeros_like(img_direction)
    img_field_elevation_vectors_3 = np.zeros_like(img_direction)
    img_field_elevation_vectors_4 = np.zeros_like(img_direction)
    img_field_elevation_vectors_5 = np.zeros_like(img_direction)
    img_field_elevation_vectors_6 = np.zeros_like(img_direction)
    img_field_elevation_vectors_7 = np.zeros_like(img_direction)

    ELEVATION_VECTOR_WEIGHT = 0.5

    distance_point_to_light_axis = np.linalg.norm(light_axis - img_normals, axis=-1)
    distance_weight = (distance_point_to_light_axis - np.min(distance_point_to_light_axis)) / np.ptp(
        distance_point_to_light_axis
    )

    # 100% elevation
    img_field_elevation_vectors_0 = img_elevation_direction

    # 100% light
    img_field_elevation_vectors_1 = img_direction

    # fixed weights: the final vector is X% light and 1-X% elevation
    img_field_elevation_vectors_2 = (
        img_direction * (1 - ELEVATION_VECTOR_WEIGHT) + img_elevation_direction * ELEVATION_VECTOR_WEIGHT
    )

    # dynamic mixture based on magnitude
    mixture_magnitude = cv2.normalize(
        apply_clipping(img_elevation_magnitude, 0, 1), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX
    )
    mixture_magnitude = mixture_magnitude[:, :, np.newaxis]
    img_field_elevation_vectors_3 = (
        img_direction * (1 - mixture_magnitude) + img_elevation_direction * mixture_magnitude
    )

    # dynamic mixture based on magnitude with thresholds and a linear slope
    mixture_elevation_magnitude_newaxis = mixture_elevation_magnitude[:, :, np.newaxis]
    img_field_elevation_vectors_4 = (
        img_direction * (1 - mixture_elevation_magnitude_newaxis)
        + img_elevation_direction * mixture_elevation_magnitude_newaxis
    )

    # hard cut: below MAGNITUDE_THRESHOLD follow the light vector, above the elevation vector
    mask = mixture_elevation_magnitude > MAGNITUDE_THRESHOLD
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
        img_direction_crossed * (1 - mixture_elevation_magnitude_newaxis)
        + img_elevation_direction * mixture_elevation_magnitude_newaxis
    )

    # Mapping Distance

    mapping_distance = img_gray
    # mapping_distance = ((mapping_distance - np.min(mapping_distance)) / np.ptp(mapping_distance) * 255).astype(np.uint8)

    # Mapping Line Length

    # distance from pixel location to origin
    img_distance = np.linalg.norm(img_pxpos, axis=-1)
    img_distance = np.nan_to_num(img_distance)

    # print(np.min(img_distance), np.max(img_distance))
    # cv2.imwrite(str("img_distance.png"), (img_distance * 255 / np.max(img_distance)).astype(np.uint8))

    WINDOW_SIZE = 20
    MAX_WIN_VAR = 1e-6
    win_mean = ndimage.uniform_filter(img_distance, (WINDOW_SIZE, WINDOW_SIZE))
    win_sqr_mean = ndimage.uniform_filter(img_distance**2, (WINDOW_SIZE, WINDOW_SIZE))
    win_var = win_sqr_mean - win_mean**2

    win_var = np.clip(win_var, 0, MAX_WIN_VAR)
    win_var = win_var * -1 + MAX_WIN_VAR

    mapping_line_length = (np.iinfo(np.uint8).max * ((win_var - np.min(win_var)) / np.ptp(win_var))).astype(np.uint8)

    # Mapping Flat

    mapping_flat = np.zeros_like(img_pxpos, dtype=np.uint8)
    mapping_flat[np.isnan(img_pxpos)] = 255

    mapping_flat[mapping_distance < CUTOUT_THRESHOLD] = 255

    if CROSS_FLOW:
        # img_directions = np.cross(img_directions, img_normals)
        img_field_elevation_vectors_0 = np.cross(img_field_elevation_vectors_0, img_normals)
        img_field_elevation_vectors_1 = np.cross(img_field_elevation_vectors_1, img_normals)
        img_field_elevation_vectors_2 = np.cross(img_field_elevation_vectors_2, img_normals)
        img_field_elevation_vectors_3 = np.cross(img_field_elevation_vectors_3, img_normals)
        img_field_elevation_vectors_4 = np.cross(img_field_elevation_vectors_4, img_normals)
        img_field_elevation_vectors_5 = np.cross(img_field_elevation_vectors_5, img_normals)

    if VISUALIZE:
        centers = img_pxpos.reshape([-1, 3])
        # normals = img_normals.reshape([-1, 3])
        direction = img_direction.reshape([-1, 3])
        # visualize(centers, [normals, directions], light_axis).show()
        # visualize(centers, [directions], light_axis).show()

        field_elevation_vectors = img_field_elevation_vectors_4.reshape([-1, 3])
        visualize(centers, [direction, field_elevation_vectors], light_axis).show()

    def export_angles(arr: np.ndarray) -> np.ndarray:
        arr[:, :, 1] *= -1  # blender Y up / numpy Y down
        mapping_angle = np.atan2(arr[:, :, 1], arr[:, :, 0])
        mapping_angle = (mapping_angle + np.pi) / (np.pi * 2.0)
        mapping_angle = (mapping_angle * 255).astype(np.uint8)
        return mapping_angle

    def adjust_contrast_brightness(img, contrast: float = 1.0, brightness: int = 0) -> np.ndarray:
        """
        Adjusts contrast and brightness of an uint8 image.
        contrast:   (0.0,  inf) with 1.0 leaving the contrast as is
        brightness: [-255, 255] with 0 leaving the brightness as is
        """
        brightness += int(round(255 * (1 - contrast) / 2))
        return cv2.addWeighted(img, contrast, img, 0, brightness)

    if EXPORT:
        if CONTRAST_ENHANCEMENT:
            # evaluating a good CONTRAST_VALUE:
            # for i in range(1, 10):
            #     contrast = 1.0+i/10
            #     cv2.imwrite(str(args.output / f"mapping_distance_{contrast:5.2f}.png"), adjust_contrast_brightness(mapping_distance, contrast=contrast))

            mapping_distance = adjust_contrast_brightness(mapping_distance, contrast=CONTRAST_VALUE)

        if CLIPPING:
            minval = np.percentile(mapping_distance, CLIPPING_CUTOFF_PERCENTILE)
            maxval = np.percentile(mapping_distance, 100 - CLIPPING_CUTOFF_PERCENTILE)
            mapping_distance = np.clip(mapping_distance, minval, maxval)
            mapping_distance = (((mapping_distance - minval) / (maxval - minval)) * 255).astype(np.uint8)

        cv2.imwrite(str(args.output / "mapping_distance.png"), mapping_distance)

        # cv2.imwrite(
        #     str(args.output / "mapping_angle.png"),
        #     export_angles(img_field_elevation_vectors_4),
        # )

        # cv2.imwrite(
        #     str(args.output / "mapping_angle.png"),
        #     export_angles(img_directions),
        # )
        #
        cv2.imwrite(
            str(args.output / "mapping_angle_0.png"),
            export_angles(img_field_elevation_vectors_0),
        )

        cv2.imwrite(
            str(args.output / "mapping_angle_1.png"),
            export_angles(img_field_elevation_vectors_1),
        )

        cv2.imwrite(
            str(args.output / "mapping_angle_2.png"),
            export_angles(img_field_elevation_vectors_2),
        )

        cv2.imwrite(
            str(args.output / "mapping_angle_3.png"),
            export_angles(img_field_elevation_vectors_3),
        )

        cv2.imwrite(
            str(args.output / "mapping_angle_4.png"),
            export_angles(img_field_elevation_vectors_4),
        )

        cv2.imwrite(
            str(args.output / "mapping_angle_5.png"),
            export_angles(img_field_elevation_vectors_5),
        )

        cv2.imwrite(
            str(args.output / "mapping_angle_6.png"),
            export_angles(img_field_elevation_vectors_6),
        )

        cv2.imwrite(
            str(args.output / "mapping_angle_7.png"),
            export_angles(img_field_elevation_vectors_7),
        )

        cv2.imwrite(str(args.output / "mapping_line_length.png"), mapping_line_length)

        cv2.imwrite(str(args.output / "mapping_flat.png"), mapping_flat)

    print(f"total time: {(datetime.datetime.now() - timer_start).total_seconds():5.2f}s")
