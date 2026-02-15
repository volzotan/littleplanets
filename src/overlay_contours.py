import argparse
from pathlib import Path

import toml
from pydantic import BaseModel
from shapely.geometry import LineString, Polygon
import numpy as np
from util.misc import write_linestrings_to_npz, visualize_linestrings

import cv2

MAX_SEGMENT_LENGTH = 10.0

VISUALIZE = False

class OverlayContoursConfig(BaseModel):
    double_line_distance: float | None = None
    shrink: float | None = None
    simplify: float | None = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raytrace", type=Path, help="Raytracing distance raster (NPY)")
    parser.add_argument("--output", type=Path, default="overlay.npz", help="Output filename (NPZ)")
    parser.add_argument("--config", type=Path, help="Configuration file (TOML)")

    args = parser.parse_args()

    config = OverlayContoursConfig()
    if args.config is not None:
        with open(args.config, "r") as f:
            data = toml.load(f)
            config = OverlayContoursConfig.model_validate(data)

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

    if config.shrink is not None and config.shrink > 0:
        polygons_silhouette = [p.buffer(-config.shrink) for p in polygons_silhouette]

    if config.simplify is not None and config.simplify > 0:
        polygons_silhouette = [p.simplify(config.simplify) for p in polygons_silhouette]

    linestrings = [LineString(p.exterior.coords) for p in polygons_silhouette]

    # 2D TO 3D

    linestrings_3d = []
    for ls in linestrings:
        new_coords = np.array([img_pxpos[int(p[1]), int(p[0])] for p in ls.coords])
        new_coords = new_coords[~np.isnan(np.sum(new_coords, axis=1))]

        # new_coords[:, 2] = np.mean(new_coords[:, 2])

        if config.double_line_distance is not None:

            # Convert to spherical coordinates
            x, y, z = new_coords[:, 0], new_coords[:, 1], new_coords[:, 2]
            r = np.sqrt(x**2 + y**2 + z**2)
            theta = np.arctan2(y, x)
            phi = np.arccos(z / r)

            r += config.double_line_distance

            # Convert to Euclidean coordinates
            x_new = r * np.sin(phi) * np.cos(theta)
            y_new = r * np.sin(phi) * np.sin(theta)
            z_new = r * np.cos(phi)
            coords_double_line = np.column_stack((x_new, y_new, z_new))
            linestrings_3d.append(LineString(coords_double_line))

        linestrings_3d.append(LineString(new_coords))

    # VISUALIZE

    if VISUALIZE:
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
