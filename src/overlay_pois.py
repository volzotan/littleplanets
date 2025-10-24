import argparse
import json
from pathlib import Path
import shapely
from shapely.geometry import LineString, Point, MultiLineString
import numpy as np
import math
import zipfile
from fastkml import kml
import pygeoif
from util.hershey import HersheyFont
from util.misc import write_linestrings_to_npz, visualize_linestrings, _rotate_linestrings

from loguru import logger


def _latlon_to_cartesian(xs, ys, zs=None) -> tuple[np.array]:
    x = np.zeros([len(xs)])
    y = np.zeros([len(xs)])
    z = np.zeros([len(xs)])

    lats = np.array(ys) / 180 * math.pi
    lons = (np.array(xs) - 90) / 360 * math.tau

    for i in range(len(xs)):
        x[i] = math.cos(lats[i]) * math.cos(lons[i])
        y[i] = math.cos(lats[i]) * math.sin(lons[i])
        z[i] = math.sin(lats[i])

    return x, y, z


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("pois", type=Path, help="Position of interest data [JSON]")
    parser.add_argument("--rotX", type=float, default=0, help="rotation X in degrees [float]")
    parser.add_argument("--rotY", type=float, default=0, help="rotation Y in degrees [float]")
    parser.add_argument("--rotZ", type=float, default=0, help="rotation Z in degrees [float]")
    parser.add_argument("--grid-num-lat", type=int, default=0, help="number of latitude grid lines")
    parser.add_argument("--grid-num-lon", type=int, default=0, help="number of longitude grid lines")
    parser.add_argument("--output", type=Path, default="overlay.npz", help="Output filename [NPZ]")
    parser.add_argument("--circle-radius", type=float, default=0.02, help="POI circle radius [float]")
    parser.add_argument("--font-size", type=float, default=0.03, help="Label font size [float]")
    parser.add_argument("--visualize", action="store_true", default=False, help="Enable interactive visualization")

    args = parser.parse_args()

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

        linestrings_poi = []

        if "path" not in poi:

            radius = poi.get("circle_radius", args.circle_radius)
            circle = Point([0, 0]).buffer(radius).boundary.segmentize(0.05)
            linestrings_poi.append(circle)

        if "name" in poi:

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
            linestrings_poi.append(text)


        for i in range(len(linestrings_poi)):  # add Z
            ls = linestrings_poi[i]
            coords = shapely.get_coordinates(ls)
            new_col = np.full([coords.shape[0], 1], 1.0)
            coords_with_z = np.concatenate((coords, new_col), axis=1)
            linestrings_poi[i] = LineString(coords_with_z)

        poi_rot_x = (poi["lat"] * -1 + 90.0) / 180 * math.pi
        poi_rot_z = (poi["lon"]) / 360 * math.tau

        linestrings += _rotate_linestrings(linestrings_poi, *[poi_rot_x, 0, poi_rot_z])

        if "path" in poi:

            with zipfile.ZipFile(args.pois.parent / poi["path"], "r") as kmz:
                kml_filenames = [f for f in kmz.namelist() if f.lower().endswith(".kml")]

                if len(kml_filenames) > 1:
                    logger.warning(f"Multiple KML files in KMZ {poi['path']}")

                with kmz.open(kml_filenames[0]) as kml_file:
                    k = kml.KML().from_string(kml_file.read())

                    for document in k.features:
                        for placemark in document.features:
                            name = placemark.name
                            geom = placemark.geometry

                            if type(geom) is not pygeoif.LineString:
                                logger.warning(f"Geometry {name} is not a (PyGeoIf.)LineString but {type(geom)}")
                                continue

                            shapely_geom = shapely.geometry.shape(geom.__geo_interface__)

                            # post-processing / line smoothing
                            # shapely_geom = shapely.segmentize(shapely_geom, max_segment_length=1e-2)
                            # shapely_geom = shapely_geom.buffer(0.1).buffer(-0.1)
                            # shapely_geom = shapely.simplify(shapely_geom, 1e-2, preserve_topology=True)

                            shapely_geom_xyz = shapely.ops.transform(_latlon_to_cartesian, shapely_geom)
                            linestrings.append(shapely_geom_xyz)

    # ROTATE

    linestrings = _rotate_linestrings(linestrings, *BLENDER_ROTATION)

    # VISUALIZE

    if args.visualize:
        visualize_linestrings(linestrings).show()

    # EXPORT

    write_linestrings_to_npz(args.output, linestrings)


if __name__ == "__main__":
    main()