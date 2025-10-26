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
from util.misc import write_linestrings_to_npz, visualize_linestrings, rotate_linestrings, dash_linestring

from loguru import logger

DASH_LENGTH = 0.01
PAUSE_LENGTH = 0.01


def _latlon_to_cartesian(xs: list[float], ys: list[float], zs: list[float]=None) -> tuple[np.array]:
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

def _latlon_to_rotation_angles(lat: float, lon: float) -> tuple[float, float, float]:
    poi_rot_x = (lat * -1 + 90.0) / 180 * math.pi
    poi_rot_z = (lon) / 360 * math.tau

    return (poi_rot_x, 0, poi_rot_z)


def _linestrings_add_z(linestrings: list[LineString]) -> list[LineString]:
    for i in range(len(linestrings)):  # add Z
        ls = linestrings[i]
        coords = shapely.get_coordinates(ls)
        new_col = np.full([coords.shape[0], 1], 1.0)
        coords_with_z = np.concatenate((coords, new_col), axis=1)
        linestrings[i] = LineString(coords_with_z)

    return linestrings

def _linestrings_flip_ud(linestrings: list[LineString]) -> list[LineString]:
    return [LineString(shapely.get_coordinates(l) * np.array([1, -1])) for l in linestrings]


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
        linestrings_poi: list[LineString] = []
        marker_geometry = None

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

                            # add path to linestrings, not linestrings_poi since no further rotation is necessary
                            # linestrings.append(shapely_geom_xyz)
                            linestrings += dash_linestring(shapely_geom_xyz, DASH_LENGTH, PAUSE_LENGTH)

                            # TODO: rotate path back to 0,0 so it can act as a marker_object

        else:  # no path, draw simple circle
            radius = poi.get("circle_radius", args.circle_radius)
            circle = Point([0, 0]).buffer(radius).boundary.segmentize(0.05)
            linestrings += rotate_linestrings(_linestrings_add_z([circle]), * _latlon_to_rotation_angles(poi["lat"], poi["lon"]))
            marker_geometry = circle

        if "name" in poi:
            # circular text
            # path_text_baseline = Point([0, 0]).buffer(CIRCLE_RADIUS + 0.01).boundary.segmentize(0.01)
            # linestrings_along_path = font.lines_for_text(
            #     poi["name"], FONT_SIZE, path=path_text_baseline, align=Align.CENTER, reverse_path=True
            # )

            # linear text
            text = font.lines_for_text(poi["name"], args.font_size)
            angle = poi.get("label_angle", 0.0)

            geom = MultiLineString(text)

            center_x = (geom.bounds[2]-geom.bounds[0]) / 2
            center_y = (geom.bounds[3]-geom.bounds[1]) / 2

            x = 0

            if math.isclose(angle, 90):
                x += -center_x
            elif math.isclose(angle, 270):
                x += -center_x
            elif angle > 90 and angle < 270:
                x += -(geom.bounds[2]-geom.bounds[0])
            else:
                pass

            text = [shapely.affinity.translate(ls, xoff=x, yoff=center_y) for ls in text]

            if "label_lat" in poi and "label_lon" in poi:

                if args.visualize:
                    circle = Point([0, 0]).buffer(0.005).boundary
                    linestrings += rotate_linestrings(_linestrings_add_z([circle]), *_latlon_to_rotation_angles(poi["label_lat"], poi["label_lon"]))

                text = _linestrings_flip_ud(text)
                text = _linestrings_add_z(text)
                linestrings += rotate_linestrings(text, *_latlon_to_rotation_angles(poi["label_lat"], poi["label_lon"]))
            else:

                # TODO: extent of the marker_geometry bounding box in relation to the label_angle
                dist = args.circle_radius * 1.30

                text = [
                    shapely.affinity.translate(ls, xoff=dist * math.cos(math.radians(angle)), yoff=dist * math.sin(math.radians(angle))) for ls in text
                ]

                text = _linestrings_flip_ud(text)
                text = _linestrings_add_z(text)
                linestrings += rotate_linestrings(text, *_latlon_to_rotation_angles(poi["lat"], poi["lon"]))



    # ROTATE

    linestrings = rotate_linestrings(linestrings, *BLENDER_ROTATION)

    # FILTER

    linestrings = [ls for ls in linestrings if np.min(np.array(ls.coords)[:, 2]) > 0]

    # VISUALIZE

    if args.visualize:
        visualize_linestrings(linestrings).show()

    # EXPORT

    write_linestrings_to_npz(args.output, linestrings)


if __name__ == "__main__":
    main()
