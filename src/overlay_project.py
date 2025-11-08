import argparse
import math
from pathlib import Path
from datetime import datetime

import numpy as np
import shapely
import trimesh
import rtree
from shapely.geometry import LineString

from util.misc import write_linestrings_to_npz

SCALE = 1.02


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
        z = dist * math.cos(lats[i]) * scale

        proj[i, :] = [x, y, z]

    return proj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", type=Path, help="Input mesh path [PLY]")
    parser.add_argument("linestrings", type=Path, help="Input LineStrings of overlay [NPZ]")
    parser.add_argument("--output", type=Path, default="overlay_projected.npz", help="Output filename [NPZ]")
    args = parser.parse_args()

    timer_start = datetime.now()

    linestrings = [LineString(e) for e in list(np.load(args.linestrings).values())]

    mesh = trimesh.load(args.mesh)
    index = rtree.index
    p = index.Property()
    p.dimension = 3
    tree = index.Index(properties=p)
    for i, v in enumerate(mesh.vertices.tolist()):
        tree.insert(i, v, obj=v)

    projected_linestrings = [LineString(project_vertices(tree, shapely.get_coordinates(l, include_z=True), SCALE)) for l in linestrings]
    write_linestrings_to_npz(args.output, projected_linestrings)

    print("Completed overlay projection of {} in: {:.3f}s".format(args.mesh, (datetime.now() - timer_start).total_seconds()))


if __name__ == "__main__":
    main()
