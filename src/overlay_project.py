import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import shapely
import trimesh
from scipy.spatial import KDTree
from shapely.geometry import LineString

from util.misc import write_linestrings_to_npz

SCALE = 1.02


def project_vertices(kdtree: KDTree, vertices: np.ndarray, points: np.ndarray, scale: float) -> np.ndarray:
    proj = np.zeros_like(points)

    r = np.sqrt(np.sum(np.power(points, 2), axis=1))
    lats = np.arccos(np.clip(points[:, 2] / r, -1, 1))
    lons = np.arctan2(points[:, 1], points[:, 0])

    distances, indices = kdtree.query(points)
    nearest_vertices = vertices[indices]
    dists = np.linalg.norm(nearest_vertices, axis=1)

    proj[:, 0] = dists * np.sin(lats) * np.cos(lons)
    proj[:, 1] = dists * np.sin(lats) * np.sin(lons)
    proj[:, 2] = dists * np.cos(lats) * scale

    return proj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", type=Path, help="Input mesh path (PLY)")
    parser.add_argument("linestrings", type=Path, help="Input LineStrings of overlay (NPZ)")
    parser.add_argument("--output", type=Path, default="overlay_projected.npz", help="Output filename (NPZ)")
    args = parser.parse_args()

    timer_start = datetime.now()

    linestrings = [LineString(e) for e in list(np.load(args.linestrings).values())]
    mesh = trimesh.load(args.mesh)
    kdtree = KDTree(mesh.vertices)

    projected_linestrings = []
    for i, linestring in enumerate(linestrings):
        coords = shapely.get_coordinates(linestring, include_z=True)
        projected_coords = project_vertices(kdtree, mesh.vertices, coords, SCALE)
        projected_linestrings.append(LineString(projected_coords))

    write_linestrings_to_npz(args.output, projected_linestrings)

    print("Completed overlay projection of {} in: {:.3f}s".format(args.mesh, (datetime.now() - timer_start).total_seconds()))


if __name__ == "__main__":
    main()
