from pathlib import Path
import trimesh
import shapely
from shapely.geometry import LineString, Point
import numpy as np
import math

# read mesh

ASSETS_DIR = Path("../assets")
INPUT_FILE = ASSETS_DIR / "Moon_Z.ply"

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
a
    lines_rotated = []
    for line in lines:
        lines_rotated.append(shapely.ops.transform(lambda x, y, z: R_z @ R_y @ R_x @ np.array([x, y, z]), line))

    return lines_rotated

linestrings_for_projection = []
linestrings_for_projection.append(Point([0, 0]).buffer(0.10).boundary.segmentize(0.1))
linestrings_for_projection.append(Point([0, 0]).buffer(0.20).boundary.segmentize(0.1))
linestrings_for_projection.append(Point([0, 0]).buffer(0.30).boundary.segmentize(0.1))

# add Z
for i in range(len(linestrings_for_projection)):
    ls = linestrings_for_projection[i]
    coords = shapely.get_coordinates(ls)
    new_col = np.full([coords.shape[0], 1], 1.0)
    coords_with_z = np.concatenate((coords, new_col), axis=1)
    linestrings_for_projection[i] = LineString(coords_with_z)

linestrings_for_projection = _rotate_linestrings(linestrings_for_projection, math.pi/4, 0, math.pi/4)
linestrings_for_projection = [_project_linestring(l, P, scaling_factor) for l in linestrings_for_projection]



# read mesh

mesh = trimesh.load(INPUT_FILE)
print(mesh.vertices)


# rotate and project POIs on mesh surface

# export XYZ linestrings as JSON