import math

from collections import deque

from dataclasses import dataclass

import rtree

import numpy as np
from loguru import logger
from shapely import LineString, Polygon, Point, MultiLineString

from run import Mesh

MAX_ITERATIONS = 2_000_000

Linepoint = list[float, float, float]


@dataclass
class FlowlineHatcherConfig:
    LINE_DISTANCE: tuple[float, float] | list[float] = (8e-3, 8e-3)
    LINE_DISTANCE_END_FACTOR = 0.5
    LINE_STEP_DISTANCE: float = 5e-3

    LINE_MAX_LENGTH: tuple[float, float] = (0.6, 0.6)

    # max difference (in radians) in slope between line points
    MAX_ANGLE_DISCONTINUITY: float = math.pi / 2
    MIN_INCLINATION: float = 0.001  # 50.0

    # How many line segments should be skipped before the next seedpoint is extracted
    SEEDPOINT_EXTRACTION_SKIP_LINE_SEGMENTS: int = 20


def _normalize_vector(v: np.array) -> np.array:
    return v / np.linalg.norm(v)


def ray_triangle_intersection(
    vertex_0: np.array,
    vertex_1: np.array,
    vertex_2: np.array,
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    epsilon: float = 1e-6,
) -> tuple[float, float, float] | None:
    edge_1 = vertex_1 - vertex_0
    edge_2 = vertex_2 - vertex_0

    p_vec = np.cross(ray_direction, edge_2)

    determinant = np.dot(p_vec, edge_1)

    if np.abs(determinant) < epsilon:
        return None

    inv_determinant = 1.0 / determinant

    t_vec = ray_origin - vertex_0
    u = np.dot(p_vec, t_vec) * inv_determinant
    if u < 0.0 or u > 1.0:
        return None

    q_vec = np.cross(t_vec, edge_1)
    v = np.dot(q_vec, ray_direction) * inv_determinant
    if v < 0.0 or (u + v) > 1.0:
        return None

    t = np.dot(q_vec, edge_2) * inv_determinant
    if t < epsilon:
        return None

    # barycentric to world coordinate system
    return u * vertex_0 + v * vertex_1 + (1 - u - v) * vertex_2


def find_point_on_surface(m: Mesh, candidate_faces: list[int], p: np.array) -> tuple[np.array, int] | tuple[None, None]:
    origin = np.array([0, 0, 0])

    for ind in candidate_faces:
        f = m.faces[ind]
        surface_point = ray_triangle_intersection(m.points[f[0]], m.points[f[1]], m.points[f[2]], origin, p)

        if surface_point is not None:
            return p, ind

    return None, None


def build_neighbour_maps(m: Mesh) -> dict[int, set[int]]:
    point_to_face = {}
    for f in range(len(m.faces)):
        for p in m.faces[f].tolist():
            point_to_face[p] = point_to_face.get(p, []) + [f]

    face_to_face = {}
    for f in range(len(m.faces)):
        for p in m.faces[f].tolist():
            face_to_face[f] = face_to_face.get(f, set()).union(set(point_to_face[p]))

    face_to_face_second = {}
    for f in range(len(m.faces)):
        first_order_neighbours = face_to_face[f]
        for fon in first_order_neighbours:
            face_to_face_second[f] = face_to_face_second.get(f, set()).union(face_to_face[fon])

    return face_to_face, face_to_face_second


class FlowlineHatcher:
    """
    A Python implementation of the paper "Creating Evenly-Spaced Streamlines
    of Arbitrary Density" by Bruno Jobard and Wilfrid Lefer.

    """

    def __init__(
        self,
        mesh: Mesh,
        mapping_elevation: np.ndarray,
        mapping_distance: np.ndarray,
        mapping_max_length: np.ndarray,
        mapping_flat: np.ndarray,
        config: FlowlineHatcherConfig,
        initial_seed_points: list[Linepoint] = [],
    ):
        self.m = mesh
        self.config = config

        self.mapping_elevation = mapping_elevation
        self.mapping_distance = mapping_distance
        self.mapping_max_length = mapping_max_length
        self.mapping_flat = mapping_flat

        self.initial_seed_points = initial_seed_points

        # tree linepoints
        index = rtree.index
        p = index.Property()
        p.dimension = 3
        self.tree_linepoints = index.Index(properties=p)

        # tree triangles
        self.tree_triangles = index.Index(properties=p)
        for i, t in enumerate(self.m.faces):
            center_point = np.mean(
                np.array(
                    [
                        self.m.points[t[0]],
                        self.m.points[t[1]],
                        self.m.points[t[2]],
                    ]
                ),
                axis=0,
            )
            self.tree_triangles.insert(i, center_point.tolist())

        self.neighbour_map_1, self.neighbour_map_2 = build_neighbour_maps(self.m)

    def _cartesian_to_lat_lon(self, p: Linepoint) -> tuple[float, float]:
        r = np.linalg.norm(np.array(p))
        lat = np.arccos(p[2] / r)
        lon = np.atan2(p[1], p[0])
        return lat, lon

    def _lat_lon_to_pixel(self, lat: float, lon: float, shape: list[int]) -> tuple[int, int]:
        x = int(lon / math.tau * shape[1])
        y = int(lat / math.pi * shape[0])
        return x, y

    def _map_elevation(self, p: Linepoint) -> np.array:
        lat, lon = self._cartesian_to_lat_lon(p)
        x, y = self._lat_lon_to_pixel(lat, lon, self.mapping_elevation.shape)

        elevation = self.mapping_elevation[y, x]

        return np.array(
            [
                elevation * math.sin(lat) * math.cos(lon),
                elevation * math.sin(lat) * math.sin(lon),
                elevation * math.cos(lat),
            ]
        )

    def _map_flat(self, p: Linepoint) -> bool:
        return False  # TODO

    def _map_line_distance(self, p: Linepoint) -> float:
        lat, lon = self._cartesian_to_lat_lon(p)
        x, y = self._lat_lon_to_pixel(lat, lon, self.mapping_distance.shape)

        diff = self.config.LINE_DISTANCE[1] - self.config.LINE_DISTANCE[0]
        return self.config.LINE_DISTANCE[0] + diff * (self.mapping_distance[y, x] / 255)

    def _map_line_max_length(self, p: Linepoint) -> float:
        return self.config.LINE_MAX_LENGTH[0]  # TODO

    def _collision(self, p: Linepoint, factor: float = 1.0) -> bool:
        neighbour = list(self.tree_linepoints.nearest(p, 1, objects="raw"))

        if neighbour is None or len(neighbour) == 0:
            return False

        dist = np.linalg.norm(np.array(p) - np.array(neighbour[0]))
        if dist * factor < self._map_line_distance(p):
            return True

        return False

    def _next_point(self, p: Linepoint, forwards: bool) -> Linepoint | None:
        # direction
        nearest_triangle_ind = list(self.tree_triangles.nearest(p, 1))[0]
        _, new_nearest_triangle_ind = find_point_on_surface(self.m, self.neighbour_map_1[nearest_triangle_ind], np.array(p))
        if new_nearest_triangle_ind is None:
            _, new_nearest_triangle_ind = find_point_on_surface(self.m, self.neighbour_map_2[nearest_triangle_ind], np.array(p))
            if new_nearest_triangle_ind is None:
                print("no nearest_triangle_ind found")
        else:
            nearest_triangle_ind = new_nearest_triangle_ind

        # v1 = self.m.field_vectors[nearest_triangle_ind]
        v1 = self.m.field_elevation_vectors[nearest_triangle_ind]

        if self._map_flat(p):
            return None

        dir = 1
        if not forwards:
            dir = -1

        v2 = np.array(p) + np.array(v1) * self.config.LINE_STEP_DISTANCE * dir

        # v3, _ = find_point_on_surface(self.m, self.neighbour_map_1[nearest_triangle_ind], v2)
        # if v3 is None:
        #     v3, _ = find_point_on_surface(self.m, self.neighbour_map_2[nearest_triangle_ind], v2)
        #     if v3 is None:
        #         print("no surface point found")
        #         return None

        v3 = self._map_elevation(v2)

        if self._collision(v3.tolist(), factor=self.config.LINE_DISTANCE_END_FACTOR):
            return None

        # Sanity check:
        # if a point is right on the edge between two faces, aligned with the field vectors, we've got a weird
        # liftoff effect. Fix: ensure closest distance between point and plane of the face should not exceed tolerance
        # c = self.m.centers[nearest_triangle_ind]
        # n = self.m.normals[nearest_triangle_ind]
        # distance_to_face = np.dot(n, v2 - c)
        # if abs(distance_to_face) > 1e-2:
        #     return None

        # if self.config.MAX_ANGLE_DISCONTINUITY > 0:
        #     a2 = self._map_angle(x2, y2)
        #
        #     if abs(a2 - a1) > self.config.MAX_ANGLE_DISCONTINUITY:
        #         # print("MAX_ANGLE_DISCONTINUITY")
        #         return None

        return v3.tolist()

    def _seed_points(self, line_points: list[Linepoint] | deque[Linepoint]) -> list[Linepoint]:
        num_seedpoints = 1
        seed_points: list[Linepoint] = []

        if len(line_points) > self.config.SEEDPOINT_EXTRACTION_SKIP_LINE_SEGMENTS:
            num_seedpoints = (len(line_points) - 1) // self.config.SEEDPOINT_EXTRACTION_SKIP_LINE_SEGMENTS

        for i in range(num_seedpoints):
            p1 = np.array(line_points[i * self.config.SEEDPOINT_EXTRACTION_SKIP_LINE_SEGMENTS])
            p2 = np.array(line_points[i * self.config.SEEDPOINT_EXTRACTION_SKIP_LINE_SEGMENTS + 1])

            # midpoint
            p3 = (p1 + p2) / 2

            # closest triangle normal vector
            triangle_index = list(self.tree_triangles.nearest(p3, 1))
            if triangle_index is None or len(triangle_index) == 0:
                continue

            n = np.array(self.m.normals[triangle_index[0]])
            c = np.array(self.m.centers[triangle_index[0]])

            dist = self._map_line_distance(p3) * 2.01

            if i % 2 == 0:
                dist *= -1

            midpoint_v = p3 - p2
            midpoint_v = _normalize_vector(midpoint_v)

            cross = p3 + np.cross(midpoint_v, n) * dist
            cross_projected = cross - np.dot(cross - c, n) * n

            # print(f"{self._map_line_distance(p3)} {np.linalg.norm(p3 - cross)} {np.linalg.norm(p1 - cross)} {np.linalg.norm(p2 - cross)}")
            # exit()

            # seed_points.append(cross_projected.tolist())
            seed_points.append(cross.tolist())

        return seed_points

    # def generate_starting_points(self) -> None:
    #     # point grid for starting points, grid distance is mean line distance
    #     num_gridpoints_x = int(self.dimensions[0] / (self.config.LINE_DISTANCE[0] * 1.5))
    #     num_gridpoints_y = int(self.dimensions[1] / (self.config.LINE_DISTANCE[0] * 1.5))
    #
    #     for i in np.linspace(1, self.dimensions[0] - 1, endpoint=False, num=num_gridpoints_x):
    #         for j in np.linspace(1, self.dimensions[1] - 1, endpoint=False, num=num_gridpoints_y):
    #             self.starting_points.append((i, j))

    def visualize(
        self,
        linestrings: list[LineString],
        new_line_points: deque[Linepoint],
        starting_points: deque[Linepoint],
    ) -> None:
        import pyvista as pv

        plotter = pv.Plotter()

        # X, Y, Z axes
        plotter.add_mesh(
            pv.Spline(np.array([[0, 0, 0], [1, 0, 0]]), 10).tube(radius=0.02),
            color=[255, 0, 0],
        )
        plotter.add_mesh(
            pv.Spline(np.array([[0, 0, 0], [0, 1, 0]]), 10).tube(radius=0.02),
            color=[0, 255, 0],
        )
        plotter.add_mesh(
            pv.Spline(np.array([[0, 0, 0], [0, 0, 1]]), 10).tube(radius=0.02),
            color=[0, 0, 255],
        )

        # mesh
        points_pv = np.stack(self.m.points)
        faces_pv = np.hstack([[3, *face] for face in self.m.faces])
        pvmesh = pv.PolyData(points_pv, faces_pv)
        plotter.add_mesh(pvmesh, opacity=0.5)  # , show_edges=True) #), opacity=0.5)

        for i in range(len(self.m.faces)):
            arrow = pv.Arrow(self.m.centers[i], self.m.centers[i] + self.m.normals[i], scale=0.2)
            plotter.add_mesh(arrow, color=[255, 255, 255])

        for lp in starting_points:
            plotter.add_mesh(pv.Sphere(0.01, np.array(lp)), color=[255, 0, 0])

        # light axis
        # light_axis = _normalize_vector(np.array([0, 1, 1]))
        # spline = pv.Spline(np.array([[0, 0, 0], light_axis], dtype=np.float32), 10).tube(radius=0.02)
        # plotter.add_mesh(spline, color=[255, 255, 0])

        for line in linestrings:
            spline = pv.Spline(np.array(list(line.coords))).tube(radius=0.01)
            plotter.add_mesh(spline, color=[125, 125, 125])

        spline = pv.Spline(new_line_points).tube(radius=0.01)
        plotter.add_mesh(spline, color=[0, 0, 0])

        plotter.show()

    def hatch(self) -> list[LineString]:
        linestrings = []
        starting_points: deque[Linepoint] = deque(self.initial_seed_points)

        # self.generate_starting_points()

        for i in range(MAX_ITERATIONS):
            if i >= MAX_ITERATIONS - 1:
                logger.warning("max iterations exceeded")

            if len(starting_points) == 0:
                break

            seed = starting_points.popleft()

            if self._collision(seed):
                continue

            line_points = deque([seed])

            # follow gradient upwards
            for _ in range(int(self.config.LINE_MAX_LENGTH[1] / self.config.LINE_STEP_DISTANCE)):
                p = self._next_point(line_points[-1], True)

                if p is None:
                    break

                if len(line_points) * self.config.LINE_STEP_DISTANCE > self._map_line_max_length(p):
                    break

                line_points.append(p)

            # follow gradient downwards
            for _ in range(int(self.config.LINE_MAX_LENGTH[1] / self.config.LINE_STEP_DISTANCE)):
                p = self._next_point(line_points[0], False)

                if p is None:
                    break

                if len(line_points) * self.config.LINE_STEP_DISTANCE > self._map_line_max_length(p):
                    break

                line_points.appendleft(p)

            if len(line_points) < 2:
                continue

            linestrings.append(LineString(line_points))

            # seed points
            # new_seed_points = self._seed_points(line_points)
            # starting_points.extendleft(new_seed_points)

            # collision checks
            for lp in line_points:
                self.tree_linepoints.insert(1, lp, obj=lp)

            # self.visualize(linestrings, line_points, starting_points)

        return linestrings


if __name__ == "__main__":
    pass
