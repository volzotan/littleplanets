import math

from collections import deque
from dataclasses import dataclass

import numpy as np
from loguru import logger
from shapely import LineString

MAX_ITERATIONS = 20_000_000


@dataclass
class FlowlineHatcherConfig:
    # distance between lines in mm
    LINE_DISTANCE: tuple[float, float] | list[float] = (0.3, 5.0)
    LINE_DISTANCE_END_FACTOR = 0.5

    # distance between points constituting a line in mm
    LINE_STEP_DISTANCE: float = 0.3

    LINE_MAX_LENGTH: tuple[float, float] = (10, 50)

    # lines shorter than LINE_MIN_LENGTH will be discarded
    LINE_MIN_LENGTH: float = 1.0

    # max difference (in radians) in slope between line points
    MAX_ANGLE_DISCONTINUITY: float = math.pi / 2
    MIN_INCLINATION: float = 0.001  # 50.0

    # How many line segments should be skipped before the next seedpoint is extracted
    SEEDPOINT_EXTRACTION_SKIP_LINE_SEGMENTS: int = 5

    # SCALE_ADJUSTMENT_VALUE: float = 0.3

    COLLISION_APPROXIMATE: bool = True
    VIZ_LINE_THICKNESS: int = 5


class FlowlineHatcher:
    """
    A Python implementation of the paper "Creating Evenly-Spaced Streamlines
    of Arbitrary Density" by Bruno Jobard and Wilfrid Lefer.
    """

    def __init__(
        self,
        dimensions: list[int] | tuple[int, int],
        mapping_distance: np.ndarray,
        mapping_angles: np.ndarray,
        mapping_max_length: np.ndarray,
        mapping_flat: np.ndarray,
        config: FlowlineHatcherConfig,
        initial_seed_points: list[tuple[float, float]] = [],
        exclusion_points: list[tuple[float, float]] = [],
    ):
        self.dimensions = dimensions
        self.config = config

        self.scale_x = mapping_distance.shape[1] / dimensions[0]
        self.scale_y = mapping_distance.shape[0] / dimensions[1]

        self.distance = mapping_distance
        self.angles = (mapping_angles.astype(float) / 255.0) * math.tau - math.pi
        self.max_length = mapping_max_length
        self.flat = mapping_flat

        self.initial_seed_points = initial_seed_points

        if self.config.COLLISION_APPROXIMATE:
            self.MAPPING_FACTOR_COLLISION = int(math.ceil(1 / self.config.LINE_DISTANCE[0]))
            self.point_raster = np.zeros(
                [
                    self.dimensions[1] * self.MAPPING_FACTOR_COLLISION,
                    self.dimensions[0] * self.MAPPING_FACTOR_COLLISION,
                ],
                dtype=bool,
            )
        else:
            self.point_bins = []
            self.bin_size = self.config.LINE_DISTANCE[1]
            self.num_bins_x = int(self.dimensions[0] // self.bin_size + 1)
            self.num_bins_y = int(self.dimensions[1] // self.bin_size + 1)

            for x in range(self.num_bins_x):
                self.point_bins.append([np.empty([0, 2], dtype=float)] * self.num_bins_y)

        self._register_for_collision_check(exclusion_points)

    def _map_line_distance(self, x: float, y: float) -> float:
        return float(
            self.config.LINE_DISTANCE[0]
            + self.distance[int(y * self.scale_y), int(x * self.scale_x)] / 255 * (self.config.LINE_DISTANCE[1] - self.config.LINE_DISTANCE[0])
        )

    def _map_angle(self, x: float, y: float) -> float:
        return float(self.angles[int(y * self.scale_y), int(x * self.scale_x)])

    def _map_line_max_length(self, x: float, y: float) -> float:
        return float(
            self.config.LINE_MAX_LENGTH[0]
            + self.max_length[int(y * self.scale_y), int(x * self.scale_x)] / 255 * (self.config.LINE_MAX_LENGTH[1] - self.config.LINE_MAX_LENGTH[0])
        )

    def _map_flat(self, x: float, y: float) -> bool:
        return self.flat[int(y * self.scale_y), int(x * self.scale_x)] > 0

    def _collision_approximate(self, x: float, y: float, factor: float) -> bool:
        if x >= self.dimensions[0]:
            return True

        if y >= self.dimensions[1]:
            return True

        min_d = int(self._map_line_distance(x, y) * factor * self.MAPPING_FACTOR_COLLISION)

        rm_x = int(x * self.MAPPING_FACTOR_COLLISION)
        rm_y = int(y * self.MAPPING_FACTOR_COLLISION)

        return np.any(
            self.point_raster[
                max(rm_y - min_d, 0) : rm_y + min_d + 1,
                max(rm_x - min_d, 0) : rm_x + min_d + 1,
            ]
        )

    def _collision_precise(self, x: float, y: float, factor: float) -> bool:
        min_d = self._map_line_distance(x, y) * factor
        bin_pos = [int(x / self.bin_size), int(y / self.bin_size)]

        bins = [
            [bin_pos[0], bin_pos[1] - 1],
            [bin_pos[0] - 1, bin_pos[1]],
            [bin_pos[0], bin_pos[1]],
            [bin_pos[0] + 1, bin_pos[1]],
            [bin_pos[0], bin_pos[1] + 1],
        ]

        for ix, iy in bins:
            if ix < 0 or ix >= self.num_bins_x:
                continue

            if iy < 0 or iy >= self.num_bins_y:
                continue

            arr = self.point_bins[ix][iy]

            if arr.shape[0] == 0:
                continue

            distance = np.linalg.norm(arr - np.array([x, y]), axis=1)
            if np.any(distance < min_d):
                return True

        return False

    def _collision(self, x: float, y: float, factor: float = 1.0) -> bool:
        if self.config.COLLISION_APPROXIMATE:
            return self._collision_approximate(x, y, factor)
        else:
            return self._collision_precise(x, y, factor)

    def _register_for_collision_check(self, line_points: list[tuple[float, float]]) -> None:
        for lp in line_points:
            x = int(lp[0])
            y = int(lp[1])
            if self.config.COLLISION_APPROXIMATE:
                self.point_raster[
                    int(y * self.MAPPING_FACTOR_COLLISION),
                    int(x * self.MAPPING_FACTOR_COLLISION),
                ] = True
            else:
                # self.point_map[f"{x},{y}"].append(lp)
                # self.point_bins[int(x/self.bin_size)][int(y/self.bin_size)].append(lp)
                self.point_bins[int(x / self.bin_size)][int(y / self.bin_size)] = np.append(
                    self.point_bins[int(x / self.bin_size)][int(y / self.bin_size)],
                    [lp],
                    axis=0,
                )

    def _next_point(self, x1: float, y1: float, forwards: bool) -> tuple[float, float] | None:
        a1 = self._map_angle(x1, y1)

        if self._map_flat(x1, y1):
            return None

        dir = 1
        if not forwards:
            dir = -1

        x2 = x1 + self.config.LINE_STEP_DISTANCE * math.cos(a1) * dir
        y2 = y1 + self.config.LINE_STEP_DISTANCE * math.sin(a1) * dir

        if x2 < 0 or x2 >= self.dimensions[0] or y2 < 0 or y2 >= self.dimensions[1]:
            return None

        if self._collision(x2, y2, factor=self.config.LINE_DISTANCE_END_FACTOR):
            return None

        if self.config.MAX_ANGLE_DISCONTINUITY > 0:
            a2 = self._map_angle(x2, y2)

            if abs(a2 - a1) > self.config.MAX_ANGLE_DISCONTINUITY:
                # print("MAX_ANGLE_DISCONTINUITY")
                return None

        return x2, y2

    def _seed_points(self, line_points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        num_seedpoints = 1
        seed_points = []

        if len(line_points) > self.config.SEEDPOINT_EXTRACTION_SKIP_LINE_SEGMENTS:
            num_seedpoints = (len(line_points) - 1) // self.config.SEEDPOINT_EXTRACTION_SKIP_LINE_SEGMENTS

        for i in range(num_seedpoints):
            x1, y1 = line_points[i * self.config.SEEDPOINT_EXTRACTION_SKIP_LINE_SEGMENTS]
            x2, y2 = line_points[i * self.config.SEEDPOINT_EXTRACTION_SKIP_LINE_SEGMENTS + 1]

            # midpoint
            x3 = x1 + (x2 - x1) / 2.0
            y3 = y1 + (y2 - y1) / 2.0

            a1 = math.atan2(y1 - y3, x1 - x3)

            a2 = a1
            if i % 2 == 0:
                a2 += math.radians(90)
            else:
                a2 -= math.radians(90)

            x4 = self._map_line_distance(x3, y3)
            y4 = 0

            x5 = x4 * math.cos(a2) - y4 * math.sin(a2) + x3
            y5 = x4 * math.sin(a2) + y4 * math.cos(a2) + y3

            if x5 < 0 or x5 >= self.dimensions[0] or y5 < 0 or y5 >= self.dimensions[1]:
                continue

            seed_points.append((x5, y5))

        return seed_points

    def hatch(self) -> list[LineString]:
        linestrings = []
        starting_points = deque()
        starting_points_priority = deque(self.initial_seed_points)

        # point grid for starting points, grid distance is mean line distance
        num_gridpoints_x = int(self.dimensions[0] / (self.config.LINE_DISTANCE[0] * 1.5))
        num_gridpoints_y = int(self.dimensions[1] / (self.config.LINE_DISTANCE[0] * 1.5))

        for i in np.linspace(1, self.dimensions[0] - 1, endpoint=False, num=num_gridpoints_x):
            for j in np.linspace(1, self.dimensions[1] - 1, endpoint=False, num=num_gridpoints_y):
                starting_points.append([i, j])

        for i in range(MAX_ITERATIONS):
            if i >= MAX_ITERATIONS - 1:
                logger.warning("max iterations exceeded")

            if len(starting_points_priority) + len(starting_points) == 0:
                break

            seed = None
            if len(starting_points_priority) > 0:
                seed = starting_points_priority.popleft()
            else:
                seed = starting_points.popleft()

            if self._collision(*seed):
                continue

            line_points = deque([seed])

            # follow gradient upwards
            for _ in range(int(self.config.LINE_MAX_LENGTH[1] / self.config.LINE_STEP_DISTANCE)):
                p = self._next_point(*line_points[-1], True)

                if p is None:
                    break

                if len(line_points) * self.config.LINE_STEP_DISTANCE > self._map_line_max_length(int(p[0]), int(p[1])):
                    break

                line_points.append(p)

            # follow gradient downwards
            for _ in range(int(self.config.LINE_MAX_LENGTH[1] / self.config.LINE_STEP_DISTANCE)):
                p = self._next_point(*line_points[0], False)

                if p is None:
                    break

                if len(line_points) * self.config.LINE_STEP_DISTANCE > self._map_line_max_length(int(p[0]), int(p[1])):
                    break

                line_points.appendleft(p)

            if len(line_points) * self.config.LINE_STEP_DISTANCE < self.config.LINE_MIN_LENGTH:
                continue

            linestrings.append(LineString(line_points))

            # seed points
            starting_points.extendleft(self._seed_points(line_points))

            # collision checks
            self._register_for_collision_check(line_points)

        return linestrings
