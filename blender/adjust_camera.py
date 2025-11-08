import math

import bpy
import os

import argparse
import sys
from pathlib import Path

DEBUG = False

# src: https://blender.stackexchange.com/a/134596/118415
class ArgumentParserForBlender(argparse.ArgumentParser):
    def _get_argv_after_doubledash(self):
        try:
            idx = sys.argv.index("--")
            return sys.argv[idx + 1 :]  # the list after '--'
        except ValueError as e:  # '--' not in the list:
            return []

    # overrides superclass
    def parse_args(self):
        return super().parse_args(args=self._get_argv_after_doubledash())

    def parse_known_args(self):
        return super().parse_known_args(args=self._get_argv_after_doubledash())

# ---

def _calculate_z_distance_circle(focal_length: float, radius: float, sensor_size: float = 36.0) -> float:
    fov = 2 * math.atan(sensor_size / (2 * focal_length))

    # one of two tangent points (the x negative one) of a line with slope fov/2 and
    # a circle of a given radius at the origin
    tangent_slope = math.tan(fov/2)
    tangent_point_x = -(tangent_slope * radius) / (math.sqrt(1 + math.pow(tangent_slope, 2)))
    tangent_point_y = (radius) / (math.sqrt(1 + math.pow(tangent_slope, 2)))

    # X axis intersection of a line going through the tangent point with slope fov/2
    return (tangent_point_x - tangent_point_y / tangent_slope) * -1

parser = ArgumentParserForBlender()
# parser.add_argument("--camera-z", type=float, help="Camera Z position (float)"))
parser.add_argument("--horizontal-width", type=float, default=2.2, help="Camera Z distance to cover N units of the Y axis (float)")
parser.add_argument("--camera-focal-length", type=float, default=50.0, help="Camera focal length (float)")
args, _ = parser.parse_known_args()

cam_ob = bpy.context.scene.camera

if cam_ob is None:
    raise Exception("No scene camera found")

camera_z = _calculate_z_distance_circle(args.camera_focal_length, args.horizontal_width/2)

cam_ob.location = (cam_ob.location[0], cam_ob.location[1], camera_z)
cam_ob.data.lens = args.camera_focal_length

if not DEBUG:
    bpy.ops.wm.save_as_mainfile()
    bpy.ops.wm.quit_blender()