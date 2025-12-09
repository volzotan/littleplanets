import bpy

import math
import argparse
import sys

DEBUG = False

LIGHT_NAME = "Light"

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
    tangent_slope = math.tan(fov / 2)
    tangent_point_x = -(tangent_slope * radius) / (math.sqrt(1 + math.pow(tangent_slope, 2)))
    tangent_point_y = (radius) / (math.sqrt(1 + math.pow(tangent_slope, 2)))

    # X axis intersection of a line going through the tangent point with slope fov/2
    return (tangent_point_x - tangent_point_y / tangent_slope) * -1


parser = ArgumentParserForBlender()
parser.add_argument("--horizontal-width", type=float, default=2.2, help="Camera Z distance to cover N units of the Y axis (float)")
parser.add_argument("--camera-focal-length", type=float, default=50.0, help="Camera focal length (float)")
parser.add_argument("--resolution-x", type=int, default=1000, help="Camera resolution X (int)")
parser.add_argument("--resolution-y", type=int, default=1000, help="Camera resolution Y (int)")
parser.add_argument("--light-pos-x", type=float, default=0.5, help="Light position X (float)")
parser.add_argument("--light-pos-y", type=float, default=0, help="Light position Y (float)")
parser.add_argument("--light-pos-z", type=float, default=10, help="Light position Z (float)")
parser.add_argument("--light-size", type=float, default=0.5, help="Light size (float)")
parser.add_argument("--light-power", type=float, default=1500, help="Light power (float)")
args, _ = parser.parse_known_args()

# Camera

cam_ob = bpy.context.scene.camera

if cam_ob is None:
    raise Exception("No camera found in scene")

camera_z = _calculate_z_distance_circle(args.camera_focal_length, args.horizontal_width / 2)

cam_ob.location = (cam_ob.location[0], cam_ob.location[1], camera_z)
cam_ob.data.lens = args.camera_focal_length

scene = bpy.context.scene
render = scene.render

render.resolution_x = args.resolution_x
render.resolution_y = args.resolution_y

# Light

light_obj = bpy.data.objects.get(LIGHT_NAME)

if not light_obj:
    light_data = bpy.data.lights.new(name=LIGHT_NAME, type="AREA")
    light_obj = bpy.data.objects.new(name=LIGHT_NAME, object_data=light_data)
    bpy.context.collection.objects.link(light_obj)

if light_obj.type != "LIGHT":
    raise Exception(f"Object {light_obj} is not a Light object")

light_obj.location = (args.light_pos_x, args.light_pos_y, args.light_pos_z)
light_obj.data.size = args.light_size
light_obj.data.energy = args.light_power
light_obj.data.shape = "DISK"

if not DEBUG:
    bpy.ops.wm.save_as_mainfile()
    bpy.ops.wm.quit_blender()
