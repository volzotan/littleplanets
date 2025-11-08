# This file must be run within Blender

import bpy
import numpy as np

import sys
import argparse
from pathlib import Path
from datetime import datetime
from mathutils import Vector

# camera property calculations from:
# https://blender.stackexchange.com/a/177530
# https://blender.stackexchange.com/a/120063

DEBUG = False


# src: https://blender.stackexchange.com/a/134596/118415
class ArgumentParserForBlender(argparse.ArgumentParser):
    def _get_argv_after_doubledash(self):
        try:
            idx = sys.argv.index("--")
            return sys.argv[idx + 1 :]  # the list after '--'
        except ValueError:  # '--' not in the list:
            return []

    # overrides superclass
    def parse_args(self):
        return super().parse_args(args=self._get_argv_after_doubledash())


parser = ArgumentParserForBlender()
parser.add_argument("--output", type=Path, default="Raytrace.npy", help="Output filename [NPY]")
args = parser.parse_args()

context = bpy.context
scene = context.scene
vl = context.view_layer

debug_collection = bpy.data.collections.get("debug")
if DEBUG:
    if debug_collection is not None:
        for obj in debug_collection.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(debug_collection)
    debug_collection = bpy.data.collections.new("debug")
    scene.collection.children.link(debug_collection)

cam = scene.camera
camd = cam.data

if camd.type != "PERSP":
    raise ValueError("Non-perspective cameras not supported")

frame = camd.view_frame(scene=bpy.context.scene)
top_right = frame[0]
bottom_right = frame[1]
bottom_left = frame[2]
top_left = frame[3]

resolution_x = int(bpy.context.scene.render.resolution_x * (bpy.context.scene.render.resolution_percentage / 100))
resolution_y = int(bpy.context.scene.render.resolution_y * (bpy.context.scene.render.resolution_percentage / 100))

x_range = np.linspace(top_left[0], top_right[0], resolution_x)
y_range = np.linspace(top_left[1], bottom_left[1], resolution_y)

timer_start = datetime.now()

values = np.full([resolution_y, resolution_x, 3], np.nan, dtype=float)
origin = cam.matrix_world.translation

for x in range(resolution_x):
    for y in range(resolution_y):
        direction = Vector((x_range[x], y_range[y], top_left[2]))
        direction.rotate(cam.matrix_world.to_quaternion())
        hit, location, norm, idx, obj, mw = scene.ray_cast(vl.depsgraph, origin, direction.normalized())

        print(f"> progress: {x / resolution_x * 100:5.2f} %", end="\r")

        if hit:
            values[y, x, :] = location

        if DEBUG:
            curve_data = bpy.data.curves.new("debug_curve", "CURVE")
            curve_data.dimensions = "3D"

            polyline = curve_data.splines.new("POLY")
            polyline.points.add(1)

            polyline.points[0].co = (*origin, 1)
            polyline.points[1].co = (*location, 1)

            curve_obj = bpy.data.objects.new("debug_curve", curve_data)
            curve_data.bevel_depth = 0.002
            debug_collection.objects.link(curve_obj)


with open(args.output, "wb") as f:
    np.save(f, values)

print("Completed in: {:.3f}s".format((datetime.now() - timer_start).total_seconds()))

if not DEBUG:
    bpy.ops.wm.quit_blender()
