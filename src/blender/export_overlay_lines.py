import bpy
from mathutils import Vector

import argparse
import sys
from pathlib import Path
import numpy as np

DEBUG = False

LIGHT_NAME = "Light"


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


# ---


def find_layer_collection(layer_coll, coll_name):
    if layer_coll.collection.name == coll_name:
        return layer_coll
    for child in layer_coll.children:
        found = find_layer_collection(child, coll_name)
        if found:
            return found
    return None


parser = ArgumentParserForBlender()
parser.add_argument("--input", type=Path, default="overlay.npz", help="Input filename (NPZ)")
parser.add_argument("--output", type=Path, default="overlay_visible.npz", help="Output filename (NPZ)")
parser.add_argument("--raycast-from-light", action="store_true", help="Use the light location as the raycasting origin instead of the camera")
parser.add_argument("--save-to", type=Path, default=None, help="Output filename of the blender file (blend)")
args = parser.parse_args()

scene = bpy.context.scene

collection_name = f"overlay_{args.input.stem.lower()}"

# cleanup

bpy.ops.object.select_all(action="DESELECT")

collection = bpy.data.collections.get(collection_name)
if collection is not None:
    for obj in collection.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.collections.remove(collection)

# import

overlay_collection = bpy.data.collections.new(collection_name)
scene.collection.children.link(overlay_collection)

overlay_npz = np.load(args.input)

for lines in overlay_npz.values():
    curve_data = bpy.data.curves.new("overlay_curve", "CURVE")
    curve_data.dimensions = "3D"

    polyline = curve_data.splines.new("POLY")
    polyline.points.add(len(lines) - 1)
    for i, point_coords in enumerate(lines):
        x, y, z = point_coords
        polyline.points[i].co = (x, y, z, 1)

    curve_obj = bpy.data.objects.new("overlay_curve", curve_data)
    curve_data.bevel_depth = 0.002

    overlay_collection.objects.link(curve_obj)

# Ray casting

cam = scene.camera
if cam.data.type != "PERSP":
    raise ValueError("Non-perspective cameras not supported")

debug_collection = bpy.data.collections.get("debug")
if DEBUG:
    if debug_collection is not None:
        for obj in debug_collection.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(debug_collection)
    debug_collection = bpy.data.collections.new("debug")
    scene.collection.children.link(debug_collection)

visibility_list = []
bpy.context.view_layer.depsgraph.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
origin = cam.matrix_world.translation

if args.raycast_from_light:
    light_obj = bpy.data.objects.get(LIGHT_NAME)

    if not light_obj:
        raise Exception(f"No light object found for name {LIGHT_NAME}")

    if light_obj.type != "LIGHT":
        raise Exception(f"Object found for name {LIGHT_NAME} is not of type 'LIGHT'")

    origin = light_obj.location

for lines in overlay_npz.values():
    visible_points = np.zeros([lines.shape[0]], dtype=bool)

    for i, point_coords in enumerate(lines):
        destination = Vector(point_coords)
        direction = (destination - origin).normalized()
        hit, location, norm, idx, obj, mw = scene.ray_cast(depsgraph, origin, direction)

        if obj is not None and obj.name.startswith("overlay_curve"):
            # TODO: if two elements both lie along the ray cast from the camera,
            # the first object would be visible and the second is blocked by the mesh,
            # yet the second would still be marked as visible due to the hit on the first one.

            visible_points[i] = True

            if DEBUG:
                curve_data = bpy.data.curves.new("debug_curve", "CURVE")
                curve_data.dimensions = "3D"

                polyline = curve_data.splines.new("POLY")
                polyline.points.add(1)

                polyline.points[0].co = (*origin, 1)
                polyline.points[1].co = (*destination, 1)

                curve_obj = bpy.data.objects.new("debug_curve", curve_data)
                curve_data.bevel_depth = 0.002
                debug_collection.objects.link(curve_obj)

    visibility_list.append(visible_points)

# Save & Quit

# pass on a file-like object, not a path, to prevent numpy from appending ".npz" to the filename
with open(args.output, "wb") as f:
    np.savez(f, *visibility_list)

layer_collection = find_layer_collection(bpy.context.view_layer.layer_collection, collection_name)
if layer_collection:
    layer_collection.exclude = True

if not DEBUG:
    if args.save_to is not None:
        bpy.ops.wm.save_as_mainfile(filepath=str(args.save_to))
    # else:
    #     bpy.ops.wm.save_as_mainfile()
    bpy.ops.wm.quit_blender()
