import bpy
import os

import argparse
import sys
from pathlib import Path


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


# ---

parser = ArgumentParserForBlender()
parser.add_argument("--output", type=Path, default="mesh_blender.ply", help="Output filename [PLY]")
args = parser.parse_args()

obj = bpy.context.active_object
if obj is None:
    raise Exception("No active object to export")

bpy.ops.object.select_all(action="DESELECT")
obj.select_set(True)
bpy.context.view_layer.objects.active = obj

bpy.ops.wm.ply_export(filepath=str(args.output))

bpy.ops.wm.quit_blender()
