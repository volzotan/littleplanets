from pathlib import Path

import bpy

import math
import argparse
import sys

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

parser = ArgumentParserForBlender()
parser.add_argument("--render-output-dir", type=Path, help="Output directory for rendered images")
args, _ = parser.parse_known_args()

bpy.context.scene.render.filepath = str(args.render_output_dir)

for node in bpy.context.scene.node_tree.nodes:
    if not node.type == "OUTPUT_FILE":
        continue

    node.base_path = str(args.render_output_dir)


if not DEBUG:
    bpy.ops.wm.save_as_mainfile()
    bpy.ops.wm.quit_blender()
