import bpy
import os

import argparse
import sys
from pathlib import Path
import math


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


parser = ArgumentParserForBlender()
parser.add_argument("--input", type=Path, default="mesh.ply", help="Input filename [PLY]")
parser.add_argument("--rotX", type=float, default=0, help="rotation X in degrees [float]")
parser.add_argument("--rotY", type=float, default=0, help="rotation Y in degrees [float]")
parser.add_argument("--rotZ", type=float, default=0, help="rotation Z in degrees [float]")
args = parser.parse_args()

scene = bpy.context.scene

# cleanup

bpy.ops.object.select_all(action="DESELECT")

obj = scene.objects.get(args.input.stem)
if obj is not None:
    print(f"deleting existing object {obj}")
    obj.select_set(True)
    bpy.ops.object.delete()

# import

bpy.ops.wm.ply_import(filepath="build/mesh.ply")

obj = bpy.context.selected_objects[0]
obj.rotation_euler = (
    math.radians(args.rotX),
    math.radians(args.rotY),
    math.radians(args.rotZ),
)

# add material (derived from the vertice colors)

mesh = obj.data
mat = bpy.data.materials.get("vertex_colors")
if mat is None:
    mat = bpy.data.materials.new(name="vertex_colors")

mat.use_nodes = True

if mesh.materials:
    mesh.materials[0] = mat
else:
    mesh.materials.append(mat)

nodes = mat.node_tree.nodes
principled_bsdf_node = nodes.get("Principled BSDF")
principled_bsdf_node.inputs["Roughness"].default_value = 1.0

vertex_color_node = None
if not "VERTEX_COLOR" in [node.type for node in nodes]:
    vertex_color_node = nodes.new(type="ShaderNodeVertexColor")
else:
    vertex_color_node = nodes.get("Color Attribute")

vertex_color_node.layer_name = "Col"

links = mat.node_tree.links
link = links.new(vertex_color_node.outputs[0], principled_bsdf_node.inputs[0])

# Save & Quit

bpy.ops.wm.save_as_mainfile()
bpy.ops.wm.quit_blender()
