import bpy
import numpy as np

from mathutils import Vector

# camera property calculations from:
# https://blender.stackexchange.com/a/177530
# https://blender.stackexchange.com/a/120063

context = bpy.context
scene = context.scene
vl = context.view_layer

cam = scene.camera
camd = cam.data

if camd.type != 'PERSP':
    raise ValueError('Non-perspective cameras not supported')

frame = camd.view_frame(scene=bpy.context.scene)
top_right = frame[0]
bottom_right = frame[1]
bottom_left = frame[2]
top_left = frame[3]

resolution_x = int(bpy.context.scene.render.resolution_x * (bpy.context.scene.render.resolution_percentage / 100))
resolution_y = int(bpy.context.scene.render.resolution_y * (bpy.context.scene.render.resolution_percentage / 100))

x_range = np.linspace(top_left[0], top_right[0], resolution_x)
y_range = np.linspace(top_left[1], bottom_left[1], resolution_y)

values = np.zeros([resolution_y, resolution_x, 3], dtype=np.float32)

origin = cam.matrix_world.translation

for x in range(resolution_x):
    for y in range(resolution_y):

        pixel_vector = Vector((x_range[x], y_range[y], top_left[2]))
        pixel_vector.rotate(cam.matrix_world.to_quaternion())
        hit, location, norm, idx, obj, mw = scene.ray_cast(vl.depsgraph, origin, pixel_vector)

        print(f"{x / resolution_x * 100:5.2f} %", end="\r")

        if hit:
            values[y, x, :] = location

print("raytracing finished")

with open("raytracing.npy", "wb") as f:
    np.save(f, values)
