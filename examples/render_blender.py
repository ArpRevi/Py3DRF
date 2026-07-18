import bpy
import numpy as np
import open3d as o3d
import os


from Py3DRF import Scene
from Py3DRF import Mesh
from Py3DRF import Location, Rotation, Scale

# Data load
"""suzanne = o3d.io.read_triangle_mesh('000000_tumoredbrain.stl')"""
suzanne = o3d.io.read_triangle_mesh('smooth_suzanne.obj')

# Convert to array and switch from y-up to z-up
V = np.asarray(suzanne.vertices)
V = np.column_stack((V[:, 0], V[:, 2], V[:, 1]))
F = np.asarray(suzanne.triangles)

# Generate attributes
float_attr = V[:, 2]
color_attr = np.hstack((V, np.ones((V.shape[0], 1))))

# Init scene
scene = Scene(resolution=(1080, 1080), engine="CYCLES", deafult_sun=True)

# Init mesh
mesh = Mesh("Suzanne", V, [], F, scale=Scale(0.5, 0.5, 0.5))

# Link attributes to mesh
mesh.addColorAttribute(color_attr, "color_attr")
mesh.addFloatAttribute(float_attr, "float_attr")

# Link mesh attributes to material
mesh.material.setColorAttributeAsColor("color_attr")
mesh.material.setFloatAttributeAsEmissionColor("float_attr")
mesh.material.setEmissionStrength(10)

# shade smooth
mesh.setShadeSmooth()

# use render as pointclouds
mesh.asPointCloud()

# get floor shadow catcher
floor = mesh.getFloor()

# get camera centered on mesh
camera = mesh.getCamera(azimuth=70 * np.pi/180, elevation=20 * np.pi/180, distance=2.5)

# link objects and camera to scene
scene.addObject(mesh)
scene.addObject(floor)
scene.addCamera(camera)

# render and save file

scene.setShadowCatcherAlpha(0.75)
scene.setSamples(8)
scene.renderToFile(os.path.join(os.getcwd(), "output.png"))

# saveToFile used to be a (broken/dead) free function imported from Py3DRF. Saving a
# .blend file is a bpy operation, so it now lives on Scene, the package's only
# bpy-aware class.
# scene.saveToFile(os.path.join(os.getcwd(), "test.blend"))