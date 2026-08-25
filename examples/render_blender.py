import bpy
import numpy as np
import open3d as o3d
import os


from Py3DRF import Scene
from Py3DRF import Mesh
from Py3DRF import Location, Rotation, Scale

# Data load
#suzanne = o3d.io.read_triangle_mesh('s0677_vertebrae_L3.nii.g_1.stl')
suzanne = o3d.io.read_triangle_mesh(r"..\Py3DRF_testdata\suzanne\smooth_suzanne.obj")

# Convert to array and switch from y-up to z-up
V = np.asarray(suzanne.vertices)
V = np.column_stack((V[:, 0], V[:, 2], V[:, 1]))
F = np.asarray(suzanne.triangles)

# Generate attributes
#float_attr = V[:, 2]

xyz = V
xyz_norm = (xyz - xyz.min(axis=0)) / (xyz.max(axis=0) - xyz.min(axis=0))
float_attr = xyz_norm[:, 2]
color_attr = np.hstack((xyz_norm, np.ones((V.shape[0], 1))))
#color_attr = np.hstack((V, np.ones((V.shape[0], 1))))

# Init scene
scene = Scene(resolution=(1080, 1080), engine="CYCLES", deafult_sun=True, transparent=False)

# Init mesh
mesh = Mesh("Suzanne", V, [], F, scale=Scale(0.1, 0.1, 0.1), location=Location(0, 0, 0))

# Link attributes to mesh
mesh.addColorAttribute(color_attr, "color_attr")
mesh.addFloatAttribute(float_attr, "float_attr")

# Link mesh attributes to material
mesh.material.setColorAttributeAsColor("color_attr")
#mesh.material.setFloatAttributeAsEmissionColor("float_attr")
#mesh.material.setEmissionStrength(10)

# shade smooth
mesh.setShadeSmooth()

# use render as pointclouds
#mesh.asPointCloud()
# thickness is the diameter of the edge cylinders, in the mesh's local units
"""
mesh.asWireframe(thickness=0.03)
mesh.asPointCloud(radius=1)
"""

# get floor shadow catcher
floor = mesh.getFloor(shadow_catcher=False, size=(10, 10))

# add color and material to floor
floor.material.setColor((0.5, 0.5, 0.5, 1))
floor.material.setRoughness(0.5)

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