import numpy as np
import open3d as o3d
import os

from Py3DRF import Scene, Mesh, Scale

suzanne = o3d.io.read_triangle_mesh('s0677_vertebrae_L3.nii.g_1.stl')

V = np.asarray(suzanne.vertices)
V = np.column_stack((V[:, 0], V[:, 2], V[:, 1]))
F = np.asarray(suzanne.triangles)

scene = Scene(backend="usd")

mesh = Mesh("Suzanne", V, [], F, scale=Scale(0.01, 0.01, 0.01))
mesh.material.setColor((0.8, 0.4, 0.2, 1.0))
mesh.setShadeSmooth()

floor = mesh.getFloor()

camera = mesh.getCamera(azimuth=70 * np.pi/180, elevation=20 * np.pi/180, distance=2.5)

scene.addObject(mesh)
scene.addCamera(camera)
scene.addObject(floor)

scene.exportToFile(os.path.join(os.getcwd(), "output_usd.usda"))
print("usd export ok")

# exportToFile only writes the ASCII .usda format -- any other extension should raise

try:
    scene.exportToFile(os.path.join(os.getcwd(), "output_usd.usd"))
    print("ERROR: should have raised")
except Exception as e:
    print("OK, non-.usda path raised:", type(e).__name__, "-", e)

# USD is a scene-description format, not a renderer: renderToFile should raise, not
# silently produce nothing
try:
    scene.renderToFile(os.path.join(os.getcwd(), "output_usd.png"))
    print("ERROR: should have raised")
except Exception as e:
    print("OK, unsupported method raised:", type(e).__name__, "-", e)
    