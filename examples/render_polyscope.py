import numpy as np
import open3d as o3d
import os

from Py3DRF import Scene, Mesh

suzanne = o3d.io.read_triangle_mesh('smooth_suzanne.obj')

V = np.asarray(suzanne.vertices)
V = np.column_stack((V[:, 0], V[:, 2], V[:, 1]))
F = np.asarray(suzanne.triangles)

float_attr = V[:, 2]

scene = Scene(backend="polyscope", resolution=(800, 800))

mesh = Mesh("Suzanne", V, [], F, scale=(0.5, 0.5, 0.5))
mesh.addFloatAttribute(float_attr, "float_attr")
mesh.material.setFloatAttributeAsColor("float_attr", colors=[(0, 0, 1, 1), (1, 0, 0, 1)])
mesh.setShadeSmooth()

camera = mesh.getCamera(azimuth=70 * np.pi/180, elevation=20 * np.pi/180, distance=2.5)

scene.addObject(mesh)
scene.addCamera(camera)
scene.renderToFile(os.path.join(os.getcwd(), "output_polyscope.png"))
print("polyscope render ok")

# point cloud path
mesh2 = Mesh("SuzannePoints", V, [], F, scale=(0.5, 0.5, 0.5))
mesh2.asPointCloud(radius=0.02)

scene2 = Scene(backend="polyscope", resolution=(800, 800))
scene2.addObject(mesh2)
scene2.addCamera(camera)
scene2.renderToFile(os.path.join(os.getcwd(), "output_polyscope_points.png"))
print("polyscope point cloud render ok")

# adding a SunLight should raise cleanly, not be silently dropped
from Py3DRF import SunLight
try:
    scene2.addObject(SunLight())
    print("ERROR: should have raised")
except Exception as e:
    print("OK, unsupported object raised:", type(e).__name__, "-", e)
