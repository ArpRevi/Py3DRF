import numpy as np
import open3d as o3d
import os

from Py3DRF import Scene, Mesh, Scale

suzanne = o3d.io.read_triangle_mesh(r"..\Py3DRF_testdata\suzanne\smooth_suzanne.obj")


V = np.asarray(suzanne.vertices)
V = np.column_stack((V[:, 0], V[:, 2], V[:, 1]))
F = np.asarray(suzanne.triangles)

float_attr = V[:, 2]

scene = Scene(backend="polyscope", resolution=(800, 800))

mesh = Mesh("Suzanne", V, [], F, scale=Scale(0.5, 0.5, 0.5))
mesh.addFloatAttribute(float_attr, "float_attr")
mesh.material.setFloatAttributeAsColor("float_attr", colors=[(0, 0, 1, 1), (1, 0, 0, 1)])
mesh.setShadeSmooth()

camera = mesh.getCamera(azimuth=70 * np.pi/180, elevation=20 * np.pi/180, distance=2.5)

scene.addObject(mesh)
scene.addCamera(camera)
scene.renderToFile(os.path.join(os.getcwd(), "output_polyscope.png"))
print("polyscope render ok")

# point cloud path
mesh2 = Mesh("SuzannePoints", V, [], F, scale=Scale(0.5, 0.5, 0.5))
mesh2.asPointCloud(radius=0.01)

scene2 = Scene(backend="polyscope", resolution=(800, 800))
scene2.addObject(mesh2)
scene2.addCamera(camera)
scene2.renderToFile(os.path.join(os.getcwd(), "output_polyscope_points.png"))
print("polyscope point cloud render ok")

# also exercise the wireframe path: a genuine screen-space edge overlay (constant
# pixel width, not a 3D tube like the Blender backend builds) on top of the shaded
# surface -- Polyscope can't hide the surface while keeping the overlay, so
# hide_surface must stay False here (see ScenePolyscope's module docstring)
from Py3DRF import Material
mesh3 = Mesh("SuzanneWireframe", V, [], F, scale=Scale(0.5, 0.5, 0.5))
edge_material = Material()
edge_material.setColor((1.0, 0.05, 0.05, 1))
mesh3.asWireframe(thickness=0.003, hide_surface=False, material=edge_material)

scene3 = Scene(backend="polyscope", resolution=(800, 800))
scene3.addObject(mesh3)
scene3.addCamera(camera)
scene3.renderToFile(os.path.join(os.getcwd(), "output_polyscope_wireframe.png"))
print("polyscope wireframe render ok")

# adding a SunLight should raise cleanly, not be silently dropped
from Py3DRF import SunLight
try:
    scene2.addObject(SunLight())
    print("ERROR: should have raised")
except Exception as e:
    print("OK, unsupported object raised:", type(e).__name__, "-", e)
