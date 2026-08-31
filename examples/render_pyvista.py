import numpy as np
import open3d as o3d
import os

from Py3DRF import Scene, Mesh, Scale, SunLight, Rotation, Material

suzanne = o3d.io.read_triangle_mesh(r"..\Py3DRF_testdata\suzanne\smooth_suzanne.obj")

V = np.asarray(suzanne.vertices)
V = np.column_stack((V[:, 0], V[:, 2], V[:, 1]))
F = np.asarray(suzanne.triangles)

float_attr = V[:, 2]

scene = Scene(backend="pyvista", resolution=(800, 800), transparent=False)

mesh = Mesh("Suzanne", V, [], F, scale=Scale(0.5, 0.5, 0.5))
mesh.addFloatAttribute(float_attr, "float_attr")
# Reproduced exactly via a custom pyvista.LookupTable, unlike Polyscope's
# fixed-colormap approximation of the same ramp.
mesh.material.setFloatAttributeAsColor("float_attr", colors=[(0, 0, 1, 1), (1, 0, 0, 1)])
mesh.setShadeSmooth()

camera = mesh.getCamera(azimuth=70 * np.pi/180, elevation=20 * np.pi/180, distance=2.5)

scene.addObject(mesh)
scene.addCamera(camera)
scene.addObject(SunLight(rotation=Rotation(-30 * np.pi/180, 0, -10 * np.pi/180), strength=5.0))
scene.renderToFile(os.path.join(os.getcwd(), "output_pyvista.png"))
print("pyvista render ok")

# point cloud path (icosphere glyphs, like Blender's point-cloud instances)
mesh2 = Mesh("SuzannePoints", V, [], F, scale=Scale(0.5, 0.5, 0.5))
mesh2.asPointCloud(radius=0.01, subdivison=2)

scene2 = Scene(backend="pyvista", resolution=(800, 800), transparent=False)
scene2.addObject(mesh2)
scene2.addCamera(camera)
scene2.addObject(SunLight(rotation=Rotation(-30 * np.pi/180, 0, -10 * np.pi/180), strength=5.0))
scene2.renderToFile(os.path.join(os.getcwd(), "output_pyvista_points.png"))
print("pyvista point cloud render ok")

# wireframe path: real 3D tube geometry (like Blender, not Plotly/Polyscope's flat
# screen-space lines), with support for a per-vertex thickness_attribute that
# Blender only approximates and Plotly/Polyscope reject outright
mesh3 = Mesh("SuzanneWireframe", V, [], F, scale=Scale(0.5, 0.5, 0.5))
edge_material = Material()
edge_material.setColor((1.0, 0.05, 0.05, 1))
mesh3.asWireframe(thickness=0.003, hide_surface=False, material=edge_material)

scene3 = Scene(backend="pyvista", resolution=(800, 800), transparent=False)
scene3.addObject(mesh3)
scene3.addCamera(camera)
scene3.addObject(SunLight(rotation=Rotation(-30 * np.pi/180, 0, -10 * np.pi/180), strength=5.0))
scene3.renderToFile(os.path.join(os.getcwd(), "output_pyvista_wireframe.png"))
print("pyvista wireframe render ok")

# a method PyVista genuinely doesn't support should raise, not silently pass
try:
    scene.setSamples(64)
    print("ERROR: should have raised")
except Exception as e:
    print("OK, unsupported method raised:", type(e).__name__, "-", e)
