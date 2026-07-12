import numpy as np
import open3d as o3d
import os

from Py3DRF import Scene, Mesh

suzanne = o3d.io.read_triangle_mesh('smooth_suzanne.obj')

V = np.asarray(suzanne.vertices)
V = np.column_stack((V[:, 0], V[:, 2], V[:, 1]))
F = np.asarray(suzanne.triangles)

float_attr = V[:, 2]
color_attr = np.hstack((V, np.ones((V.shape[0], 1))))
# normalize color_attr into [0,1] plausible range for a css color string
color_attr = (color_attr - color_attr.min()) / (color_attr.max() - color_attr.min())

scene = Scene(backend="plotly", resolution=(800, 800))

mesh = Mesh("Suzanne", V, [], F, scale=(0.5, 0.5, 0.5))
mesh.addColorAttribute(color_attr, "color_attr")
mesh.material.setColorAttributeAsColor("color_attr")
mesh.setShadeSmooth()

camera = mesh.getCamera(azimuth=70 * np.pi/180, elevation=20 * np.pi/180, distance=2.5)

scene.addObject(mesh)
scene.addCamera(camera)

scene.renderToFile(os.path.join(os.getcwd(), "output_plotly.html"))
print("plotly render ok")

# also exercise the point-cloud path
mesh2 = Mesh("SuzannePoints", V, [], F, scale=(0.5, 0.5, 0.5))
mesh2.addFloatAttribute(float_attr, "float_attr")
mesh2.material.setFloatAttributeAsColor("float_attr", colors=[(0, 0, 1, 1), (1, 0, 0, 1)])
mesh2.asPointCloud(radius=0.02)

scene2 = Scene(backend="plotly", resolution=(800, 800))
scene2.addObject(mesh2)
scene2.renderToFile(os.path.join(os.getcwd(), "output_plotly_points.html"))
print("plotly point cloud render ok")

# a method Plotly genuinely doesn't support should raise, not silently pass
try:
    scene.setSamples(64)
    print("ERROR: should have raised")
except Exception as e:
    print("OK, unsupported method raised:", type(e).__name__, "-", e)
