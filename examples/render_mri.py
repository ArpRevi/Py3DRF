import os
import bpy
import open3d as o3d
import numpy as np

from Py3DRF import Scene
from Py3DRF import Mesh
from Py3DRF import Location, Rotation, Scale

from Py3DRF import Scene
from Py3DRF.io import load_nifti
from Py3DRF.mri import pick_slices

# Requires a real .nii/.nii.gz volume on disk, and a display (this example opens
# interactive matplotlib windows -- it can't run headless/on CI).

volume = load_nifti(r"..\Py3DRF_testdata\brain-scan\NII\ATE23_seg_1mm_Crop_relabelled_merged.nii")


suzanne = o3d.io.read_triangle_mesh(r"..\Py3DRF_testdata\brain-scan\segment_02.obj")
V = np.asarray(suzanne.vertices)
F = np.asarray(suzanne.triangles)
mesh = Mesh("Suzanne", V, [], F, scale=Scale(1., 1., 1.), location=Location(0, 0, 0))


# Step 1: interactively pick all three canonical slices in a single window
# (one image + slider per axis). Blocks until the window is closed; the
# returned SliceSelections carry the chosen axis/index plus their world-space
# origin/normal.
selections = pick_slices(volume, axes=("sagittal", "coronal", "axial"))
for selection in selections:
    print(f"{selection.axis}: index={selection.index} origin={selection.origin}")

# Step 2: build the slice meshes and render them together -- their correct,
# shared world-space positioning (via the volume's affine) is what makes the
# three planes visibly intersect in the render.
scene = Scene(backend="blender", resolution=(800, 800), deafult_sun=True, transparent=False)
for selection in selections:
    scene.addObject(volume.getSlice(selection))
    scene.addObject(mesh)
    scene.addCamera(mesh.getCamera(azimuth=70 * np.pi/180, elevation=20 * np.pi/180, distance=500))

scene.renderToFile(os.path.join(os.getcwd(), "output_mri.png"))
print("mri blender render ok")

"""
scene = Scene(backend="plotly")
for selection in selections:
    scene.addObject(volume.getSlice(selection))
    scene.addObject(mesh)

scene.renderToFile(os.path.join(os.getcwd(), "output_mri.png"))
print("mri plotly render ok")
"""

"""
scene = Scene(backend="polyscope", resolution=(800, 800))
for selection in selections:
    scene.addObject(volume.getSlice(selection))
    scene.addObject(mesh)
    scene.addCamera(mesh.getCamera(azimuth=70 * np.pi/180, elevation=20 * np.pi/180, distance=500))

scene.renderToFile(os.path.join(os.getcwd(), "output_mri.png"))
print("mri polyscope render ok")
"""