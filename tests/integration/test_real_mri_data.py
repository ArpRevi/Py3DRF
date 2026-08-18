from pathlib import Path

import numpy as np
import pytest

from Py3DRF import Scene, Mesh
from Py3DRF.io import load_nifti

_TESTDATA_ROOT = Path(__file__).resolve().parents[3] / "Py3DRF_testdata"
_NII_PATH = _TESTDATA_ROOT / "brain-scan" / "NII" / "ATE23_seg_1mm_Crop_relabelled_merged.nii"
_OBJ_PATH = _TESTDATA_ROOT / "brain-scan" / "segment_02.obj"

pytestmark = pytest.mark.skipif(
    not (_NII_PATH.exists() and _OBJ_PATH.exists()),
    reason="Py3DRF_testdata not checked out next to this repo (private repo, needs CI's TESTDATA_TOKEN, or a local sibling checkout)",
)


def test_real_volume_slices_and_segmentation_mesh_build_and_render(tmp_path):
    import open3d as o3d  # only this test needs it -- see the `testdata` extra

    volume = load_nifti(str(_NII_PATH))

    selections = [
        volume.selectionFor(axis, volume.array.shape[dim] // 2)
        for axis, dim in (("sagittal", 0), ("coronal", 1), ("axial", 2))
    ]

    obj = o3d.io.read_triangle_mesh(str(_OBJ_PATH))
    vertices = np.asarray(obj.vertices)
    faces = np.asarray(obj.triangles)
    assert len(vertices) > 0 and len(faces) > 0
    segmentation_mesh = Mesh("segment_02", vertices, [], faces)

    scene = Scene(backend="plotly", resolution=(400, 400))
    for selection in selections:
        # step=4: downsample -- this is a real, full-resolution clinical
        # volume, unlike the small synthetic arrays in tests/io/test_nifti.py.
        scene.addObject(volume.getSlice(selection, step=4))
    scene.addObject(segmentation_mesh)

    out = tmp_path / "real_mri_smoke.html"
    scene.renderToFile(str(out))
    assert out.exists() and out.stat().st_size > 0
