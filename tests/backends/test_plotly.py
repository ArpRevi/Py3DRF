import numpy as np
import pytest

pytest.importorskip("plotly")

from Py3DRF import Mesh, NotSupportedByBackendError, Scene, SunLight
from Py3DRF.io.nifti import NiftiVolume

QUAD = dict(vertices=[(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], faces=[(0, 1, 2, 3)])


def _first_trace(scene):
    return scene._impl.figure.data[0]


def test_quad_is_triangulated_not_truncated():
    # Regression: go.Mesh3d takes triangle indices only, and the backend used to
    # pass faces[:, 0:3] -- silently dropping every quad's 4th corner, so half of
    # each quad never rendered.
    scene = Scene(backend="plotly")
    scene.addObject(Mesh("Quad", **QUAD))

    trace = _first_trace(scene)
    assert len(trace.i) == 2
    assert sorted(zip(map(int, trace.i), map(int, trace.j), map(int, trace.k))) == [
        (0, 1, 2),
        (0, 2, 3),
    ]


def test_mixed_face_lengths_do_not_raise():
    # Regression: ragged faces made np.asarray produce an object array, and
    # faces[:, 0] then raised an opaque IndexError.
    mesh = Mesh(
        "Mixed",
        vertices=[(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (2, 0, 0)],
        faces=[(0, 1, 2, 3), (1, 4, 2)],
    )
    scene = Scene(backend="plotly")
    scene.addObject(mesh)

    assert len(_first_trace(scene).i) == 3


def test_faceless_mesh_does_not_raise():
    scene = Scene(backend="plotly")
    scene.addObject(Mesh("Edges", vertices=[(0, 0, 0), (1, 0, 0)], edges=[(0, 1)]))

    assert len(_first_trace(scene).i) == 0


def test_nifti_slice_renders_every_quad():
    # The end-to-end case the truncation bug actually broke: a slice grid is all
    # quads, so half of every rendered MRI slice went missing.
    volume = NiftiVolume(np.random.rand(8, 8, 8), np.eye(4))
    slice_mesh = volume.getSlice(volume.selectionFor("axial", 4))

    scene = Scene(backend="plotly")
    scene.addObject(slice_mesh)

    assert len(_first_trace(scene).i) == 2 * len(slice_mesh.faces)


def test_float_attribute_drives_an_intensity_colorscale():
    mesh = Mesh("T", vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0)], faces=[(0, 1, 2)])
    mesh.addFloatAttribute([0.0, 0.5, 1.0], "v")
    mesh.material.setFloatAttributeAsColor("v", colors=[(0, 0, 0, 1), (1, 1, 1, 1)])

    scene = Scene(backend="plotly")
    scene.addObject(mesh)

    trace = _first_trace(scene)
    np.testing.assert_allclose(np.asarray(trace.intensity, dtype=float), [0.0, 0.5, 1.0])
    assert trace.colorscale is not None


def test_sun_light_is_rejected_rather_than_silently_dropped():
    scene = Scene(backend="plotly")
    with pytest.raises(NotSupportedByBackendError):
        scene.addObject(SunLight())


def test_unsupported_methods_raise():
    scene = Scene(backend="plotly")
    for call in (
        lambda: scene.setSamples(64),
        lambda: scene.setGamma(1.0),
        lambda: scene.setExposure(0.0),
        lambda: scene.setShadowCatcherAlpha(0.5),
        lambda: scene.setRenderEngine("CYCLES"),
        lambda: scene.setDevice("GPU"),
    ):
        with pytest.raises(NotSupportedByBackendError):
            call()


def test_render_to_html(tmp_path):
    scene = Scene(backend="plotly", resolution=(200, 200))
    scene.addObject(Mesh("Quad", **QUAD))

    out = tmp_path / "out.html"
    scene.renderToFile(str(out))

    assert out.exists() and out.stat().st_size > 0
