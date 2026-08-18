import numpy as np
import pytest

from Py3DRF.io.nifti import NiftiVolume, load_nifti


def _identity_affine(spacing=(1.0, 1.0, 1.0)):
    affine = np.eye(4)
    affine[0, 0], affine[1, 1], affine[2, 2] = spacing
    return affine


def test_selection_for_axial_identity_affine():
    volume = NiftiVolume(np.zeros((4, 5, 6)), _identity_affine())
    selection = volume.selectionFor("axial", 3)

    assert selection.axis == "axial"
    assert selection.index == 3
    np.testing.assert_allclose(selection.origin, [2.0, 2.5, 3.0])
    np.testing.assert_allclose(selection.normal, [0.0, 0.0, 1.0])


def test_selection_for_respects_affine_scale():
    volume = NiftiVolume(np.zeros((4, 4, 4)), _identity_affine(spacing=(2.0, 2.0, 2.0)))
    selection = volume.selectionFor("sagittal", 1)

    np.testing.assert_allclose(selection.origin, [2.0, 4.0, 4.0])
    np.testing.assert_allclose(selection.normal, [1.0, 0.0, 0.0])


def test_selection_for_rejects_unknown_axis():
    volume = NiftiVolume(np.zeros((2, 2, 2)), np.eye(4))
    with pytest.raises(ValueError):
        volume.selectionFor("diagonal", 0)


def test_get_slice_vertex_positions_axial():
    array = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(float)  # (I=2, J=3, K=4)
    volume = NiftiVolume(array, np.eye(4))
    selection = volume.selectionFor("axial", 1)  # K fixed at 1

    mesh = volume.getSlice(selection)
    world = mesh._worldVertices()

    assert len(mesh.vertices) == 2 * 3
    assert len(mesh.faces) == 1 * 2

    expected = np.array(
        [
            [0, 0, 1], [0, 1, 1], [0, 2, 1],
            [1, 0, 1], [1, 1, 1], [1, 2, 1],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(world, expected)


def test_get_slice_vertex_positions_sagittal_with_step():
    array = np.arange(4 * 4 * 4).reshape(4, 4, 4).astype(float)
    volume = NiftiVolume(array, np.eye(4))
    selection = volume.selectionFor("sagittal", 2)  # I fixed at 2

    mesh = volume.getSlice(selection, step=2)
    world = mesh._worldVertices()

    # free dims for sagittal are (J, K), each strided by 2 over a length-4 axis -> 2x2 grid
    assert len(mesh.vertices) == 2 * 2
    expected = np.array(
        [
            [2, 0, 0], [2, 0, 2],
            [2, 2, 0], [2, 2, 2],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(world, expected)


def test_get_slice_intensity_window_normalizes_to_unit_range():
    array = np.zeros((2, 2, 2))
    array[:, :, 0] = [[0, 50], [100, 150]]
    volume = NiftiVolume(array, np.eye(4))
    selection = volume.selectionFor("axial", 0)

    mesh = volume.getSlice(selection, window=(0, 150))
    intensities, domain = mesh.float_attributes["intensity"]

    np.testing.assert_allclose(intensities, [0.0, 1 / 3, 2 / 3, 1.0])
    assert domain == "POINT"


def test_get_slice_default_percentile_clip_stays_in_unit_range():
    values = np.linspace(0, 100, 100)
    values[0] = -10000
    values[-1] = 10000
    array = values.reshape(1, 10, 10).astype(float)
    volume = NiftiVolume(array, np.eye(4))
    selection = volume.selectionFor("sagittal", 0)

    mesh = volume.getSlice(selection)
    intensities, _ = mesh.float_attributes["intensity"]

    assert intensities.min() == 0.0
    assert intensities.max() == 1.0
    assert not np.any(np.isnan(intensities))


def test_get_slice_uses_base_color_attribute_not_emission():
    # Base color, not emission, must drive the intensity ramp: emission is a
    # Blender-only concept that ScenePlotly/ScenePolyscope both ignore, so an
    # emission-driven slice would render blank on those backends.
    volume = NiftiVolume(np.arange(2 * 2 * 2).reshape(2, 2, 2).astype(float), np.eye(4))
    mesh = volume.getSlice(volume.selectionFor("axial", 0))

    assert mesh.material.color_attribute == "intensity"
    assert mesh.material.emission_color_attribute is None


def test_load_nifti_round_trip(tmp_path):
    nib = pytest.importorskip("nibabel")

    array = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(np.float32)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    img = nib.Nifti1Image(array, affine)
    path = tmp_path / "test.nii"
    nib.save(img, str(path))

    volume = load_nifti(str(path))

    assert volume.array.shape == (2, 3, 4)
    np.testing.assert_allclose(np.diagonal(volume.affine)[:3], [2.0, 2.0, 2.0], atol=1e-5)
