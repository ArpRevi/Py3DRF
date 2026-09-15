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


def test_get_slice_face_winding_is_consistent_across_all_axes():
    # Regression test: coronal's (free0, free1, dim) grid-axis assignment is an
    # odd permutation of (0, 1, 2), unlike axial/sagittal's even ones, so its
    # quad winding came out flipped -- its face normal pointed exactly
    # opposite to SliceSelection.normal (and to the other two orientations'),
    # leaving it unlit from light directions that correctly lit sagittal/axial.
    # A non-trivial (rotated) affine is used since the bug is about
    # orientation/permutation, not about any specific affine's values -- a
    # clean, exactly-orthonormal rotation (not a hand-copied approximation)
    # so any mismatch here reflects a real sign flip, not affine imprecision.
    angle = 6 * np.pi / 180
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    affine = np.eye(4)
    affine[:3, :3] = rotation
    volume = NiftiVolume(np.random.rand(6, 6, 6), affine)

    for axis in ("sagittal", "coronal", "axial"):
        selection = volume.selectionFor(axis, 3)
        mesh = volume.getSlice(selection)
        world = mesh._worldVertices()

        face = mesh.faces[len(mesh.faces) // 2]
        v0, v1, v2 = world[face[0]], world[face[1]], world[face[2]]
        face_normal = np.cross(v1 - v0, v2 - v0)
        face_normal = face_normal / np.linalg.norm(face_normal)

        np.testing.assert_allclose(face_normal, selection.normal, atol=1e-6, err_msg=axis)


def test_non_finite_voxels_do_not_poison_the_whole_slice():
    # Regression: np.percentile over a slice containing NaN returns NaN for both
    # bounds, and the `hi <= lo` guard can't catch it (NaN comparisons are always
    # False), so one NaN voxel turned the entire intensity attribute into NaN.
    array = np.zeros((2, 2, 2))
    array[:, :, 0] = [[1.0, np.nan], [3.0, 4.0]]
    volume = NiftiVolume(array, np.eye(4))

    mesh = volume.getSlice(volume.selectionFor("axial", 0))
    intensities, _ = mesh.float_attributes["intensity"]

    assert not np.any(np.isnan(intensities))
    assert intensities.min() >= 0.0 and intensities.max() <= 1.0
    # The non-finite voxel is pinned to the bottom of the ramp.
    assert intensities[1] == 0.0


def test_all_non_finite_slice_degrades_gracefully():
    volume = NiftiVolume(np.full((2, 2, 2), np.nan), np.eye(4))

    intensities, _ = volume.getSlice(volume.selectionFor("axial", 0)).float_attributes["intensity"]

    assert not np.any(np.isnan(intensities))
    np.testing.assert_allclose(intensities, 0.0)


def test_infinite_voxels_are_excluded_from_the_window():
    array = np.zeros((2, 2, 2))
    array[:, :, 0] = [[1.0, np.inf], [3.0, 4.0]]
    volume = NiftiVolume(array, np.eye(4))

    intensities, _ = volume.getSlice(volume.selectionFor("axial", 0)).float_attributes["intensity"]

    assert np.all(np.isfinite(intensities))


def test_numpy_color_ramp_is_accepted():
    # Regression: `colors or _GRAYSCALE_RAMP` raised "truth value of an array is
    # ambiguous" for a numpy ramp -- the natural thing to pass, e.g. from a
    # matplotlib colormap.
    volume = NiftiVolume(np.random.rand(4, 4, 4), np.eye(4))
    ramp = np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]])

    mesh = volume.getSlice(volume.selectionFor("axial", 1), colors=ramp)

    assert mesh.material.color_attribute == "intensity"


@pytest.mark.parametrize("index", [-1, 4, 10**6])
def test_selection_for_rejects_out_of_range_index(index):
    # Regression: indices were never bounds-checked. Negative ones were wrapped by
    # np.take, so the plane was positioned outside the volume but textured with the
    # far side's voxels -- silently wrong geometry with no error.
    volume = NiftiVolume(np.zeros((4, 4, 4)), np.eye(4))
    with pytest.raises(IndexError):
        volume.selectionFor("axial", index)


def test_selection_for_accepts_both_ends_of_the_valid_range():
    volume = NiftiVolume(np.zeros((4, 5, 6)), np.eye(4))
    assert volume.selectionFor("axial", 0).index == 0
    assert volume.selectionFor("axial", 5).index == 5


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
