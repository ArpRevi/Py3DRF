import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from Py3DRF import Camera, Location
from Py3DRF.io.nifti import NiftiVolume
from Py3DRF.mri import browser
from Py3DRF.mri.browser import pick_slice_index, pick_slices, pick_camera_angle


def test_pick_slice_index_builds_and_returns_selection(monkeypatch):
    # Agg's plt.show() doesn't block, but stub it anyway so this test can't
    # hang under a different backend picked up from the environment.
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    volume = NiftiVolume(np.random.rand(4, 5, 6), np.eye(4))

    selection = pick_slice_index(volume, "axial", initial=2)

    assert selection.axis == "axial"
    assert selection.index == 2


def test_pick_slice_index_defaults_to_middle_slice(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    volume = NiftiVolume(np.random.rand(4, 5, 7), np.eye(4))

    selection = pick_slice_index(volume, "sagittal")

    assert selection.index == 1  # (4 - 1) // 2


def test_pick_slice_index_rejects_unknown_axis():
    volume = NiftiVolume(np.zeros((2, 2, 2)), np.eye(4))
    with pytest.raises(ValueError):
        pick_slice_index(volume, "diagonal")


def test_pick_slices_builds_one_window_and_returns_all_selections(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    volume = NiftiVolume(np.random.rand(6, 7, 8), np.eye(4))

    selections = pick_slices(volume, axes=("sagittal", "coronal", "axial"))

    assert [s.axis for s in selections] == ["sagittal", "coronal", "axial"]
    assert [s.index for s in selections] == [2, 3, 3]  # (shape - 1) // 2 per axis


def test_pick_slices_respects_initial_and_subset_of_axes(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    volume = NiftiVolume(np.random.rand(6, 7, 8), np.eye(4))

    selections = pick_slices(volume, axes=("axial", "sagittal"), initial={"axial": 5})

    assert [s.axis for s in selections] == ["axial", "sagittal"]
    assert selections[0].index == 5
    assert selections[1].index == 2  # defaulted to the middle slice


def test_pick_slices_rejects_unknown_axis():
    volume = NiftiVolume(np.zeros((2, 2, 2)), np.eye(4))
    with pytest.raises(ValueError):
        pick_slices(volume, axes=("axial", "diagonal"))


@pytest.mark.parametrize("initial", [-1, 99])
def test_pick_slice_index_rejects_out_of_range_initial(monkeypatch, initial):
    # Regression: `initial` reached np.take before selectionFor ever saw it, so an
    # out-of-range value surfaced as a raw IndexError from imshow, and a negative
    # one silently displayed the far side of the volume while the slider -- clamped
    # at its own minimum -- disagreed with it.
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    volume = NiftiVolume(np.random.rand(4, 5, 6), np.eye(4))

    with pytest.raises(IndexError):
        pick_slice_index(volume, "axial", initial=initial)


def test_pick_slices_rejects_out_of_range_initial(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    volume = NiftiVolume(np.random.rand(4, 5, 6), np.eye(4))

    with pytest.raises(IndexError):
        pick_slices(volume, axes=("axial",), initial={"axial": 99})


def test_pick_slices_keeps_repeated_axes_independent(monkeypatch):
    # Regression: getters were keyed by axis name, so a repeated axis let the second
    # subplot's getter overwrite the first -- returning one slider's value twice and
    # silently discarding the other.
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    volume = NiftiVolume(np.random.rand(4, 5, 6), np.eye(4))

    selections = pick_slices(volume, axes=("axial", "axial"), initial={"axial": 2})

    assert [s.axis for s in selections] == ["axial", "axial"]
    assert len(selections) == 2


def test_pick_slices_checkbox_defaults_to_checked(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    volume = NiftiVolume(np.random.rand(6, 7, 8), np.eye(4))

    _, get_visible = browser._add_slice_subplot(
        *plt.subplots(), volume, "axial", 3, [0.2, 0.05, 0.6, 0.03], [0.2, 0.15, 0.6, 0.08]
    )

    assert get_visible() is True


def test_pick_slice_index_has_no_checkbox_and_is_always_visible(monkeypatch):
    # pick_slice_index doesn't pass checkbox_rect -- it always returns a single
    # SliceSelection unconditionally, so visibility must always read True.
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    volume = NiftiVolume(np.random.rand(6, 7, 8), np.eye(4))

    _, get_visible = browser._add_slice_subplot(*plt.subplots(), volume, "axial", 3, [0.2, 0.05, 0.6, 0.03])

    assert get_visible() is True


def test_pick_slices_unchecked_axis_is_omitted_from_the_result(monkeypatch):
    # Simulates the user unchecking "coronal"'s checkbox before closing the window:
    # wraps _add_slice_subplot so the coronal subplot's real checkbox is toggled off
    # via its own public set_active() before pick_slices reads it back.
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    original = browser._add_slice_subplot

    def toggle_coronal_off(fig, ax, volume, axis, index, slider_rect, checkbox_rect=None):
        get_index, get_visible = original(fig, ax, volume, axis, index, slider_rect, checkbox_rect)
        if axis == "coronal":
            ax._checkbox.set_active(0)
        return get_index, get_visible

    monkeypatch.setattr(browser, "_add_slice_subplot", toggle_coronal_off)

    volume = NiftiVolume(np.random.rand(6, 7, 8), np.eye(4))
    selections = browser.pick_slices(volume, axes=("sagittal", "coronal", "axial"))

    assert [s.axis for s in selections] == ["sagittal", "axial"]


def test_pick_slices_all_unchecked_returns_empty_list(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    original = browser._add_slice_subplot

    def uncheck_all(fig, ax, volume, axis, index, slider_rect, checkbox_rect=None):
        get_index, get_visible = original(fig, ax, volume, axis, index, slider_rect, checkbox_rect)
        ax._checkbox.set_active(0)
        return get_index, get_visible

    monkeypatch.setattr(browser, "_add_slice_subplot", uncheck_all)

    volume = NiftiVolume(np.random.rand(6, 7, 8), np.eye(4))
    selections = browser.pick_slices(volume, axes=("sagittal", "coronal", "axial"))

    assert selections == []


def test_pick_camera_angle_builds_and_returns_camera(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    camera = pick_camera_angle(Location(0, 0, 0), distance=5)

    assert isinstance(camera, Camera)


def test_pick_camera_angle_matches_focus_on_point_at_default_azimuth(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    point = Location(1, 2, 3)
    picked = pick_camera_angle(point, elevation=0.3, distance=7, initial_degrees=45)

    expected = Camera()
    expected.focusOnPoint(point, azimuth=np.deg2rad(45), elevation=0.3, distance=7)

    np.testing.assert_allclose(picked.location.to_array(), expected.location.to_array())
    np.testing.assert_allclose(picked.rotation.to_array(), expected.rotation.to_array())
    np.testing.assert_allclose(picked.target.to_array(), expected.target.to_array())


def test_pick_camera_angle_non_default_initial_degrees(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    point = Location(0, 0, 0)
    picked = pick_camera_angle(point, elevation=0.1, distance=10, initial_degrees=200)

    expected = Camera()
    expected.focusOnPoint(point, azimuth=np.deg2rad(200), elevation=0.1, distance=10)

    np.testing.assert_allclose(picked.location.to_array(), expected.location.to_array())
