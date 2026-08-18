import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from Py3DRF.io.nifti import NiftiVolume
from Py3DRF.mri.browser import pick_slice_index, pick_slices


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
