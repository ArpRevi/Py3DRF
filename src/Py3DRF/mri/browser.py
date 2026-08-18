"""
Interactive visual picker for MRI slices.

This is the only module in Py3DRF.mri that imports matplotlib. pick_slice_index()
opens a matplotlib window with a slider to scrub through a NiftiVolume along a
chosen axis; pick_slices() does the same for several axes at once, as subplots
in a single window. Both block until their window is closed -- they require a
GUI-capable matplotlib backend (e.g. TkAgg) and are not usable headless/on CI.

Py3DRF.mri is never imported by Py3DRF's top-level package, matching the
project's rule that importing Py3DRF never pulls in an optional third-party
dependency.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from ..io.nifti import AXIS_TO_DIM, NiftiVolume, SliceSelection


def _validate_axis(axis):
    if axis not in AXIS_TO_DIM:
        raise ValueError(f"Unknown axis {axis!r}. Must be one of {list(AXIS_TO_DIM)}.")


def _add_slice_subplot(fig, ax, volume, axis, index, slider_rect):
    """
    Wire up one axis' imshow + Slider pair (used by both pick_slice_index and
    pick_slices), returning a 0-argument getter for the slider's current value.

    :param fig: Figure the slider's Axes will be added to.
    :param ax: Axes the slice image is drawn into.
    :param volume: NiftiVolume being browsed.
    :param axis: "sagittal", "coronal", or "axial".
    :param index: Initial voxel index to display.
    :param slider_rect: [left, bottom, width, height] passed to fig.add_axes()
        for the slider itself, in figure-fraction coordinates.
    :return: A zero-argument callable returning the slider's current int value.

    """
    dim = AXIS_TO_DIM[axis]
    max_index = volume.array.shape[dim] - 1

    image_artist = ax.imshow(np.take(volume.array, index, axis=dim).T, cmap="gray", origin="lower")
    ax.set_title(axis)
    ax.set_axis_off()

    slider = Slider(fig.add_axes(slider_rect), "Index", 0, max_index, valinit=index, valstep=1)
    state = {"index": index}

    def _on_change(value):
        state["index"] = int(value)
        image_artist.set_data(np.take(volume.array, state["index"], axis=dim).T)
        fig.canvas.draw_idle()

    slider.on_changed(_on_change)
    # Keep a reference alive on the Axes itself: Slider stops responding to
    # input if its only reference is the local variable above and that gets
    # garbage-collected before the figure is closed.
    ax._slider = slider

    return lambda: state["index"]


def pick_slice_index(volume: NiftiVolume, axis, initial=None) -> SliceSelection:
    """
    Open an interactive slider window to choose a voxel index along `axis`,
    blocking until the window is closed.

    :param volume: NiftiVolume to browse.
    :param axis: "sagittal", "coronal", or "axial".
    :param initial: Initial voxel index to display. Defaults to the middle slice.
    :return: SliceSelection describing the confirmed slice (the slider's value
        when the window was closed), including its world-space origin/normal.

    """
    _validate_axis(axis)
    dim = AXIS_TO_DIM[axis]
    max_index = volume.array.shape[dim] - 1
    index = max_index // 2 if initial is None else int(initial)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    ax.set_title(f"{axis} slice -- close window to confirm")
    get_index = _add_slice_subplot(fig, ax, volume, axis, index, [0.2, 0.05, 0.6, 0.03])

    plt.show()

    return volume.selectionFor(axis, get_index())


def pick_slices(volume: NiftiVolume, axes=("sagittal", "coronal", "axial"), initial=None) -> list:
    """
    Open a single window with one image + slider per axis in `axes`, side by
    side, blocking until the window is closed -- so all of them are confirmed
    together in one interaction instead of one pick_slice_index() call (and
    window) per axis.

    :param volume: NiftiVolume to browse.
    :param axes: Axes to pick, in display and return order. Defaults to all
        three canonical axes.
    :param initial: Optional {axis: index} dict of initial voxel indices. Any
        axis not present defaults to the middle slice.
    :return: List of SliceSelection, one per axis in `axes`, in the same order.

    """
    for axis in axes:
        _validate_axis(axis)
    initial = initial or {}

    n = len(axes)
    fig, subplot_axes = plt.subplots(1, n, figsize=(4 * n, 4.5))
    if n == 1:
        subplot_axes = [subplot_axes]
    fig.suptitle("close window to confirm all slices")
    plt.subplots_adjust(bottom=0.25, wspace=0.3)

    getters = {}
    slider_width = 0.8 / n
    for i, axis in enumerate(axes):
        dim = AXIS_TO_DIM[axis]
        max_index = volume.array.shape[dim] - 1
        index = max_index // 2 if axis not in initial else int(initial[axis])

        slider_rect = [0.1 + i * slider_width, 0.08, slider_width - 0.05, 0.03]
        getters[axis] = _add_slice_subplot(fig, subplot_axes[i], volume, axis, index, slider_rect)

    plt.show()

    return [volume.selectionFor(axis, getters[axis]()) for axis in axes]
