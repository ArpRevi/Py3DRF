"""
Interactive visual pickers for MRI slices and camera placement.

This is the only module in Py3DRF.mri that imports matplotlib. pick_slice_index()
opens a matplotlib window with a slider to scrub through a NiftiVolume along a
chosen axis; pick_slices() does the same for several axes at once, as subplots
in a single window; pick_camera_angle() opens a slider (with a top-down
schematic) to choose a camera azimuth orbiting a point. All three block until
their window is closed -- they require a GUI-capable matplotlib backend (e.g.
TkAgg) and are not usable headless/on CI.

Py3DRF.mri is never imported by Py3DRF's top-level package, matching the
project's rule that importing Py3DRF never pulls in an optional third-party
dependency.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from ..core.camera import Camera
from ..core.types import Location
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


def pick_camera_angle(point: Location, elevation=np.pi / 9, distance=3, initial_degrees=45) -> Camera:
    """
    Open an interactive slider (0-360 deg) to choose a camera azimuth orbiting
    `point`, with a top-down schematic showing the current direction, blocking
    until the window is closed.

    :param point: World-space point the camera orbits around and looks at.
    :param elevation: Elevation angle in radians (see Camera.focusOnPoint) --
        fixed, not part of what's picked interactively here.
    :param distance: Distance from `point` (see Camera.focusOnPoint) -- fixed.
    :param initial_degrees: Initial slider value in degrees.
    :return: A Camera already built via Camera().focusOnPoint(point, azimuth,
        elevation, distance) at the confirmed azimuth.

    """
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    ax.set_aspect("equal")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_title("choose camera azimuth -- close window to confirm")
    ax.set_axis_off()

    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), linestyle="--", color="lightgray")
    ax.plot(0, 0, marker="o", color="black")
    ax.annotate("target", (0, 0), textcoords="offset points", xytext=(6, 6))

    az0 = np.deg2rad(initial_degrees)
    (camera_dot,) = ax.plot([np.cos(az0)], [np.sin(az0)], marker="o", color="tab:red", markersize=10)
    arrow = ax.annotate("", xy=(np.cos(az0), np.sin(az0)), xytext=(0, 0),
                         arrowprops=dict(arrowstyle="->", color="tab:red"))

    slider = Slider(fig.add_axes([0.2, 0.08, 0.6, 0.03]), "Azimuth (deg)", 0, 360,
                     valinit=initial_degrees, valstep=1)
    # Keep a reference alive on the Axes itself: Slider stops responding to
    # input if its only reference is the local variable above and that gets
    # garbage-collected before the figure is closed (see _add_slice_subplot).
    ax._slider = slider

    state = {"degrees": initial_degrees}

    def _on_change(value):
        state["degrees"] = float(value)
        az = np.deg2rad(state["degrees"])
        x, y = np.cos(az), np.sin(az)
        camera_dot.set_data([x], [y])
        arrow.xy = (x, y)
        fig.canvas.draw_idle()

    slider.on_changed(_on_change)
    plt.show()

    camera = Camera()
    camera.focusOnPoint(point, azimuth=np.deg2rad(state["degrees"]), elevation=elevation, distance=distance)
    return camera
