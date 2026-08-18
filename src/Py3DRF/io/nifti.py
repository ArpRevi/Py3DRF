"""
NIfTI (.nii/.nii.gz) loading and MRI slice-to-Mesh geometry.

This module has no hard dependency on nibabel at import time -- only calling
load_nifti() does (nibabel is imported lazily, inside that one function), so
NiftiVolume can be built and used directly (e.g. in tests, or from data loaded
some other way) without nibabel installed. Py3DRF.io itself is never imported
by Py3DRF's top-level package, matching the project's rule that importing
Py3DRF never pulls in an optional third-party dependency.

A slice is described by a SliceSelection (axis, voxel index, and its
world-space origin/normal, derived from the volume's affine) and turned into a
plain Py3DRF.core.Mesh by NiftiVolume.getSlice() -- reusing Mesh and Material's
existing float-attribute-as-color-ramp mechanism (Material.
setFloatAttributeAsColor) to display intensity, rather than adding any new
core/backend features. Base color, not emission, is used deliberately: emission
is a Blender-only concept that ScenePlotly/ScenePolyscope both ignore, so an
emission-driven slice would render on Blender but come out blank everywhere
else (see NiftiVolume.getSlice's docstring). Because each slice's geometry is
positioned via the volume's real affine transform, multiple slices added to
the same Scene land in consistent, correctly-spaced world coordinates -- so
their spatial intersection is simply visible when rendered together, with no
separate intersection-line computation needed.

Known cross-backend caveat: Blender and Plotly reproduce a custom color ramp
exactly; Polyscope substitutes its own built-in colormap instead (see
Py3DRF.backends.polyscope.scene), so slice grayscale/windowing renders
correctly on Blender/Plotly but only approximately on Polyscope.
"""

from dataclasses import dataclass

import numpy as np

from ..core.mesh import Mesh

# axis name -> the voxel array axis that stays fixed for that slice orientation,
# assuming the array is in canonical RAS+ order (axis 0 = left-right, axis 1 =
# posterior-anterior, axis 2 = inferior-superior), as produced by load_nifti().
AXIS_TO_DIM = {"sagittal": 0, "coronal": 1, "axial": 2}

# axis name -> the two voxel array axes that vary across that slice (grid
# rows/cols), in this order. Removing the fixed axis always preserves the
# relative order of the other two, so this is just AXIS_TO_DIM's complement.
_FREE_DIMS = {"axial": (0, 1), "coronal": (0, 2), "sagittal": (1, 2)}

_GRAYSCALE_RAMP = [(0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0)]


def _validate_axis(axis):
    if axis not in AXIS_TO_DIM:
        raise ValueError(f"Unknown axis {axis!r}. Must be one of {list(AXIS_TO_DIM)}.")


@dataclass(frozen=True)
class SliceSelection:
    """
    Plain data describing one chosen MRI slice: which axis/index it is, and
    where it sits in world space. Returned by NiftiVolume.selectionFor() and by
    Py3DRF.mri.pick_slice_index() -- not tied to any session or backend.

    Note: origin/normal are numpy arrays, so comparing two SliceSelection
    instances with == will raise (numpy array comparison isn't a plain bool);
    compare individual fields instead.
    """

    axis: str
    index: int
    origin: np.ndarray
    normal: np.ndarray


class NiftiVolume:
    """
    A loaded NIfTI volume: a 3D voxel array plus the affine transform mapping
    voxel indices to world-space (RAS, mm) coordinates.
    """

    def __init__(self, array, affine, header=None):
        """
        :param array: 3D numpy array of voxel intensities, in canonical RAS+
            axis order (see load_nifti).
        :param affine: 4x4 voxel-index -> world (RAS, mm) transform matrix.
        :param header: Optional nibabel header, kept for reference only.

        """
        self.array = np.asarray(array)
        self.affine = np.asarray(affine, dtype=float)
        self.header = header

    def selectionFor(self, axis, index) -> SliceSelection:
        """
        Build the world-space description of a slice, from the volume's affine
        alone -- no image data is touched.

        :param axis: "sagittal", "coronal", or "axial".
        :param index: Voxel index along that axis.
        :return: SliceSelection with the plane's world-space origin (the
            transformed center of the slice) and unit normal.

        """
        _validate_axis(axis)
        dim = AXIS_TO_DIM[axis]

        voxel_center = np.array(self.array.shape, dtype=float) / 2.0
        voxel_center[dim] = index
        origin = (self.affine @ np.append(voxel_center, 1.0))[:3]

        axis_direction = np.zeros(3)
        axis_direction[dim] = 1.0
        normal = self.affine[:3, :3] @ axis_direction
        normal = normal / np.linalg.norm(normal)

        return SliceSelection(axis=axis, index=int(index), origin=origin, normal=normal)

    def getSlice(
        self,
        selection: SliceSelection,
        name=None,
        colors=None,
        colors_positions=None,
        window=None,
        percentile_clip=(1.0, 99.0),
        step=1,
    ) -> Mesh:
        """
        Build a Mesh for the 2D cross-section described by `selection`: a grid of
        quads, one vertex per voxel (optionally strided by `step`), positioned in
        world space via the volume's affine, with a per-vertex float "intensity"
        attribute driving the Material's base color through a color ramp. This
        uses Material.setFloatAttributeAsColor rather than
        setFloatAttributeAsEmissionColor because emission is a Blender-only
        concept -- ScenePlotly and ScenePolyscope both ignore it entirely and
        only ever look at the base color/color_attribute, so an emission-driven
        slice would render correctly on Blender but come out blank/flat-black
        everywhere else. The tradeoff is that the slice is an ordinarily-shaded
        surface, subject to each backend's own lighting model, rather than a
        flat "unlit image" look.

        :param selection: SliceSelection describing which slice to build (as
            returned by selectionFor() or Py3DRF.mri.pick_slice_index()).
        :param name: Name of the returned Mesh. Defaults to "{axis}_{index}".
        :param colors: Color ramp for the intensity attribute (see
            Material.setFloatAttributeAsColor). Defaults to a 2-stop
            black-to-white grayscale ramp.
        :param colors_positions: Optional stop positions for `colors`.
        :param window: Optional explicit (min, max) intensity range to clip to
            before normalizing to [0, 1]. Overrides percentile_clip.
        :param percentile_clip: (low, high) percentiles of the slice's own
            intensities to clip to before normalizing, used when `window` isn't
            given. Guards against a few outlier voxels wrecking contrast.
        :param step: Voxel stride for the grid (1 = full resolution). Increase
            to downsample large slices for performance.
        :return: The built Mesh.

        """
        _validate_axis(selection.axis)
        dim = AXIS_TO_DIM[selection.axis]
        free0, free1 = _FREE_DIMS[selection.axis]

        image = np.take(self.array, selection.index, axis=dim)[::step, ::step].astype(float)
        rows, cols = image.shape

        row_voxels = np.arange(0, self.array.shape[free0], step)[:rows]
        col_voxels = np.arange(0, self.array.shape[free1], step)[:cols]
        rr, cc = np.meshgrid(row_voxels, col_voxels, indexing="ij")

        voxel_coords = np.empty((rows, cols, 3))
        voxel_coords[..., free0] = rr
        voxel_coords[..., free1] = cc
        voxel_coords[..., dim] = selection.index

        homogeneous = np.concatenate(
            [voxel_coords.reshape(-1, 3), np.ones((rows * cols, 1))], axis=1
        )
        world_vertices = (self.affine @ homogeneous.T).T[:, :3]

        r = np.arange(rows - 1)
        c = np.arange(cols - 1)
        rg, cg = np.meshgrid(r, c, indexing="ij")
        idx00 = (rg * cols + cg).ravel()
        idx10 = ((rg + 1) * cols + cg).ravel()
        idx11 = ((rg + 1) * cols + (cg + 1)).ravel()
        idx01 = (rg * cols + (cg + 1)).ravel()
        faces = np.stack([idx00, idx10, idx11, idx01], axis=1)

        if window is not None:
            lo, hi = window
        else:
            lo, hi = np.percentile(image, percentile_clip)
        if hi <= lo:
            hi = lo + 1.0
        intensities = (np.clip(image, lo, hi) - lo) / (hi - lo)

        mesh = Mesh(
            name or f"{selection.axis}_{selection.index}",
            vertices=list(world_vertices),
            faces=list(faces),
        )
        mesh.addFloatAttribute(intensities.ravel(), "intensity", domain="POINT")
        mesh.material.setFloatAttributeAsColor(
            "intensity", colors=colors or _GRAYSCALE_RAMP, colors_positions=colors_positions
        )
        return mesh


def load_nifti(path) -> NiftiVolume:
    """
    Load a .nii/.nii.gz file into a NiftiVolume, reoriented to canonical RAS+
    axis order (axis 0 = left-right, axis 1 = posterior-anterior, axis 2 =
    inferior-superior). This reorientation is required, not optional: raw
    NIfTI files can store axes in any order, and without it "sagittal"/
    "coronal"/"axial" would not reliably mean what they claim to for every
    file.

    :param path: Path to the .nii/.nii.gz file.
    :return: The loaded NiftiVolume.

    """
    import nibabel as nib

    img = nib.as_closest_canonical(nib.load(str(path)))
    return NiftiVolume(np.asarray(img.dataobj), img.affine, img.header)
