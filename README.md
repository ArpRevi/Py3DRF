# Python 3D Rendering Facade

The goal of this project is to create a facade that allows the use of various back-end
software for the purpose of 3D scene description and rendering.

Currently supported back-ends: **Blender** (via `bpy`), **Plotly**, **Polyscope**,
**PyVista**, and **USD** (export-only, via `usd-core`).

Also included: **MRI (NIfTI) support** -- load a `.nii`/`.nii.gz` volume, interactively
pick slices and a camera angle, and render the result through any backend above.

## Usage

`Scene` is the only class you need. Pick a backend by name; everything else about your
script stays the same.

```python
from Py3DRF import Scene, Mesh, Location, Rotation, Scale

vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
faces = [(0, 1, 2)]

scene = Scene(backend="blender", resolution=(1080, 1080), engine="CYCLES")
# scene = Scene(backend="plotly", resolution=(1080, 1080))
# scene = Scene(backend="polyscope", resolution=(1080, 1080))
# scene = Scene(backend="pyvista", resolution=(1080, 1080))
# scene = Scene(backend="usd")   # export-only -- see below

mesh = Mesh("Triangle", vertices, [], faces)
scene.addObject(mesh)
scene.addCamera(mesh.getCamera())
scene.renderToFile("output.png")
```

If a method isn't supported by the chosen backend (e.g. `setSamples` on Plotly, which has
no sampled renderer), calling it raises `Py3DRF.NotSupportedByBackendError` rather than
silently doing nothing or approximating the feature. Each backend's module docstring
lists exactly what it does and doesn't support, and why.

### USD is export-only

USD is a scene-description format, not a renderer, so `SceneUSD` has no `renderToFile`
-- it implements `exportToFile` instead, which writes the scene as ASCII `.usda`:

```python
scene = Scene(backend="usd")
scene.addObject(mesh)
scene.exportToFile("output.usda")   # only .usda is accepted
```

### Backend conflicts

`bpy` (Blender) and `pxr` (USD) each bundle their own compiled USD libraries, which can
crash or misbehave if both end up imported in the same process. If you load one after the
other is already imported, `Scene(...)` raises `Py3DRF.BackendConflictError`. This is why
the `all` extra (below) installs both -- using them together in one script needs an
explicit opt-in:

```python
Scene(backend="usd", allow_backend_conflict=True)   # warns instead of raising
```

For demonstration purposes, see the scripts in `examples/` -- one per backend, plus an
MRI example (see below).

## Installing backends

Each backend is an optional extra, so you only install what you use:

```bash
pip install "Py3DRF[blender]"     # requires bpy
pip install "Py3DRF[plotly]"      # requires plotly (+ kaleido for static image export)
pip install "Py3DRF[polyscope]"   # requires polyscope
pip install "Py3DRF[pyvista]"     # requires pyvista
pip install "Py3DRF[usd]"         # requires usd-core
pip install "Py3DRF[mri]"         # requires nibabel + matplotlib (see MRI support below)
pip install "Py3DRF[all]"         # every backend + mri (see the conflict note above)
```

`import Py3DRF` never imports any backend's dependency. Only calling
`Scene(backend="...")` imports that one backend's third-party package -- the others are
left completely untouched, so a machine with only `plotly` installed can use the Plotly
backend without ever attempting to import `bpy`. The same rule applies to `Py3DRF.io` and
`Py3DRF.mri`: neither is imported by `import Py3DRF`, so `nibabel`/`matplotlib` are only
ever needed if you import them yourself.

## MRI (NIfTI) support

`Py3DRF.io.load_nifti` loads a `.nii`/`.nii.gz` volume into a `NiftiVolume`, reoriented to
canonical RAS+ so `"sagittal"`/`"coronal"`/`"axial"` reliably mean what they claim to.
`NiftiVolume.getSlice()` turns a chosen slice into a plain `Mesh`, positioned via the
volume's real affine transform -- so slices from the same volume land in consistent world
coordinates, and their spatial intersection is simply visible when several are rendered
together.

`Py3DRF.mri` adds interactive, matplotlib-based pickers on top (require a display, not
usable headless/on CI):

```python
from Py3DRF import Scene
from Py3DRF.io import load_nifti
from Py3DRF.mri import pick_slices, pick_camera_angle

volume = load_nifti("brain.nii.gz")

# one window, one slider per axis, a "show" checkbox to drop any of them
selections = pick_slices(volume, axes=("sagittal", "coronal", "axial"))

scene = Scene(backend="pyvista")
for selection in selections:
    scene.addObject(volume.getSlice(selection))
scene.addCamera(pick_camera_angle(point=selections[0].origin, distance=300))
scene.renderToFile("mri_render.png")
```

See `examples/render_mri.py` for a full script. `pick_slice_index` is also available for
picking a single axis at a time.

## Architecture

```
src/Py3DRF/
  core/               backend-agnostic data model: Camera, Mesh, SunLight, Material,
                       PointCloudSettings, WireFrameSettings, Location/Rotation/Scale.
                       Pure Python/numpy, no rendering-backend imports anywhere in
                       this subpackage.
  backends/
    base.py           SceneBackend: the full method surface a backend can
                       implement. Unimplemented methods raise
                       NotSupportedByBackendError by default.
    __init__.py        registry mapping backend name -> module, loaded lazily,
                       plus the cross-backend conflict guard.
    blender/scene.py   SceneBlender    (imports bpy)
    plotly/scene.py     ScenePlotly    (imports plotly)
    polyscope/scene.py  ScenePolyscope (imports polyscope)
    pyvista/scene.py    ScenePyVista   (imports pyvista)
    usd/scene.py        SceneUSD       (imports pxr; export-only)
  io/
    nifti.py            NiftiVolume, SliceSelection, load_nifti (imports nibabel,
                        lazily, only inside load_nifti)
  mri/
    browser.py          pick_slice_index, pick_slices, pick_camera_angle
                        (imports matplotlib)
  scene.py             Scene: the public facade. Wraps whichever backend was
                       requested and forwards every call to it.
```

Rules this structure enforces:

- **Backends never import each other**, and never import anything the user is meant to
  touch directly -- only `Py3DRF.core` and their own third-party dependency.
- **The user only ever imports `Scene`** (plus the `core` data classes to build a scene).
  There is no `SceneBlender`/`ScenePlotly`/`ScenePolyscope`/`ScenePyVista`/`SceneUSD` in
  the public API.
- **Nothing is faked.** A backend implements the subset of `SceneBackend` it can
  genuinely support; anything else raises `NotSupportedByBackendError` with a clear
  message naming the backend and the missing feature.
- **Loading is on-demand.** Naming a backend imports only that backend's dependency,
  at the moment it's requested -- not at package import time, and not for backends you
  never asked for. `Py3DRF.io` and `Py3DRF.mri` follow the same rule.
- **Known conflicts are caught, not silently risked.** Loading a backend whose
  dependency is known to clash with one already imported in the process raises
  `BackendConflictError` unless explicitly overridden.
