# Python 3D Rendering Facade

The goal of this project is to create a facade that allows the use of various back-end
software for the purpose of 3D scene rendering.

Currently supported back-ends: **Blender** (via `bpy`), **Plotly**, **Polyscope**.

Another goal is to make this project compatible with USD.

## Usage

`Scene` is the only class you need. Pick a backend by name; everything else about your
script stays the same.

```python
from Py3DRF import Scene, Mesh, Camera

scene = Scene(backend="blender", resolution=(1080, 1080), engine="CYCLES")
# scene = Scene(backend="plotly", resolution=(1080, 1080))
# scene = Scene(backend="polyscope", resolution=(1080, 1080))

mesh = Mesh("Suzanne", vertices, edges, faces)
scene.addObject(mesh)
scene.addCamera(mesh.getCamera())
scene.renderToFile("output.png")
```

If a method isn't supported by the chosen backend (e.g. `setSamples` on Plotly, which has
no sampled renderer), calling it raises `Py3DRF.NotSupportedByBackendError` rather than
silently doing nothing or approximating the feature.

For demonstration purposes, run `examples/render_blender.py`, `examples/render_plotly.py` or
`examples/render_polyscope.py` for the same scene rendered through different backends.

## Installing backends

Each backend is an optional extra, so you only install what you use:

```bash
pip install "Py3DRF[blender]"     # requires bpy
pip install "Py3DRF[plotly]"      # requires plotly (+ kaleido for static image export)
pip install "Py3DRF[polyscope]"   # requires polyscope
pip install "Py3DRF[all]"         # all of the above
```

`import Py3DRF` never imports any backend's dependency. Only calling
`Scene(backend="...")` imports that one backend's third-party package -- the others are
left completely untouched, so a machine with only `plotly` installed can use the Plotly
backend without ever attempting to import `bpy`.

## Architecture

```
src/Py3DRF/
  core/               backend-agnostic data model: Camera, Mesh, SunLight,
                       Material, PointCloudSettings. Pure Python/numpy, no
                       rendering-backend imports anywhere in this subpackage.
  backends/
    base.py           SceneBackend: the full method surface a backend can
                       implement. Unimplemented methods raise
                       NotSupportedByBackendError by default.
    __init__.py        registry mapping backend name -> module, loaded lazily.
    blender/scene.py   SceneBlender  (imports bpy)
    plotly/scene.py     ScenePlotly   (imports plotly)
    polyscope/scene.py  ScenePolyscope (imports polyscope)
  scene.py             Scene: the public facade. Wraps whichever backend was
                       requested and forwards every call to it.
```

Rules this structure enforces:

- **Backends never import each other**, and never import anything the user is meant to
  touch directly -- only `Py3DRF.core` and their own third-party dependency.
- **The user only ever imports `Scene`** (plus the `core` data classes to build a scene).
  There is no `SceneBlender`/`ScenePlotly`/`ScenePolyscope` in the public API.
- **Nothing is faked.** A backend implements the subset of `SceneBackend` it can
  genuinely support; anything else raises `NotSupportedByBackendError` with a clear
  message naming the backend and the missing feature.
- **Loading is on-demand.** Naming a backend imports only that backend's dependency,
  at the moment it's requested -- not at package import time, and not for backends you
  never asked for.
