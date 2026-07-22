"""
Backend registry for Py3DRF.

Importing Py3DRF.backends (or Py3DRF itself) never imports any backend's
third-party dependency (bpy, plotly, polyscope, ...). Only naming a backend
via load_backend_class() triggers importing that one backend's module -- the
others are left completely untouched. This means:

  * A machine with only `plotly` installed can `import Py3DRF` and use
    `Scene(backend="plotly")` without ever attempting to import bpy.
  * Adding a new backend to _BACKENDS below does not require any other
    backend, or Py3DRF.core, to know about it.
"""

import importlib

# name -> (module path, class name). The class name follows the
# "Scene<BackendName>" convention, e.g. SceneBlender, ScenePlotly.
_BACKENDS = {
    "blender": ("Py3DRF.backends.blender.scene", "SceneBlender"),
    "plotly": ("Py3DRF.backends.plotly.scene", "ScenePlotly"),
    "polyscope": ("Py3DRF.backends.polyscope.scene", "ScenePolyscope"),
}


def available_backends():
    """
    :return: Names of all backends Py3DRF knows about, regardless of whether
        their dependencies are actually installed on this machine.
    """
    return list(_BACKENDS)


def load_backend_class(name):
    """
    Import and return the concrete SceneBackend subclass for a given backend
    name, without importing or touching any other backend.

    :param name: Backend name, e.g. "blender", "plotly", "polyscope".
    :return: The backend's SceneBackend subclass (not yet instantiated).
    """
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown backend {name!r}. Available backends: {available_backends()}."
        )

    module_path, class_name = _BACKENDS[name]
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"Backend {name!r} could not be loaded, most likely because its "
            f"dependency isn't installed. Try: pip install Py3DRF[{name}]"
        ) from exc

    return getattr(module, class_name)
