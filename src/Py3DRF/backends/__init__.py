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
import sys
import warnings

# name -> (module path, class name). The class name follows the
# "Scene<BackendName>" convention, e.g. SceneBlender, ScenePlotly.
_BACKENDS = {
    "blender": ("Py3DRF.backends.blender.scene", "SceneBlender"),
    "plotly": ("Py3DRF.backends.plotly.scene", "ScenePlotly"),
    "polyscope": ("Py3DRF.backends.polyscope.scene", "ScenePolyscope"),
    "usd": ("Py3DRF.backends.usd.scene", "SceneUSD"),
}

# backend name -> top-level module name that, if already imported in this
# process, is known to conflict with this backend's own package. bpy bundles
# its own compiled USD libraries; pxr (usd-core) can clash with them if both
# end up loaded together in the same process, regardless of which was
# imported first or through what path (Py3DRF or directly by the caller).
_CONFLICTS = {
    "blender": "pxr",
    "usd": "bpy",
}


class BackendConflictError(RuntimeError):
    """
    Raised when loading a backend would import a third-party package known to
    conflict with another backend's package already loaded in this process.
    Pass allow_backend_conflict=True (to load_backend_class or Scene(...)) to
    load it anyway -- doing so still emits a BackendConflictWarning.
    """


class BackendConflictWarning(RuntimeWarning):
    """Emitted when a known backend conflict is bypassed via allow_backend_conflict=True."""


def available_backends():
    """
    :return: Names of all backends Py3DRF knows about, regardless of whether
        their dependencies are actually installed on this machine.
    """
    return list(_BACKENDS)


def _check_conflicts(name, allow_backend_conflict):
    conflicting_module = _CONFLICTS.get(name)
    if conflicting_module is None or conflicting_module not in sys.modules:
        return

    message = (
        f"Loading the {name!r} backend imports a package known to conflict with "
        f"{conflicting_module!r}, which this process has already imported "
        f"(possibly via another backend). bpy and pxr each bundle their own USD "
        f"libraries, and loading both in one process can cause crashes or subtly "
        f"broken behavior."
    )
    if not allow_backend_conflict:
        raise BackendConflictError(
            message + " Pass allow_backend_conflict=True to load it anyway, at your own risk."
        )
    warnings.warn(message, BackendConflictWarning, stacklevel=3)


def load_backend_class(name, allow_backend_conflict=False):
    """
    Import and return the concrete SceneBackend subclass for a given backend
    name, without importing or touching any other backend.

    :param name: Backend name, e.g. "blender", "plotly", "polyscope", "usd".
    :param allow_backend_conflict: If False (default), raises BackendConflictError
        instead of importing when a known-conflicting backend's package is already
        imported in this process (see _CONFLICTS). If True, imports anyway and
        emits a BackendConflictWarning instead.
    :return: The backend's SceneBackend subclass (not yet instantiated).
    """
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown backend {name!r}. Available backends: {available_backends()}."
        )

    _check_conflicts(name, allow_backend_conflict)

    module_path, class_name = _BACKENDS[name]
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"Backend {name!r} could not be loaded, most likely because its "
            f"dependency isn't installed. Try: pip install Py3DRF[{name}]"
        ) from exc

    return getattr(module, class_name)
