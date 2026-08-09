"""
Scene: the single public entry point for rendering with Py3DRF.

Users/developers never instantiate or import a backend class (SceneBlender,
ScenePlotly, ScenePolyscope, ...) directly. They only ever use Scene, which:

  1. Picks and lazily imports the requested backend (only that one -- see
     Py3DRF.backends).
  2. Wraps the backend instance and forwards every method call to it.

If the chosen backend doesn't implement a given method, calling it raises
NotSupportedByBackendError (see Py3DRF.backends.base) -- Scene never fakes,
no-ops, or approximates a feature a backend doesn't actually have.
"""

from .backends import load_backend_class
from .backends.base import SceneBackend

# Every public method on SceneBackend becomes a forwarding method on Scene.
# This is generated from SceneBackend itself (rather than hand-copied) so the
# facade's surface can never silently drift out of sync with the contract
# backends are written against.
_FACADE_METHODS = [
    name for name, value in vars(SceneBackend).items()
    if not name.startswith("_") and callable(value)
]


def _make_delegate(method_name):
    def delegate(self, *args, **kwargs):
        return getattr(self._impl, method_name)(*args, **kwargs)

    delegate.__name__ = method_name
    delegate.__doc__ = getattr(SceneBackend, method_name).__doc__
    return delegate


class Scene:
    """
    Facade over a single backend, chosen once at construction time.

    Example:
        scene = Scene(backend="blender", resolution=(1080, 1080), engine="CYCLES")
        scene.addObject(mesh)
        scene.addCamera(camera)
        scene.renderToFile("output.png")

    Swapping to another backend is a one-argument change:
        scene = Scene(backend="plotly", resolution=(1080, 1080))

    Any keyword arguments beyond `backend` and `allow_backend_conflict` are
    forwarded to the backend's own constructor, so they vary per backend (see
    each backend's SceneBackend subclass for what it accepts).

    Some backend pairs (e.g. "blender" and "usd") bundle third-party packages
    known to conflict when both are imported into the same process. Loading
    one after the other's package is already imported raises
    BackendConflictError; pass allow_backend_conflict=True to proceed anyway.
    """

    def __init__(self, backend="blender", allow_backend_conflict=False, **kwargs):
        backend_cls = load_backend_class(backend, allow_backend_conflict=allow_backend_conflict)
        self._impl = backend_cls(**kwargs)
        self.backend_name = backend

    def __repr__(self):
        return f"Scene(backend={self.backend_name!r})"


for _name in _FACADE_METHODS:
    setattr(Scene, _name, _make_delegate(_name))

del _name
