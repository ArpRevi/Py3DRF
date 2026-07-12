"""
SceneBackend: the contract every Py3DRF backend is written against.

Design rules for this file (see project architecture discussion):

1. This is the *complete* method surface the Scene facade can expose. A
   concrete backend (SceneBlender, ScenePlotly, ScenePolyscope, ...)
   overrides only the methods it actually supports.
2. Nothing here is declared abstract. A method that isn't overridden simply
   inherits the default implementation below, which raises
   NotSupportedByBackendError. This is intentional: not every backend can
   support every feature (e.g. Plotly has no concept of render "samples" or
   "device"), and the project's stance is to fail loudly and specifically
   rather than silently no-op or fake the behavior.
3. Backends never import each other. They only ever import from Py3DRF.core.
"""


class NotSupportedByBackendError(NotImplementedError):
    """
    Raised when the Scene facade forwards a call to a backend that doesn't
    implement it. This is a normal, expected outcome of choosing a backend
    that doesn't support a given feature -- not a bug.
    """


class SceneBackend:
    """
    Base class for all Py3DRF backends. A backend wraps whatever native
    scene representation it needs (a bpy.types.Scene, a plotly Figure, a
    polyscope session, ...) and implements the subset of this interface it
    can meaningfully support.
    """

    # ------------------------------------------------------------------
    # Core: every real backend is expected to implement these.
    # ------------------------------------------------------------------

    def addObject(self, obj):
        """
        Build the backend-native representation of a Py3DRF.core.Mesh or
        Py3DRF.core.SunLight and add it to the scene.

        :param obj: Mesh or SunLight instance to add to the scene.
        :return: The backend-native object that was created (type varies
            per backend).
        """
        raise NotSupportedByBackendError(self._msg("addObject"))

    def addCamera(self, camera):
        """
        Build the backend-native representation of a Py3DRF.core.Camera,
        add it to the scene, and set it as the active camera (where the
        backend has such a concept).

        :param camera: Camera instance to add to the scene.
        :return: The backend-native camera object that was created.
        """
        raise NotSupportedByBackendError(self._msg("addCamera"))

    def renderToFile(self, filepath):
        """
        Render (or export) the scene to a static file on disk.

        :param filepath: Destination filepath for the rendered output.
        """
        raise NotSupportedByBackendError(self._msg("renderToFile"))

    # ------------------------------------------------------------------
    # Optional / backend-specific: only some backends make sense here.
    # ------------------------------------------------------------------

    def show(self):
        """
        Open an interactive viewer for the scene, for backends that support
        one (e.g. Plotly's browser viewer, Polyscope's GUI window).
        """
        raise NotSupportedByBackendError(self._msg("show"))

    def setRenderEngine(self, engine):
        """Set the render engine, for backends with a pluggable renderer (e.g. Blender)."""
        raise NotSupportedByBackendError(self._msg("setRenderEngine"))

    def setDevice(self, device):
        """Set the compute device (GPU/CPU), for backends that draw this distinction."""
        raise NotSupportedByBackendError(self._msg("setDevice"))

    def setResolution(self, resolution):
        """Set the output resolution, for backends that rasterize/render to an image."""
        raise NotSupportedByBackendError(self._msg("setResolution"))

    def setSamples(self, samples):
        """Set the number of rendering samples, for backends with a sampled renderer."""
        raise NotSupportedByBackendError(self._msg("setSamples"))

    def setGamma(self, gamma):
        """Set gamma correction, for backends with a configurable view transform."""
        raise NotSupportedByBackendError(self._msg("setGamma"))

    def setExposure(self, exposure):
        """Set exposure, for backends with a configurable view transform."""
        raise NotSupportedByBackendError(self._msg("setExposure"))

    def setShadowCatcherAlpha(self, alpha):
        """Set shadow-catcher blend alpha, for backends that implement shadow catching."""
        raise NotSupportedByBackendError(self._msg("setShadowCatcherAlpha"))

    def saveToFile(self, filename):
        """Save the native project/session file, for backends with a serializable project format."""
        raise NotSupportedByBackendError(self._msg("saveToFile"))

    # ------------------------------------------------------------------

    def _msg(self, name):
        return (
            f"{type(self).__name__} does not implement '{name}'. "
            f"This is a limitation of the backend, not of Py3DRF's Scene facade."
        )
