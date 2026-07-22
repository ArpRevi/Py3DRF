"""
ScenePlotly: the Plotly backend for Py3DRF.

This is the only module in Py3DRF.backends.plotly that imports plotly. It
translates Py3DRF.core objects (Camera, Mesh, Material, PointCloudSettings)
into traces on a plotly.graph_objects.Figure.

Do not import this module directly -- go through the Scene facade
(Py3DRF.Scene(backend="plotly", ...)), which loads it lazily.

Honest limitations of this backend (deliberately left unimplemented rather
than faked -- calling them raises NotSupportedByBackendError):

  * setRenderEngine, setDevice, setSamples, setGamma, setExposure,
    setShadowCatcherAlpha: Plotly has no path-traced renderer, no
    GPU/CPU device selection, no sampling loop, and no shadow-catcher
    compositing -- none of these concepts exist here.
  * SunLight objects: Plotly's Mesh3d has a per-trace `lighting`/
    `lightposition`, not a scene-level light object, so a SunLight added
    via addObject() is rejected rather than silently dropped or
    approximated.
  * Material.emission_color / emission_strength: Plotly has no emission
    shading model, so these are not applied. Only Material.color (or a
    color/float attribute driving vertex colors) is mapped.
"""

import numpy as np
import plotly.graph_objects as go

from ...core.camera import Camera
from ...core.lights import SunLight
from ...core.materials import Material
from ...core.mesh import Mesh
from ..base import SceneBackend, NotSupportedByBackendError


def _rgba_to_css(color):
    """Convert an (r, g, b[, a]) float tuple in [0, 1] to a CSS rgba() string."""
    r, g, b = (int(round(c * 255)) for c in color[:3])
    a = color[3] if len(color) > 3 else 1.0
    return f"rgba({r},{g},{b},{a})"


class ScenePlotly(SceneBackend):

    def __init__(self, name="Scene", resolution=(1920, 1080), transparent=True) -> None:
        self.name = name
        self.figure = go.Figure()
        self.figure.update_layout(
            scene=dict(
                aspectmode="data",
                xaxis=dict(visible=not transparent),
            ),
            paper_bgcolor="rgba(0,0,0,0)" if transparent else "white",
            margin=dict(l=0, r=0, t=0, b=0),
        )
        self.setResolution(resolution)

    def setResolution(self, resolution):
        """
        Set the figure's output size in pixels.

        :param resolution: (width, height) tuple.

        """
        self.figure.update_layout(width=resolution[0], height=resolution[1])

    def addCamera(self, camera: Camera):
        """
        Point the Plotly scene camera at the origin from a Camera's location.

        Plotly's camera model (eye/center/up, all relative to the plot's data
        range) has no equivalent of Blender's absolute focal length or
        rotation, so only `camera.location` is used here, as the direction
        and distance of the `eye` from `center=(0, 0, 0)`. `camera.rotation`
        and `camera.focal_length` are not applicable to Plotly's camera model
        and are ignored.

        :param camera: Camera object.
        :return: The plotly camera dict that was applied to the figure.

        """
        location = camera.location
        camera_dict = dict(
            eye=dict(x=float(location.x), y=float(location.y), z=float(location.z)),
            center=dict(x=0.0, y=0.0, z=0.0),
            up=dict(x=0.0, y=0.0, z=1.0),
        )
        self.figure.update_layout(scene_camera=camera_dict)
        return camera_dict

    def addObject(self, object):
        """
        Add a Mesh to the figure as a go.Mesh3d trace (or a go.Scatter3d trace,
        when the mesh has point-cloud settings applied via asPointCloud()).

        :param object: Mesh object to add to the scene. SunLight is not
            supported (see module docstring) and raises
            NotSupportedByBackendError.
        :return: The plotly trace that was added to the figure.

        """
        if isinstance(object, SunLight):
            raise NotSupportedByBackendError(
                "ScenePlotly cannot add a SunLight: Plotly has no scene-level light "
                "object, only per-trace lighting/lightposition on Mesh3d."
            )
        if not isinstance(object, Mesh):
            raise TypeError(f"Cannot add object of type {type(object).__name__} to the scene.")

        if object.point_cloud_settings is not None:
            trace = self._buildScatter3d(object)
        else:
            trace = self._buildMesh3d(object)

        self.figure.add_trace(trace)
        return trace

    def show(self):
        """Open the figure in an interactive viewer (browser or notebook)."""
        self.figure.show()

    def renderToFile(self, filepath):
        """
        Export the figure to filepath. `.html` produces an interactive page;
        any other extension (`.png`, `.jpg`, `.svg`, `.pdf`, ...) produces a
        static image and requires the optional `kaleido` package.

        :param filepath: Destination filepath for the rendered output.

        """
        if str(filepath).lower().endswith(".html"):
            self.figure.write_html(filepath)
            return

        try:
            self.figure.write_image(filepath)
        except ValueError as exc:
            raise RuntimeError(
                "Static image export requires the optional 'kaleido' package. "
                "Install it with: pip install kaleido"
            ) from exc

    def saveToFile(self, filename):
        """
        Save the figure definition as JSON (Plotly's own serialization format,
        re-loadable with plotly.io.read_json).

        :param filename: Destination filepath for the saved .json file.

        """
        self.figure.write_json(filename)

    # -- Builders: translate plain-Python Py3DRF objects into plotly traces --

    def _buildMesh3d(self, mesh: Mesh):
        """
        Build a go.Mesh3d trace from a Mesh's world-space geometry and material.
        """
        vertices = mesh._worldVertices()
        faces = np.asarray(mesh.faces)

        trace_kwargs = dict(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            name=mesh.name,
            flatshading=not mesh.shade_smooth,
            opacity=1.0 if not mesh.is_shadow_catcher else 0.3,
        )
        trace_kwargs.update(self._materialTraceKwargs(mesh))
        return go.Mesh3d(**trace_kwargs)

    def _buildScatter3d(self, mesh: Mesh):
        """
        Build a go.Scatter3d trace approximating Blender's icosphere-per-point
        rendering: one marker per vertex, sized from PointCloudSettings.radius
        (or driven per-point by its radius_attribute, if set).
        """
        vertices = mesh._worldVertices()
        settings = mesh.point_cloud_settings

        if settings.radius_attribute is not None:
            values, _ = mesh.float_attributes[settings.radius_attribute]
            sizes = np.asarray(values, dtype=float)
            # Plotly marker `size` is in pixels, not data units, so this is a
            # relative approximation of Blender's world-space point radius.
            sizes = 4 + 40 * (sizes / sizes.max() if sizes.max() > 0 else sizes)
        else:
            sizes = settings.radius * 400  # heuristic pixel scaling

        marker = dict(size=sizes)
        marker.update(self._materialMarkerKwargs(mesh, settings.material))

        return go.Scatter3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            mode="markers",
            name=mesh.name,
            marker=marker,
        )

    def _materialTraceKwargs(self, mesh: Mesh):
        """
        Map a Mesh's Material (and any linked color/float attributes) onto
        go.Mesh3d color kwargs: a flat `color`, an explicit `vertexcolor`
        list, or an `intensity` + `colorscale` gradient.
        """
        material = mesh.material

        if material.color_attribute is not None:
            values, _ = mesh.float_attributes.get(material.color_attribute, (None, None))
            if values is not None and material.color_attribute_colors is not None:
                # Float attribute driving a color ramp -> intensity + colorscale.
                return dict(
                    intensity=np.asarray(values, dtype=float),
                    colorscale=[_rgba_to_css(c) for c in material.color_attribute_colors],
                    intensitymode="vertex",
                    showscale=False,
                )
            colors, _ = mesh.color_attributes.get(material.color_attribute, (None, None))
            if colors is not None:
                # Color-type attribute used directly -> per-vertex vertexcolor.
                return dict(vertexcolor=[_rgba_to_css(c) for c in colors])

        return dict(color=_rgba_to_css(material.color))

    def _materialMarkerKwargs(self, mesh: Mesh, material: Material):
        """Same mapping as _materialTraceKwargs, but for go.Scatter3d's `marker` dict."""
        if material is None:
            material = mesh.material

        if material.color_attribute is not None:
            values, _ = mesh.float_attributes.get(material.color_attribute, (None, None))
            if values is not None and material.color_attribute_colors is not None:
                return dict(
                    color=np.asarray(values, dtype=float),
                    colorscale=[_rgba_to_css(c) for c in material.color_attribute_colors],
                    showscale=False,
                )
            colors, _ = mesh.color_attributes.get(material.color_attribute, (None, None))
            if colors is not None:
                return dict(color=[_rgba_to_css(c) for c in colors])

        return dict(color=_rgba_to_css(material.color))
