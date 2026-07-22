"""
ScenePolyscope: the Polyscope backend for Py3DRF.

This is the only module in Py3DRF.backends.polyscope that imports polyscope.
It translates Py3DRF.core objects (Camera, Mesh, Material, PointCloudSettings)
into structures registered on a polyscope session.

Do not import this module directly -- go through the Scene facade
(Py3DRF.Scene(backend="polyscope", ...)), which loads it lazily.

Honest limitations of this backend (deliberately left unimplemented rather
than faked -- calling them raises NotSupportedByBackendError):

  * setRenderEngine, setDevice, setSamples, setGamma, setExposure,
    setShadowCatcherAlpha, saveToFile: Polyscope has no path-traced
    renderer, no configurable device/sampling/exposure pipeline, no
    shadow-catcher compositing, and no serializable native project file.
  * SunLight objects: Polyscope's ground-truth lighting is a fixed
    ambient/Matcap-style shading model with no scene-level directional
    light object, so a SunLight added via addObject() is rejected rather
    than silently dropped or approximated.
  * Material.emission_color / emission_strength: not modeled, for the
    same reason as the Plotly backend.
  * Material's custom color-ramp (color_attribute_colors): Polyscope's
    scalar quantities are colored using a named built-in colormap, not an
    arbitrary user-supplied gradient, so the ramp colors are approximated
    with the closest built-in colormap ("viridis") rather than reproduced
    exactly.

Polyscope keeps a single global session (it isn't one-object-per-scene like
Blender or Plotly), so only one ScenePolyscope should be "live" at a time;
creating a new one clears any structures registered by a previous instance.
"""

import numpy as np
import polyscope as ps

from ...core.camera import Camera
from ...core.lights import SunLight
from ...core.materials import Material
from ...core.mesh import Mesh
from ..base import SceneBackend, NotSupportedByBackendError

_DEFAULT_COLORMAP = "viridis"


class ScenePolyscope(SceneBackend):

    def __init__(self, name="Scene", resolution=(1920, 1080), transparent=True) -> None:
        self.name = name

        if not ps.is_initialized():
            ps.set_program_name(name)
            # Allows rendering (including screenshot()) on machines with no
            # display attached, falling back to an EGL/OSMesa backend.
            ps.set_allow_headless_backends(True)
            ps.init()
        else:
            # Polyscope is a single global session: reset it so this Scene
            # starts from a clean slate instead of stacking onto whatever a
            # previous ScenePolyscope instance left behind.
            ps.remove_all_structures()

        ps.set_up_dir("z_up")
        ps.set_background_color((0.0, 0.0, 0.0) if not transparent else (1.0, 1.0, 1.0))
        self.setResolution(resolution)

    def setResolution(self, resolution):
        """
        Set the render/window size in pixels.

        :param resolution: (width, height) tuple.

        """
        ps.set_window_size(int(resolution[0]), int(resolution[1]))

    def addCamera(self, camera: Camera):
        """
        Point the Polyscope view at the origin from a Camera's location.

        Polyscope's look_at(location, target) is an absolute-position camera,
        unlike Plotly's relative eye/center, but still has no equivalent of
        Blender's focal length or object rotation, so only `camera.location`
        is used, looking at the origin. `camera.rotation` and
        `camera.focal_length` are ignored.

        :param camera: Camera object.
        :return: The (location, target) tuple that was applied to the view.

        """
        location = camera.location.to_tuple()
        target = (0.0, 0.0, 0.0)
        ps.look_at(location, target)
        return (location, target)

    def addObject(self, object):
        """
        Add a Mesh to the session as a Polyscope surface mesh (or point cloud,
        when the mesh has point-cloud settings applied via asPointCloud()).

        :param object: Mesh object to add to the scene. SunLight is not
            supported (see module docstring) and raises
            NotSupportedByBackendError.
        :return: The polyscope structure that was registered (SurfaceMesh or
            PointCloud).

        """
        if isinstance(object, SunLight):
            raise NotSupportedByBackendError(
                "ScenePolyscope cannot add a SunLight: Polyscope has no scene-level "
                "light object, only a fixed built-in shading model."
            )
        if not isinstance(object, Mesh):
            raise TypeError(f"Cannot add object of type {type(object).__name__} to the scene.")

        if object.point_cloud_settings is not None:
            return self._registerPointCloud(object)
        return self._registerSurfaceMesh(object)

    def show(self):
        """Open Polyscope's interactive GUI window (requires a display; not available headless)."""
        ps.show()

    def renderToFile(self, filepath):
        """
        Render the current view to a static image file.

        :param filepath: Destination filepath for the rendered screenshot.

        """
        ps.screenshot(str(filepath))

    # -- Builders: translate plain-Python Py3DRF objects into polyscope structures --

    def _registerSurfaceMesh(self, mesh: Mesh):
        """
        Build a polyscope SurfaceMesh from a Mesh's world-space geometry and material.
        """
        vertices = mesh._worldVertices()
        faces = np.asarray(mesh.faces)

        structure = ps.register_surface_mesh(
            mesh.name,
            vertices,
            faces,
            smooth_shade=mesh.shade_smooth,
            transparency=0.3 if mesh.is_shadow_catcher else 1.0,
        )
        self._applyMaterial(structure, mesh, mesh.material)
        return structure

    def _registerPointCloud(self, mesh: Mesh):
        """
        Build a polyscope PointCloud from a Mesh's world-space vertices and its
        PointCloudSettings (radius, and optionally a per-point radius attribute).
        """
        vertices = mesh._worldVertices()
        settings = mesh.point_cloud_settings

        structure = ps.register_point_cloud(mesh.name, vertices, radius=settings.radius)

        if settings.radius_attribute is not None:
            values, _ = mesh.float_attributes[settings.radius_attribute]
            structure.add_scalar_quantity(
                settings.radius_attribute, np.asarray(values, dtype=float)
            )
            structure.set_point_radius_quantity(settings.radius_attribute)

        material = settings.material if settings.material is not None else mesh.material
        self._applyMaterial(structure, mesh, material)
        return structure

    def _applyMaterial(self, structure, mesh: Mesh, material: Material):
        """
        Map a Material (and any linked color/float attributes) onto a polyscope
        structure (SurfaceMesh or PointCloud): a flat `color`, a per-vertex
        color quantity, or a scalar quantity shown with a built-in colormap.
        """
        if material.color_attribute is not None:
            values, _ = mesh.float_attributes.get(material.color_attribute, (None, None))
            if values is not None and material.color_attribute_colors is not None:
                # Float attribute driving a color ramp -> scalar quantity with
                # a built-in colormap (the exact custom ramp isn't reproducible,
                # see module docstring).
                quantity = structure.add_scalar_quantity(
                    material.color_attribute,
                    np.asarray(values, dtype=float),
                    cmap=_DEFAULT_COLORMAP,
                    enabled=True,
                )
                return quantity

            colors, _ = mesh.color_attributes.get(material.color_attribute, (None, None))
            if colors is not None:
                # Color-type attribute used directly -> per-vertex color quantity.
                rgb = np.asarray(colors, dtype=float)[:, :3]
                quantity = structure.add_color_quantity(
                    material.color_attribute, rgb, enabled=True
                )
                return quantity

        structure.set_color(tuple(material.color[:3]))
        return None
