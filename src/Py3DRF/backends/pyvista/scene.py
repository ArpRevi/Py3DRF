"""
ScenePyVista: the PyVista backend for Py3DRF.

This is the only module in Py3DRF.backends.pyvista that imports pyvista. It
translates Py3DRF.core objects (Camera, Mesh, SunLight, Material,
PointCloudSettings, WireFrameSettings) into structures on a pyvista.Plotter.

Do not import this module directly -- go through the Scene facade
(Py3DRF.Scene(backend="pyvista", ...)), which loads it lazily.

Renders headless (off_screen=True) by default, so it's automation-friendly out
of the box like the other backends; pass off_screen=False for an interactive
window (see show()).

Honest limitations of this backend (deliberately left unimplemented rather
than faked -- calling them raises NotSupportedByBackendError):

  * setRenderEngine, setDevice, setSamples, setGamma, setExposure: PyVista's
    VTK-based renderer has no pluggable render engine, no CPU/GPU device
    switch (it uses whatever GPU context OpenGL gets), no path-traced
    sampling loop, and no separate gamma/exposure controls.
  * setShadowCatcherAlpha: is_shadow_catcher is approximated as a fixed 0.3
    opacity (the same fake Plotly and Polyscope use) -- there is no real
    shadow-catcher compositing pass here to make that alpha tunable.
  * saveToFile: pyvista.Plotter.export_html() would be the natural fit, but
    it requires the optional `trame` package, which isn't part of this
    backend's extra -- left unimplemented rather than adding a dependency
    this project doesn't otherwise need.
  * Material.emission_color / emission_strength: not modeled, for the same
    reason as Plotly/Polyscope -- no simple self-emission input on the
    property model used here.
  * WireFrameSettings edges: colored with a single flat color
    (settings.material.color), like Plotly/Polyscope's wireframes -- not
    attribute-driven, even though the mesh's surface material can be.

Where this backend does more than the others:
  * Material.color_attribute_colors (an arbitrary multi-stop color ramp) is
    reproduced EXACTLY via a custom pyvista.LookupTable, rather than
    approximated with a fixed built-in colormap the way Polyscope does.
  * WireFrameSettings.thickness_attribute (a per-vertex float attribute
    driving tube radius) is honored via tube()'s variable-radius `scalars`
    option -- Blender only approximates this with real 3D geometry and
    Plotly/Polyscope reject it outright.
"""

import numpy as np
import pyvista as pv

from ...core.camera import Camera
from ...core.lights import SunLight
from ...core.materials import Material
from ...core.mesh import Mesh
from ...core.pointcloud import PointCloudSettings
from ...core.wireframe import WireFrameSettings
from ..base import SceneBackend, NotSupportedByBackendError


def _facesToVTK(faces):
    """
    Flatten a list of variable-length faces into VTK's PolyData cell format
    ([n0, i0, i1, ..., n1, j0, j1, ...]). Unlike Plotly's go.Mesh3d, PyVista
    takes n-gons directly -- no triangulation needed.
    """
    if not faces:
        return None
    return np.hstack([[len(face), *face] for face in faces])


def _rotationToDirection(rotation):
    """
    Convert a Py3DRF Rotation into a world-space unit direction vector, using
    the same XYZ-Euler convention as Mesh._localToWorldMatrix (matching
    Blender's default rotation order), applied to (0, 0, -1) -- the direction
    an unrotated Blender Sun light points in. Used to make SunLight's rotation
    mean the same thing here as it does on the Blender backend.
    """
    rx, ry, rz = rotation.to_tuple()
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    rotation_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rotation_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rotation_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    rotation_matrix = rotation_z @ rotation_y @ rotation_x
    return rotation_matrix @ np.array([0.0, 0.0, -1.0])


def _buildLookupTable(colors, positions=None, n=256):
    """
    Build a pyvista.LookupTable reproducing an arbitrary multi-stop color ramp
    exactly, by linearly interpolating `colors` (at `positions`, or evenly
    spaced if not given) into `n` RGBA samples.
    """
    colors = np.asarray(colors, dtype=float)
    if positions is None:
        positions = np.linspace(0.0, 1.0, len(colors))
    else:
        positions = np.asarray(positions, dtype=float)

    sample_positions = np.linspace(0.0, 1.0, n)
    rgba = np.empty((n, 4))
    for channel in range(4):
        rgba[:, channel] = np.interp(sample_positions, positions, colors[:, channel])

    lookup_table = pv.LookupTable()
    lookup_table.values = (np.clip(rgba, 0.0, 1.0) * 255).astype(np.uint8)
    lookup_table.scalar_range = (0.0, 1.0)
    return lookup_table


class ScenePyVista(SceneBackend):

    def __init__(self, name="Scene", resolution=(1920, 1080), transparent=True, off_screen=True) -> None:
        self.name = name
        self.plotter = pv.Plotter(off_screen=off_screen, window_size=(int(resolution[0]), int(resolution[1])))
        self._transparent = transparent

    def setResolution(self, resolution):
        """
        Set the render/window size in pixels.

        :param resolution: (width, height) tuple.

        """
        self.plotter.window_size = (int(resolution[0]), int(resolution[1]))

    def addCamera(self, camera: Camera):
        """
        Point the PyVista camera at a Camera's target from its location.

        Like Plotly/Polyscope's look_at cameras, PyVista's camera is positioned by
        (position, focal_point, up) rather than Blender's absolute rotation, so only
        `camera.location` and `camera.target` are used; `camera.rotation` and
        `camera.focal_length` are ignored.

        `up` is world +Z, except when the view direction is itself nearly parallel to
        Z (looking straight down/up an axial-style view -- an entirely ordinary thing
        to want when viewing MRI data from directly above), in which case +Z can't be
        a valid up vector at all: cross(view_direction, up) is zero, VTK's camera
        basis becomes degenerate, and nothing renders (verified empirically -- this
        produced a silently all-white image with no error). World +Y is used instead
        for that case.

        :param camera: Camera object.
        :return: The pyvista.Camera that was configured.

        """
        location = camera.location.to_array()
        target = camera.target.to_array()

        self.plotter.camera.position = tuple(location)
        self.plotter.camera.focal_point = tuple(target)

        view_direction = target - location
        norm = np.linalg.norm(view_direction)
        view_direction = view_direction / norm if norm > 0 else np.array([0.0, 0.0, -1.0])
        world_up = np.array([0.0, 0.0, 1.0])
        up = np.array([0.0, 1.0, 0.0]) if abs(np.dot(view_direction, world_up)) > 0.999 else world_up
        self.plotter.camera.up = tuple(up)

        return self.plotter.camera

    def addObject(self, object):
        """
        Add a Mesh to the scene as a PyVista surface mesh (or a glyphed point cloud,
        when point-cloud settings are applied via asPointCloud(), or a tube-based
        wireframe, when wireframe settings are applied via asWireframe()), or a
        SunLight as a non-positional (directional) pyvista.Light.

        :param object: Mesh or SunLight object to add to the scene.
        :return: The pyvista actor/light that was added.

        """
        if isinstance(object, SunLight):
            return self._addSunLight(object)
        if isinstance(object, Mesh):
            return self._addMesh(object)
        raise TypeError(f"Cannot add object of type {type(object).__name__} to the scene.")

    def show(self):
        """Open an interactive PyVista window (requires a display; not available headless)."""
        self.plotter.show()

    def renderToFile(self, filepath):
        """
        Render the current view to a static image file.

        :param filepath: Destination filepath for the rendered screenshot.

        """
        self.plotter.screenshot(str(filepath), transparent_background=self._transparent)

    # -- Builders: translate plain-Python Py3DRF objects into pyvista structures --

    def _addSunLight(self, light: SunLight):
        """
        Build a non-positional (directional) pyvista.Light from a SunLight, using
        `light.rotation` for direction (see _rotationToDirection) -- like Blender,
        `light.location` only offsets where the light is anchored, not the direction
        it shines, since a directional light's effect depends only on that direction.
        """
        direction = _rotationToDirection(light.rotation)
        location = light.location.to_array()

        pv_light = pv.Light(light_type="scene light")
        pv_light.positional = False
        pv_light.position = tuple(location)
        pv_light.focal_point = tuple(location + direction)
        pv_light.intensity = float(light.strength)

        self.plotter.add_light(pv_light)
        return pv_light

    def _addMesh(self, mesh: Mesh):
        if mesh.wireframe_settings is not None:
            return self._addWireframe(mesh)
        if mesh.point_cloud_settings is not None:
            return self._addPointCloud(mesh)

        polydata = self._buildSurfacePolyData(mesh)
        return self._addSurface(polydata, mesh, mesh.material, smooth_shading=mesh.shade_smooth)

    def _buildSurfacePolyData(self, mesh: Mesh):
        """
        Build a pyvista.PolyData from a Mesh's world-space geometry, with its
        float/color attributes attached as point data (available to _addSurface's
        material resolution by name, exactly like the attributes they came from).
        """
        vertices = mesh._worldVertices()
        polydata = pv.PolyData(vertices, _facesToVTK(mesh.faces))

        for attr_name, (values, _domain) in mesh.float_attributes.items():
            polydata[attr_name] = np.asarray(values, dtype=float)
        for attr_name, (values, _domain) in mesh.color_attributes.items():
            colors = np.clip(np.asarray(values, dtype=float)[:, :3], 0.0, 1.0)
            polydata[attr_name] = (colors * 255).astype(np.uint8)

        return polydata

    def _addSurface(self, polydata, mesh: Mesh, material: Material, smooth_shading):
        """
        Add `polydata` to the plotter, resolving `material`'s color the same way as
        the Plotly/Polyscope backends: a float attribute driving an exact color-ramp
        LookupTable, a color-type attribute used directly per-vertex, or a flat color.
        """
        kwargs = dict(
            smooth_shading=smooth_shading and polydata.n_cells > 0,
            opacity=0.3 if mesh.is_shadow_catcher else 1.0,
            pbr=True,
            roughness=float(material.roughness),
            metallic=0.0,
            show_scalar_bar=False,
        )

        if material.color_attribute is not None:
            if material.color_attribute in polydata.point_data and material.color_attribute_colors is not None:
                lookup_table = _buildLookupTable(material.color_attribute_colors, material.color_attribute_positions)
                return self.plotter.add_mesh(
                    polydata, scalars=material.color_attribute, cmap=lookup_table, clim=(0.0, 1.0), **kwargs
                )
            if material.color_attribute in mesh.color_attributes:
                return self.plotter.add_mesh(polydata, scalars=material.color_attribute, rgb=True, **kwargs)

        return self.plotter.add_mesh(polydata, color=tuple(material.color[:3]), **kwargs)

    def _addWireframe(self, mesh: Mesh):
        """
        Build a tube-based 3D wireframe from a Mesh's edges: real geometry, like
        Blender's cylinders (not the flat screen-space lines Plotly/Polyscope draw),
        with per-vertex variable radius when settings.thickness_attribute is set --
        tube()'s `scalars` option supports this natively, unlike Blender's more
        involved geometry-nodes approximation.

        :param mesh: Mesh with wireframe_settings applied via asWireframe().
        :return: The tube actor (the surface actor, if also added because
            hide_surface=False, is not returned).

        """
        settings: WireFrameSettings = mesh.wireframe_settings
        material = settings.material if settings.material is not None else mesh.material

        vertices = mesh._worldVertices()
        edges = mesh._edgePairs()
        lines = np.hstack([[2, int(a), int(b)] for a, b in edges]) if len(edges) else None
        line_polydata = pv.PolyData(vertices, lines=lines)

        if settings.thickness_attribute is not None:
            values, _domain = mesh.float_attributes[settings.thickness_attribute]
            line_polydata["thickness"] = np.asarray(values, dtype=float)
            tube = line_polydata.tube(
                radius=settings.thickness / 2, scalars="thickness", absolute=True, n_sides=settings.resolution
            )
        else:
            tube = line_polydata.tube(radius=settings.thickness / 2, n_sides=settings.resolution)

        actor = self.plotter.add_mesh(
            tube,
            color=tuple(material.color[:3]),
            opacity=0.3 if mesh.is_shadow_catcher else 1.0,
            show_scalar_bar=False,
        )

        if not settings.hide_surface:
            surface = self._buildSurfacePolyData(mesh)
            self._addSurface(surface, mesh, mesh.material, smooth_shading=mesh.shade_smooth)

        return actor

    def _addPointCloud(self, mesh: Mesh):
        """
        Build a glyphed point cloud from a Mesh's vertices: each vertex becomes an
        icosphere (pyvista.Icosphere, matching Blender's icosphere point-cloud
        instances directly, subdivided by settings.subdivison), scaled by a constant
        radius or a per-point settings.radius_attribute.

        :param mesh: Mesh with point_cloud_settings applied via asPointCloud().
        :return: The glyph actor.

        """
        settings: PointCloudSettings = mesh.point_cloud_settings
        material = settings.material if settings.material is not None else mesh.material

        vertices = mesh._worldVertices()
        cloud = pv.PolyData(vertices)

        for attr_name, (values, _domain) in mesh.float_attributes.items():
            cloud[attr_name] = np.asarray(values, dtype=float)
        for attr_name, (values, _domain) in mesh.color_attributes.items():
            colors = np.clip(np.asarray(values, dtype=float)[:, :3], 0.0, 1.0)
            cloud[attr_name] = (colors * 255).astype(np.uint8)

        sphere = pv.Icosphere(radius=1.0, nsub=settings.subdivison)

        if settings.radius_attribute is not None:
            cloud["_py3drf_radius"] = np.asarray(mesh.float_attributes[settings.radius_attribute][0], dtype=float)
            glyphs = cloud.glyph(geom=sphere, scale="_py3drf_radius", orient=False)
        else:
            glyphs = cloud.glyph(geom=pv.Icosphere(radius=settings.radius, nsub=settings.subdivison), scale=False, orient=False)

        return self._addSurface(glyphs, mesh, material, smooth_shading=True)
