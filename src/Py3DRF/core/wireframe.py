"""
Settings describing how a Mesh should be rendered as a wireframe: its faces hidden, its
edges highlighted.

What "edges highlighted" actually builds is backend-specific, since not every backend has
the same wireframe primitive available:

  * Blender builds real 3D geometry -- every edge becomes a cylinder of the given
    thickness (a true diameter, in the mesh's local units) via geometry nodes. Being
    actual geometry, the cylinders shade, cast and receive shadows, and get thinner with
    distance like any other object in the scene. Faces are dropped structurally (the
    node graph never rebuilds them), not hidden with a transparency trick.
  * Plotly and Polyscope instead draw a genuine screen-space wireframe: a flat line of
    constant apparent pixel width along each edge, no volume, no shading. `thickness` is
    heuristically rescaled from a local-unit diameter into a pixel width for these (the
    same kind of approximation PointCloudSettings.radius already gets for Plotly marker
    size). Polyscope additionally can't hide the surface while keeping this overlay (see
    its backend module docstring), so hide_surface=True raises there.

This module (Py3DRF.core) has no dependency on any rendering backend. Building the actual
backend-native wireframe representation from a WireFrameSettings instance is the
responsibility of whichever backend the user chose via the Scene facade.
"""

from .materials import Material


class WireFrameSettings:
    """
    Describes the wireframe rendering of a mesh: its faces hidden (backend permitting),
    its edges highlighted -- as 3D cylinders or as flat lines, depending on the backend
    (see the module docstring).
    """

    def __init__(self, name="Wireframe", thickness=0.01, resolution=12, material: Material = None, hide_surface=True) -> None:
        """

        :param name: Name to give the Blender modifier/node group, or the Plotly/Polyscope
            trace/structure, built from these settings.
        :param thickness: Girth of the wireframe edges. For Blender, a real diameter in
            the mesh's local units (so it is affected by the parent mesh's scale, exactly
            like the mesh geometry itself). For Plotly/Polyscope, heuristically rescaled
            into a screen-space pixel line width instead (see the module docstring).
        :param resolution: Number of sides of the cylinders Blender builds, i.e. how many
            segments the circular profile swept along each edge is made of. Higher is
            rounder and heavier; 12 is plenty for typical renders, 6 gives visibly
            faceted tubes. Ignored by Plotly/Polyscope, which draw flat lines with no
            profile to subdivide.
        :param material: Material object to link to the wireframe. Defaults to the parent
            mesh's material if not provided. On Blender, color/float attributes defined
            on the mesh carry over onto the cylinders, since they're real geometry with
            their own material node graph; on Plotly/Polyscope only material.color (a
            flat color) is used, as the line/edge color.
        :param hide_surface: If True (default), only the wireframe edges are rendered. If
            False, the mesh's solid shaded surface is rendered as well, with the edges on
            top of it. Polyscope cannot honor True (see the module docstring) and raises
            NotSupportedByBackendError instead of faking it.

        """
        self.name = name
        self.thickness = thickness
        self.resolution = resolution
        self.material = material
        self.hide_surface = hide_surface

        self.thickness_attribute = None

    def setWireframeThickness(self, thickness):
        """
        Set thickness of the wireframe edges.

        :param thickness: Girth of the edge cylinders, as a diameter in the mesh's local
            units.

        """
        self.thickness = thickness

    def setWireframeResolution(self, resolution):
        """
        Set the number of sides of the edge cylinders.

        :param resolution: Number of segments of the circular profile swept along each edge.

        """
        self.resolution = resolution

    def setHideSurface(self, hide_surface):
        """
        Set whether the mesh's own shaded surface should be hidden, leaving only the
        wireframe edges visible. Polyscope cannot honor True (see the module docstring).

        :param hide_surface: If True, only the wireframe edges are rendered. If False, the
            solid shaded mesh is rendered as well, with the edges on top.

        """
        self.hide_surface = hide_surface

    def setFloatAttributeAsThickness(self, attribute_name):
        """
        Link a named float attribute to the thickness of the wireframe edges, overriding
        setWireframeThickness. On Blender, where a cylinder is real per-edge geometry,
        each one then varies in girth along its length, following the attribute value at
        the vertices it connects. Plotly and Polyscope have no per-vertex line-width
        concept (their whole wireframe is one trace/structure with a single scalar
        width), so they reject this rather than approximate it.

        :param attribute_name: Name of the float attribute. It must live on the POINT
            domain, since the thickness is evaluated per vertex of the edge.

        """
        self.thickness_attribute = attribute_name

    def clearThicknessAttribute(self):
        """
        Stop driving the edge thickness from an attribute; fall back to the constant value set
        by setWireframeThickness.
        """
        self.thickness_attribute = None
