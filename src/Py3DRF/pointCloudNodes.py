"""
Settings describing how a Mesh should be rendered as a point cloud.

This module has no dependency on bpy. Building the actual Blender
geometry-nodes node tree from a PointCloudSettings instance is the
responsibility of Scene (see scene.py), which is the only module in this
package allowed to import bpy.
"""

from .materials import Material


class PointCloudSettings:
    """
    Describes the point-cloud rendering of a mesh: each vertex of the mesh is
    replaced, at render time, with an icosphere of the given radius and
    subdivision level.
    """

    def __init__(self, name="Pointcloud", radius=0.01, subdivison=3, material: Material = None) -> None:
        """

        :param name: Name to give the Blender modifier/node group built from these settings.
        :param radius: Radius of the rendered pointclouds.
        :param subdivison: Number of subdivisions of the icospheres representing the points.
        :param material: Material object to link to the point cloud. Defaults to the parent
            mesh's material if not provided.

        """
        self.name = name
        self.radius = radius
        self.subdivison = subdivison
        self.material = material
        self.radius_attribute = None

    def setPointsRadius(self, radius):
        """
        Set radius of the point cloud points.

        :param radius: Radius of the point cloud points.

        """
        self.radius = radius

    def setPointsSubdivisions(self, subdivison):
        """
        Set number of subdivisions of the icospheres representing the points.

        :param subdivison: Number of subdivisions of the icospheres representing the points.

        """
        self.subdivison = subdivison

    def setFloatAttributeAsRadius(self, attribute_name):
        """
        Link a named float attribute to the radius of the point cloud points, overriding
        setPointsRadius.

        :param attribute_name: Name of the float attribute.

        """
        self.radius_attribute = attribute_name

    def clearRadiusAttribute(self):
        """
        Stop driving the points radius from an attribute; fall back to the constant value set by
        setPointsRadius.
        """
        self.radius_attribute = None


# Backwards-compatible alias: this class used to eagerly build the Blender
# geometry-nodes node tree in its constructor (hence the name). It is now a
# plain settings object; Scene builds the actual bpy node tree from it on
# demand, when the owning Mesh is added to a Scene.
MeshToPointCloudNodeTree = PointCloudSettings
