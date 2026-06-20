import bpy

from b3Dv.materials import Material

"""
Class representing a Blender modifier that converts a mesh object to a point cloud.
"""
class MeshToPointCloudNodeTree:
    def __init__(self, name="Pointcloud", radius=0.01, subdivison=3, material=None) -> None:
        """

        :param name: Name of the Blender modifier created.
        :param radius: Radius of the rendered pointclouds.
        :param subdivision: Number of subdivisions of the icospheres representing the points.
        :param material: Material object to link to the point cloud.

        """
        self.node_group = bpy.data.node_groups.new(type="GeometryNodeTree", name=name)
        self.node_group.nodes.clear()
        self.nodes = {}
        self.links = {}

        self.nodes['input'] = self.node_group.nodes.new(type="NodeGroupInput")
        self.nodes['output'] = self.node_group.nodes.new(type="NodeGroupOutput")
        self.nodes['meshToPoints'] = self.node_group.nodes.new(type="GeometryNodeMeshToPoints")
        self.nodes['instanceOnPoints'] = self.node_group.nodes.new(type="GeometryNodeInstanceOnPoints")
        self.nodes['icoSphere'] = self.node_group.nodes.new(type="GeometryNodeMeshIcoSphere")
        self.nodes['setShadeSmooth'] = self.node_group.nodes.new(type="GeometryNodeSetShadeSmooth")
        self.nodes['setMaterial'] = self.node_group.nodes.new(type="GeometryNodeSetMaterial")
        self.nodes['realizeInstances'] = self.node_group.nodes.new(type="GeometryNodeRealizeInstances")

        self.node_group.interface.new_socket(name="Geometry", description="", in_out="INPUT", socket_type="NodeSocketGeometry")
        self.node_group.interface.new_socket(name="Geometry", description="", in_out="OUTPUT", socket_type="NodeSocketGeometry")

        self.links['inputToMeshToPoints'] = self.node_group.links.new(self.nodes['input'].outputs['Geometry'], self.nodes['meshToPoints'].inputs['Mesh'])
        self.links['meshToPointsToInstanceOnPoints'] = self.node_group.links.new(self.nodes['meshToPoints'].outputs['Points'], self.nodes['instanceOnPoints'].inputs['Points'])
        self.links['icoSphereToSetShadeSmooth'] = self.node_group.links.new(self.nodes['icoSphere'].outputs['Mesh'], self.nodes['setShadeSmooth'].inputs['Geometry'])
        self.links['setShadeSmoothToInstanceOnPoints'] = self.node_group.links.new(self.nodes['setShadeSmooth'].outputs['Geometry'], self.nodes['instanceOnPoints'].inputs['Instance'])
        self.links['instanceOnPointsToSetMaterial'] = self.node_group.links.new(self.nodes['instanceOnPoints'].outputs['Instances'], self.nodes['setMaterial'].inputs['Geometry'])
        self.links['setMaterialToRealizeInstances'] = self.node_group.links.new(self.nodes['setMaterial'].outputs['Geometry'], self.nodes['realizeInstances'].inputs['Geometry'])
        self.links['realizeInstancesToOutput'] = self.node_group.links.new(self.nodes['realizeInstances'].outputs['Geometry'], self.nodes['output'].inputs['Geometry'])

        self.nodes['icoSphere'].inputs['Radius'].default_value = 1

        self.setPointsRadius(radius)
        self.setPointsSubdivisions(subdivison)

        if material is not None:
            self.nodes['setMaterial'].inputs['Material'].default_value = material.data

    def setPointsRadius(self, radius):
        """
        Set radius of the point cloud points.

        :param radius: Radius of the point cloud points.

        """
        self.nodes['instanceOnPoints'].inputs['Scale'].default_value = (radius, radius, radius)

    def setPointsSubdivisions(self, subdivison):
        """
        Set number of subdivisions of the icospheres representing the points.

        :param subdivison: Number of subdivisions of the icospheres representing the points.
        
        """
        self.nodes['icoSphere'].inputs['Subdivisions'].default_value = subdivison

    def setFloatAttributeAsRadius(self, attribute_name):
        """
        Link a float attribute to the radius of the point cloud points.

        :param attribute_name: Name of the float attribute.

        """
        self._removeAttributeAsRadius()

        self.nodes['radiusAttribute'] = self.node_group.nodes.new(type="GeometryNodeInputNamedAttribute")
        self.links['radiusAttributeToInstanceOnPoints'] = self.node_group.links.new(self.nodes['radiusAttribute'].outputs['Attribute'], self.nodes['instanceOnPoints'].inputs['Scale'])

        self.nodes['radiusAttribute'].data_type = 'FLOAT'
        self.nodes['radiusAttribute'].inputs['Name'].default_value = attribute_name

    def _removeAttributeAsRadius(self):
        self._removeLink('radiusAttributeToInstanceOnPoints')
        self._removeNode('radiusAttribute')

    def _removeLink(self, link_name):
        if link_name in self.links.keys():
            self.node_group.links.remove(self.links[link_name])
            self.links.pop(link_name)

    def _removeNode(self, node_name):
        if node_name in self.nodes.keys():
            self.node_group.nodes.remove(self.nodes[node_name])
            self.nodes.pop(node_name)