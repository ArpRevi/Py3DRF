import bpy
import numpy as np

from b3Dv.pointCloudNodes import MeshToPointCloudNodeTree
from b3Dv.materials import Material
from b3Dv.camera import Camera

class Mesh:
    """
    Mesh class representing a Blender mesh object.
    """
    def __init__(
            self,
            name="Mesh",
            vertices=[],
            edges=[],
            faces=[],
            location=(0, 0, 0),
            rotation=(0, 0, 0),
            scale=(1, 1, 1),
            material:Material=None,
            shade_smooth=False
            ) -> None:
        """
        Creates a mesh object with specified vertices, edges and faces. When faces are provided edges for the faces are inferred.

        :param name: Name of the mesh object.
        :param vertices: List of vertices in the local reference frame.
        :param edges: List of couple of indices of vertices representing the edges.
        :param faces: List of triplets of indices of vertices representing the faces.
        :param location: Vector of coordinates representing the new location of the object.
        :param rotation: Rotation of the mesh object in the local reference frame in radiants.
        :param scale: Scale on the xyz axes.
        :param material: Material object to link to the mesh.
        :param shade_smooth: Boolean setting smooth shading.

        """
        self.data = bpy.data.meshes.new(name=name)
        self.data.from_pydata(vertices, edges, faces)
        self.data.update()
        self.data.validate()
        self.object = bpy.data.objects.new(name, self.data)
        self.setLocation(location)
        self.setRotation(rotation)
        self.setScale(scale)

        if material is None:
            material = Material()

        self.setMaterial(material)

        self.setShadeSmooth(shade_smooth)

    def addFloatAttribute(self, data, name="value", domain="POINT"):
        """
        Add a float attribute to the mesh. The values are added without any normalization.

        :param data: List of floats. The lenght of the list should be equal to the number of element in the selected domain.
        :param name: Name of the new attribute.
        :param domain: Domain to which the point refers to. Can be one of POINT, FACE, EDGE, CORNER.

        """
        attribute = self.object.data.attributes.new(name=name, type='FLOAT', domain=domain)
        attribute.data.foreach_set('value', data.ravel())

    def addColorAttribute(self, data, name="value", domain="POINT"):
        """
        Add a color attribute to the mesh. The values are added without any normalization.

        :param data: List of colors. The lenght of the list should be equal to the number of element in the selected domain.
        :param name: Name of the new attribute.
        :param domain: Domain to which the point refers to. Can be one of POINT, FACE, EDGE, CORNER.

        """
        attribute = self.object.data.attributes.new(name=name, type='FLOAT_COLOR', domain=domain)
        attribute.data.foreach_set('color', data.ravel())

    def getMinZ(self):
        """
        Get the min Z value.

        :return: The min Z coordinate of the set of points.

        """
        minz = float('inf')
        for vert in self.data.vertices:
            wolrd_vert = self.object.matrix_basis @ vert.co
            if wolrd_vert[2] < minz:
                minz = wolrd_vert[2]
        return minz
    
    def getMaxZ(self):
        """
        Get the max Z value.

        :return: The max Z coordinate of the set of points.

        """
        maxz = -float('inf')
        for vert in self.data.vertices:
            wolrd_vert = self.object.matrix_basis @ vert.co
            if wolrd_vert[2] > maxz:
                maxz = wolrd_vert[2]
        return maxz
    
    def setLocation(self, location=(0, 0, 0)):
        """
        Set the location of the object with respect to the global reference frame.

        :param location: Vector of coordinates representing the new location of the object.

        """
        self.object.location = location

    def setRotation(self, rotation=(0, 0, 0)):
        """
        Set the rotation of the object with pivot point the origin of the local reference frame.

        :param rotation: Rotation of the mesh object in the local reference frame in radiants.

        """
        self.object.rotation_euler = rotation

    def setScale(self, scale=(1, 1, 1)):
        """
        Set the scale of the object in the global reference frame.

        :param scale: Scale on the xyz axes.

        """
        self.object.scale = scale

    def getFloor(self, size=(10,10), shadow_catcher=True):
        """
        Get a planar mesh object that acts as floor for the mesh object.

        :param size: Size of the floor in meters.
        :param shadow_catcher: Set the floor object visibility as shadow catcher.
        :return: The floor mesh.

        """
        minz = self.getMinZ()
        verices = [
            (-size[0]/2, -size[0]/2, 0),
            (-size[0]/2, size[0]/2, 0),
            (size[0]/2, size[0]/2, 0),
            (size[0]/2, -size[0]/2, 0),
        ]
        faces = [
            (0, 1, 2, 3)
        ]
        floor = Mesh("Floor", vertices=verices, faces=faces, location=(0,0, minz))
        floor.object.is_shadow_catcher = shadow_catcher
        return floor
    
    def getCamera(self, azimuth=np.pi/4, elevation=np.pi/9, distance=3):
        """
        Get a camera object focused on the origin of the local reference frame.

        :param azimuth: Azimuth angle in radiants.
        :param elevation: Elevation angle in radiants.
        :param distance: Distance from the focus point.
        :return: The camera object.

        """
        camera = Camera()
        camera.focusOnPoint(self.object.location, azimuth, elevation, distance)
        return camera

    def setMaterial(self, material:Material):
        """
        Link a material object to the mesh.

        :param material: Material object to link to the mesh.

        """
        self.material = material
        self.data.materials.clear()
        self.data.materials.append(material.data)

    def setShadeSmooth(self, shade_smooth=True):
        """
        Set smooth shading option of the mesh.

        :param shade_smooth: Boolean setting smooth shading.

        """
        if shade_smooth:
            self.data.shade_smooth()
        else:
            self.data.shade_flat()

    def asPointCloud(self, name="Pointcloud", radius=0.01, subdivison=3, material:Material=None):
        """
        Set rendering as pointcloud.

        :param name: Name of the Blender modifier created.
        :param radius: Radius of the rendered pointclouds.
        :param subdivision: Number of subdivisions of the icospheres representing the points.
        :param material: Material object to link to the point cloud.

        """
        if material is None and self.material is not None:
            material = self.material

        node_tree = MeshToPointCloudNodeTree(material=material, radius=radius, subdivison=subdivison)

        modifier = self.object.modifiers.new(name, 'NODES')
        modifier.node_group = node_tree.node_group