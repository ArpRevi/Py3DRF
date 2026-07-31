import logging

import numpy as np

from .materials import Material
from .camera import Camera
from .pointcloud import PointCloudSettings
from .wireframe import WireFrameSettings
from .types import Location
from .types import Rotation
from .types import Scale

logger = logging.getLogger(__name__)

class Mesh:
    """
    Mesh class representing the geometry, placement and appearance of a renderable mesh object.

    This module (Py3DRF.core) has no dependency on any rendering backend. Building the
    actual backend-native mesh object from a Mesh instance is the responsibility of
    whichever backend the user chose via the Scene facade.
    """

    def __init__(
            self,
            name="Mesh",
            vertices=[],
            edges=[],
            faces=[],
            location=Location(0, 0, 0),
            rotation=Rotation(0, 0, 0),
            scale=Scale(1, 1, 1),
            material: Material = None,
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
        self.name = name
        self.vertices = list(vertices)
        self.edges = list(edges)
        self.faces = list(faces)

        self.location = location
        self.rotation = rotation
        self.scale = scale
        self.float_attributes = {}
        self.color_attributes = {}
        self.point_cloud_settings = None
        self.wireframe_settings = None
        self.is_shadow_catcher = False

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
        self.float_attributes[name] = (np.asarray(data), domain)

    def addColorAttribute(self, data, name="value", domain="POINT"):
        """
        Add a color attribute to the mesh. The values are added without any normalization.

        :param data: List of colors. The lenght of the list should be equal to the number of element in the selected domain.
        :param name: Name of the new attribute.
        :param domain: Domain to which the point refers to. Can be one of POINT, FACE, EDGE, CORNER.

        """
        self.color_attributes[name] = (np.asarray(data), domain)

    def _localToWorldMatrix(self):
        """
        Build the 4x4 local-to-world transform matrix (translation * rotation * scale), matching
        Blender's object.matrix_basis for the default 'XYZ' Euler rotation order.

        :return: 4x4 numpy array.

        """
        rx, ry, rz = self.rotation.to_tuple()
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)

        rotation_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        rotation_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rotation_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

        rotation_matrix = rotation_z @ rotation_y @ rotation_x
        scale_matrix = np.diag(self.scale.to_tuple())

        matrix = np.eye(4)
        matrix[:3, :3] = rotation_matrix @ scale_matrix
        matrix[:3, 3] = self.location.to_tuple()
        return matrix

    def _worldVertices(self):
        """
        :return: Vertices in the global reference frame, as an (n, 3) numpy array.
        """
        if len(self.vertices) == 0:
            return np.zeros((0, 3))

        matrix = self._localToWorldMatrix()
        local = np.asarray(self.vertices, dtype=float)
        homogeneous = np.hstack([local, np.ones((len(local), 1))])
        world = (matrix @ homogeneous.T).T
        return world[:, :3]

    def _edgePairs(self):
        """
        :return: (E, 2) int numpy array of unique undirected vertex-index edges: self.edges
            if explicitly provided, otherwise inferred from self.faces (each face's
            consecutive vertex pairs, wrapping around, deduplicated).
        """
        if len(self.edges) > 0:
            return np.asarray(self.edges, dtype=int)

        pairs = set()
        for face in self.faces:
            n = len(face)
            for k in range(n):
                a, b = int(face[k]), int(face[(k + 1) % n])
                pairs.add((a, b) if a < b else (b, a))

        if not pairs:
            return np.zeros((0, 2), dtype=int)
        return np.array(sorted(pairs), dtype=int)

    def getMinZ(self):
        """
        Get the min Z value.

        :return: The min Z coordinate of the set of points.

        """
        world = self._worldVertices()
        if len(world) == 0:
            return float('inf')
        return float(world[:, 2].min())

    def getMaxZ(self):
        """
        Get the max Z value.

        :return: The max Z coordinate of the set of points.

        """
        world = self._worldVertices()
        if len(world) == 0:
            return -float('inf')
        return float(world[:, 2].max())

    def getMinX(self):
        """
        Get the min X value.

        :return: The min X coordinate of the set of points.

        """
        world = self._worldVertices()
        if len(world) == 0:
            return float('inf')
        return float(world[:, 0].min())

    def getMaxX(self):
        """
        Get the max X value.

        :return: The max X coordinate of the set of points.

        """
        world = self._worldVertices()
        if len(world) == 0:
            return -float('inf')
        return float(world[:, 0].max())

    def getMinY(self):
        """
        Get the min Y value.

        :return: The min Y coordinate of the set of points.

        """
        world = self._worldVertices()
        if len(world) == 0:
            return float('inf')
        return float(world[:, 1].min())

    def getMaxY(self):
        """
        Get the max Y value.

        :return: The max Y coordinate of the set of points.

        """
        world = self._worldVertices()
        if len(world) == 0:
            return -float('inf')
        return float(world[:, 1].max())

    def setLocation(self, location=Location(0, 0, 0)):
        """
        Set the location of the object with respect to the global reference frame.

        :param location: Vector of coordinates representing the new location of the object.

        """
        self.location = location.to_array()
    def setRotation(self, rotation=Rotation(0, 0, 0)):
        """
        Set the rotation of the object with pivot point the origin of the local reference frame.

        :param rotation: Rotation of the mesh object in the local reference frame in radiants.

        """
        self.rotation = rotation

    def setScale(self, scale=Scale(1, 1, 1)):
        """
        Set the scale of the object in the global reference frame.

        :param scale: Scale on the xyz axes.

        """
        self.scale = scale

    def getFloor(self, size=(10, 10), shadow_catcher=True):
        """
        Get a planar mesh object that acts as floor for the mesh object.

        :param size: Size of the floor in meters.
        :param shadow_catcher: Set the floor object visibility as shadow catcher.
        :return: The floor mesh.

        """
        minz = self.getMinZ()
        vertices = [
            (-size[0]/2, -size[1]/2, 0),
            (-size[0]/2, size[1]/2, 0),
            (size[0]/2, size[1]/2, 0),
            (size[0]/2, -size[1]/2, 0),
        ]
        faces = [
            (0, 1, 2, 3)
        ]
        floor = Mesh("Floor", vertices=vertices, faces=faces, location=Location(0, 0, minz))
        floor.is_shadow_catcher = shadow_catcher
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
        camera.focusOnPoint(Location((self.getMinX()+self.getMaxX())/2, (self.getMinY()+self.getMaxY())/2, (self.getMinZ()+self.getMaxZ())/2), azimuth, elevation, distance)
        return camera

    def setMaterial(self, material: Material):
        """
        Link a material object to the mesh.

        :param material: Material object to link to the mesh.

        """
        self.material = material

    def setShadeSmooth(self, shade_smooth=True):
        """
        Set smooth shading option of the mesh.

        :param shade_smooth: Boolean setting smooth shading.

        """
        self.shade_smooth = shade_smooth

    def asPointCloud(self, name="Pointcloud", radius=0.01, subdivison=3, material: Material = None):
        """
        Set rendering as pointcloud. Point cloud and wireframe rendering are mutually
        exclusive: this clears any wireframe_settings previously applied via asWireframe(),
        since no backend can meaningfully render both at once.

        :param name: Name of the Blender modifier created.
        :param radius: Radius of the rendered pointclouds.
        :param subdivision: Number of subdivisions of the icospheres representing the points.
        :param material: Material object to link to the point cloud.
        :return: The PointCloudSettings describing the point cloud rendering.

        """
        if material is None and self.material is not None:
            material = self.material

        self.point_cloud_settings = PointCloudSettings(name=name, radius=radius, subdivison=subdivison, material=material)
        if self.wireframe_settings is not None:
            # point cloud and wireframe are mutually exclusive: clear the wireframe settings
            logger.warning("Mesh.asPointCloud() called after Mesh.asWireframe(); clearing wireframe settings.")
        self.wireframe_settings = None
        return self.point_cloud_settings

    def asWireframe(self, name="Wireframe", thickness=0.01, resolution=12, material: Material = None, hide_surface=True):
        """
        Set rendering as wireframe: the mesh's faces are hidden and each of its edges is
        highlighted -- as a 3D cylinder or a flat screen-space line, depending on the
        backend (see Py3DRF.core.wireframe.WireFrameSettings). Point cloud and wireframe
        rendering are mutually exclusive: this clears any point_cloud_settings previously
        applied via asPointCloud(), since no backend can meaningfully render both at once.

        :param name: Name of the Blender modifier, or the Plotly/Polyscope trace/structure,
            created.
        :param thickness: Girth of the wireframe edges: a real 3D diameter on Blender, a
            heuristically-scaled pixel line width on Plotly/Polyscope.
        :param resolution: Number of sides of the edge cylinders. Blender-only.
        :param material: Material object to link to the wireframe.
        :param hide_surface: If True (default), only the wireframe edges are rendered; if
            False, the solid shaded mesh is rendered as well, with the edges on top.
            Polyscope cannot honor True and raises NotSupportedByBackendError instead.
        :return: The WireframeSettings describing the wireframe rendering.

        """
        if material is None and self.material is not None:
            material = self.material

        self.wireframe_settings = WireFrameSettings(name=name, thickness=thickness, resolution=resolution, material=material, hide_surface=hide_surface)
        if self.point_cloud_settings is not None:
            # point cloud and wireframe are mutually exclusive: clear the point cloud settings
            logger.warning("Mesh.asWireframe() called after Mesh.asPointCloud(); clearing point cloud settings.")
        self.point_cloud_settings = None
        return self.wireframe_settings
