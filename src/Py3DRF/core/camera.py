import numpy as np
from .types import Location
from .types import Rotation
from .types import Scale

class Camera:
    """
    Camera class representing the Blender camera object.
    """

    def __init__(self, name="Camera", location=Location(0, 0, 0), rotation=Rotation(0, 0, 0), scale=Scale(1, 1, 1), focal_length=50, target=Location(0, 0, 0)):
        """

        :param name: Name of the camera object.
        :param location: Location of the camera object in the global reference frame.
        :param rotation: Rotation of the camera object in the global reference frame in radiants.
        :param scale: Scale of the camera object in the global reference frame.
        :param focal_length: Focal length of the camera object in mm.
        :param target: Point the camera looks at. Blender's camera model doesn't use this
            directly -- it points wherever `rotation` says, exactly like a real Blender
            camera object -- but backends whose native camera is instead defined as
            look_at(location, target) (Plotly, Polyscope) need an explicit target point,
            since it generally can't be recovered from `rotation` alone.

        """
        self.name = name
        self.location = location
        self.scale = scale
        self.rotation = rotation
        self.focal_length = focal_length
        self.target = target

    def setFocalLength(self, focal_length):
        """
        Set focal length of the camera object.

        :param focal_length: Focal length in mm.

        """
        self.focal_length = focal_length

    def setLocation(self, location :Location):
        """
        Set location of the camera object.

        :param location: Location of the camera object in the global reference frame.

        """
        self.location = location

    def setRotation(self, rotation :Rotation):
        """
        Set rotation of the camera object.

        :param rotation: Rotation of the camera object in the global reference frame in radiants.

        """
        self.rotation = rotation

    def setScale(self, scale :Scale):
        """
        Set scale of the camera object.

        :param scale: Scale of the camera object in the global reference frame.

        """
        self.scale = scale

    def setTarget(self, target :Location):
        """
        Set the point the camera looks at (see the `target` parameter of __init__).

        :param target: Point the camera looks at.

        """
        self.target = target

    def focusOnPoint(self, point :Location, azimuth=np.pi/4, elevation=np.pi/9, distance=3):
        """
        Reposition the camera to focus on a point.

        :param point: Point to focus on.
        :param azimuth: Azimuth angle in radiants.
        :param elevation: Elevation angle in radiants.
        :param distance: Distance from the point.

        """
        x = distance * np.cos(azimuth) * np.cos(elevation)
        y = distance * np.sin(azimuth) * np.cos(elevation)
        z = distance * np.sin(elevation)
        coords = np.array((x, y, z)) + point.to_array()
        rotation = Rotation(np.pi/2 - elevation, 0, azimuth + np.pi/2)
        self.setLocation(Location(*coords))
        self.setRotation(rotation)
        self.setTarget(point)