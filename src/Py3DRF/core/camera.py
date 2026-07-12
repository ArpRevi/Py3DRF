import numpy as np

class Camera:
    """
    Camera class representing the Blender camera object.
    """

    def __init__(self, name="Camera", location=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1), focal_length=50):
        """

        :param name: Name of the camera object.
        :param location: Location of the camera object in the global reference frame.
        :param rotation: Rotation of the camera object in the global reference frame in radiants.
        :param scale: Scale of the camera object in the global reference frame.
        :param focal_length: Focal length of the camera object in mm.

        """
        self.name = name
        self.location = location
        self.scale = scale
        self.rotation = rotation
        self.focal_length = focal_length
    
    def setFocalLength(self, focal_length):
        """
        Set focal length of the camera object.

        :param focal_length: Focal length in mm.

        """
        self.focal_length = focal_length

    def setLocation(self, location):
        """
        Set location of the camera object.

        :param location: Location of the camera object in the global reference frame.

        """
        self.location = location

    def setRotation(self, rotation):
        """
        Set rotation of the camera object.

        :param rotation: Rotation of the camera object in the global reference frame in radiants.

        """
        self.rotation = rotation

    def setScale(self, scale):
        """
        Set scale of the camera object.

        :param scale: Scale of the camera object in the global reference frame.

        """
        self.scale = scale

    def focusOnPoint(self, point, azimuth=np.pi/4, elevation=np.pi/9, distance=3):
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
        coords = np.array((x, y, z)) + point
        rotation = (np.pi/2 - elevation, 0, azimuth + np.pi/2)
        self.setLocation(coords)
        self.setRotation(rotation)