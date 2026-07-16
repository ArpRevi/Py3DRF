import numpy as np



class Location:
    """
    Represents a location in 3D space with x, y, and z coordinates.
    """

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Location(x={self.x}, y={self.y}, z={self.z})"

    def to_array(self):
        """
        Convert the Location to a numpy array.

        :return: A numpy array of shape (3,) representing the location.
        """
        return np.array([self.x, self.y, self.z], dtype=float)

    def to_tuple(self):
        """
        Convert the Location to a tuple.

        :return: A tuple of (x, y, z) coordinates.
        """
        return (self.x, self.y, self.z)