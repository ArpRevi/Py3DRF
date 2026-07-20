import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class Rotation:
    """
    Represents a rotation in 3D space with roll, pitch, and yaw angles.
    """

    def __init__(self, roll: float, pitch: float, yaw: float):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

    def __repr__(self):
        return f"Rotation(roll={self.roll}, pitch={self.pitch}, yaw={self.yaw})"

    def to_array(self):
        """
        Convert the Rotation to a numpy array.

        :return: A numpy array of shape (3,) representing the rotation angles.
        """
        return np.array([self.roll, self.pitch, self.yaw], dtype=float)

    def to_tuple(self):
        """
        Convert the Rotation to a tuple.

        :return: A tuple of (roll, pitch, yaw) angles.
        """
        return (self.roll, self.pitch, self.yaw)

@dataclass(frozen=True)
class Scale:
    """
    Represents a scale in 3D space with x, y, and z scaling factors.
    """

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Scale(x={self.x}, y={self.y}, z={self.z})"

    def to_array(self):
        """
        Convert the Scale to a numpy array.

        :return: A numpy array of shape (3,) representing the scaling factors.
        """
        return np.array([self.x, self.y, self.z], dtype=float)

    def to_tuple(self):
        """
        Convert the Scale to a tuple.

        :return: A tuple of (x, y, z) scaling factors.
        """
        return (self.x, self.y, self.z)
