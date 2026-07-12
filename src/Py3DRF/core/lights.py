"""
SunLight class describing the placement and strength of a directional (sun) light.

This module (Py3DRF.core) has no dependency on any rendering backend. Building the actual
backend-native light object from a SunLight instance is the responsibility of the chosen
backend (e.g. Py3DRF.backends.blender.scene.SceneBlender), never of core itself.
"""


class SunLight:
    """
    SunLight class representing a directional (sun) light's placement and strength.
    """

    def __init__(self, name="Sun", location=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1), strength=5.0):
        """

        :param name: Name of the sun light object.
        :param location: Location of the sun light object in the global reference frame.
        :param rotation: Rotation of the sun light object in the global reference frame in radiants.
        :param scale: Scale of the sun light object in the global reference frame.
        :param strength: Strength of the sun light object.

        """
        self.name = name
        self.location = location
        self.rotation = rotation
        self.scale = scale
        self.strength = strength

    def setLocation(self, location):
        """
        Set location of the sun light object.

        :param location: Location of the sun light object in the global reference frame.

        """
        self.location = location

    def setRotation(self, rotation):
        """
        Set rotation of the sun light object.

        :param rotation: Rotation of the sun light object in the global reference frame in radiants.

        """
        self.rotation = rotation

    def setScale(self, scale):
        """
        Set scale of the sun light object.

        :param scale: Scale of the sun light object in the global reference frame.

        """
        self.scale = scale

    def setStrength(self, strength):
        """
        Set strength of the sun light object.

        :param strength: Strength of the sun light object.

        """
        self.strength = strength