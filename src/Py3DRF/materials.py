"""
Material class describing the appearance of a renderable surface.

This module has no dependency on bpy (Blender's Python API). A Material only
stores the parameters needed to describe a material; turning a Material
instance into an actual Blender material (node tree, etc.) is the
responsibility of Scene (see scene.py), which is the only module in this
package allowed to import bpy.
"""


class Material:
    """
    Material class representing the appearance of a Principled-BSDF-style
    surface: a base color, an emission color/strength, a roughness value, and
    optional named geometry attributes that drive color/emission instead of
    the constant values above.
    """

    def __init__(
            self,
            name="Material",
            color=(1.0, 1.0, 1.0, 1.0),
            emission_color=(0.0, 0.0, 0.0, 1.0),
            roughness=0.75,
            emission_strenght=0.0,
            color_attribute=None,
            color_attribute_colors=None,
            emission_color_attribute=None,
            emission_color_attribute_colors=None,
            emission_strength_attribute=None
            ) -> None:
        """

        :param name: Name of the material.
        :param color: Base color of the material.
        :param emission_color: Emission color of the material.
        :param roughness: Roughness of the material.
        :param emission_strenght: Emission strength of the material.
        :param color_attribute: Name of the attribute to use as color. When color_attribute_colors
            is provided the attribute is treated as a float driving a color ramp, otherwise it is
            used directly as a color attribute.
        :param color_attribute_colors: Colors to use for the color ramp driven by color_attribute.
        :param emission_color_attribute: Name of the attribute to use as emission color.
        :param emission_color_attribute_colors: Colors to use for the color ramp driven by
            emission_color_attribute.
        :param emission_strength_attribute: Name of the attribute to use as emission strength.

        """
        self.name = name
        self.color = color
        self.roughness = roughness
        self.emission_strength = emission_strenght
        self.emission_color = emission_color

        self.color_attribute = None
        self.color_attribute_colors = None
        self.color_attribute_positions = None

        self.emission_color_attribute = None
        self.emission_color_attribute_colors = None
        self.emission_color_attribute_positions = None

        self.emission_strength_attribute = None

        if color_attribute is not None:
            if color_attribute_colors is not None:
                assert len(color_attribute_colors) > 1, "Color attribute colors must have at least 2 colors"
                self.setFloatAttributeAsColor(color_attribute, color_attribute_colors)
            else:
                self.setColorAttributeAsColor(color_attribute)

        if emission_color_attribute is not None:
            if emission_color_attribute_colors is not None:
                assert len(emission_color_attribute_colors) > 1, "Emission color attribute colors must have at least 2 colors"
                self.setFloatAttributeAsEmissionColor(emission_color_attribute, emission_color_attribute_colors)
            else:
                self.setColorAttributeAsEmissionColor(emission_color_attribute)

        if emission_strength_attribute is not None:
            self.setFloatAttributeAsEmissionStrength(emission_strength_attribute)

    def setColor(self, color):
        """
        Set base color of the material.

        :param color: Base color of the material.

        """
        self.color = color

    def setRoughness(self, roughness):
        """
        Set roughness of the material.

        :param roughness: Roughness value from 0 to 1, with 0 fully specular and 1 fully diffuse.

        """
        self.roughness = roughness

    def setEmissionStrength(self, emission_strenght):
        """
        Set emission strength.

        :param emission_strenght: Strength of the emitted light. 1 makes the object in the image
            exactly of the color set by the emission color.

        """
        self.emission_strength = emission_strenght

    def setEmissionColor(self, emission_color):
        """
        Set emission color.

        :param emission_color: Color of the light emission.

        """
        self.emission_color = emission_color

    def setColorAttributeAsColor(self, attribute_name):
        """
        Use a named color-type attribute directly as the base color, overriding setColor.

        :param attribute_name: Name of the color attribute.

        """
        self.color_attribute = attribute_name
        self.color_attribute_colors = None
        self.color_attribute_positions = None

    def setFloatAttributeAsColor(self, attribute_name, colors=[(0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0)], colors_positions=None):
        """
        Use a named float-type attribute to drive the base color through a linear color ramp,
        overriding setColor.

        :param attribute_name: Name of the float attribute.
        :param colors: List of colors (vec4) that represent the color ramp.
        :param colors_positions: List of floats that determine the mapping of each color to a float value.

        """
        self.color_attribute = attribute_name
        self.color_attribute_colors = list(colors)
        self.color_attribute_positions = list(colors_positions) if colors_positions else None

    def setColorAttributeAsEmissionColor(self, attribute_name):
        """
        Use a named color-type attribute directly as the emission color, overriding setEmissionColor.

        :param attribute_name: Name of the color attribute.

        """
        self.emission_color_attribute = attribute_name
        self.emission_color_attribute_colors = None
        self.emission_color_attribute_positions = None

    def setFloatAttributeAsEmissionColor(self, attribute_name, colors=[(0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0)], colors_positions=None):
        """
        Use a named float-type attribute to drive the emission color through a linear color ramp,
        overriding setEmissionColor.

        :param attribute_name: Name of the float attribute.
        :param colors: List of colors (vec4) that represent the color ramp.
        :param colors_positions: List of floats that determine the mapping of each color to a float value.

        """
        self.emission_color_attribute = attribute_name
        self.emission_color_attribute_colors = list(colors)
        self.emission_color_attribute_positions = list(colors_positions) if colors_positions else None

    def setFloatAttributeAsEmissionStrength(self, attribute_name):
        """
        Use a named float-type attribute to drive the emission strength, overriding setEmissionStrength.

        :param attribute_name: Name of the float attribute.

        """
        self.emission_strength_attribute = attribute_name

    def clearColorAttribute(self):
        """
        Stop driving the base color from an attribute; fall back to the constant color set by setColor.
        """
        self.color_attribute = None
        self.color_attribute_colors = None
        self.color_attribute_positions = None

    def clearEmissionColorAttribute(self):
        """
        Stop driving the emission color from an attribute; fall back to the constant color set by
        setEmissionColor.
        """
        self.emission_color_attribute = None
        self.emission_color_attribute_colors = None
        self.emission_color_attribute_positions = None

    def clearEmissionStrengthAttribute(self):
        """
        Stop driving the emission strength from an attribute; fall back to the constant value set by
        setEmissionStrength.
        """
        self.emission_strength_attribute = None
