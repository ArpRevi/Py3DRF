

class Material:
    """
    Material class representing the Blender material node tree.
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
        :param color_attribute: Name of the attribute to use as color. When the attribute is of type vector color_attribute_colors is expected. If not provided the color attribute will be considered a float attribute.k
        :param color_attribute_colors: Colors to use for the color attribute.
        :param emission_color_attribute: Name of the attribute to use as emission color.
        :param emission_color_attribute_colors: Colors to use for the emission color attribute.
        :param emission_strength_attribute: Name of the attribute to use as emission strength.

        """
        self.name = name
        self.nodes = {}
        self.links = {}

        """
        self.nodes['output'] = self.node_group.nodes.new(type="ShaderNodeOutputMaterial")
        self.nodes['principledBSDF'] = self.node_group.nodes.new(type="ShaderNodeBsdfPrincipled")
        
        self.links['principledBSDFToOutput'] = self.node_group.links.new(self.nodes['principledBSDF'].outputs['BSDF'], self.nodes['output'].inputs['Surface'])
        """

        self.color = color
        self.roughness = roughness
        self.emission_strength = emission_strenght
        self.emission_color = emission_color

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
        self.nodes['principledBSDF'].inputs['Base Color'].default_value = color

    def setRoughness(self, roughness):
        """
        Set roughness of the material.

        :param roughness: Roughness value from 0 to 1, with 0 fully specular and 1 fully diffuse.

        """
        self.nodes['principledBSDF'].inputs['Roughness'].default_value = roughness

    def setEmissionStrength(self, emission_strenght):
        """
        Set emission strength.

        :param emission_strenght: Strength of the emitted light. 1 makes the object in the image exactly of the color set by the emission color.

        """
        self.nodes['principledBSDF'].inputs['Emission Strength'].default_value = emission_strenght

    def setEmissionColor(self, emission_color):
        """
        Set emission color.

        :param emission_color: Color of the light emission.

        """
        self.nodes['principledBSDF'].inputs['Emission Color'].default_value = emission_color
    
    def setColorAttributeAsColor(self, attribute_name):
        """
        Link base color to named color attribute.

        :param attribute_name: Name of the color attribute.

        """
        self._removeAttributeAsColor()

        self.nodes['colorAttribute'] = self.node_group.nodes.new(type="ShaderNodeAttribute")
        self.links['colorAttributeToPrincipledBSDF'] = self.node_group.links.new(self.nodes['colorAttribute'].outputs['Color'], self.nodes['principledBSDF'].inputs['Base Color'])

        self.nodes['colorAttribute'].attribute_name = attribute_name
    """
    def setFloatAttributeAsColor(self, attribute_name, colors = [(0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0)], colors_positions = None):
        """
        Link base color to named float attribute. The interpolation between colors is linear.

        :param attribute_name: Name of the float attribute.
        :param colors: List of colors (vec4) that represent the color ramp.
        :param colors_positions: List of floats that determine the mapping of each color to a float value.

        """
        self._removeAttributeAsColor()

        self.nodes['colorAttribute'] = self.node_group.nodes.new(type="ShaderNodeAttribute")
        self.nodes['colorRamp'] = self.node_group.nodes.new(type="ShaderNodeValToRGB")
        self.links['colorAttributeToColorRamp'] = self.node_group.links.new(self.nodes['colorAttribute'].outputs['Fac'], self.nodes['colorRamp'].inputs['Fac'])
        self.links['colorRampToPrincipledBSDF'] = self.node_group.links.new(self.nodes['colorRamp'].outputs['Color'], self.nodes['principledBSDF'].inputs['Base Color'])

        self.nodes['colorAttribute'].attribute_name = attribute_name

        for i, c in enumerate(colors):
            if i == 0:
                self.nodes['colorRamp'].color_ramp.elements[i].color = c
                self.nodes['colorRamp'].color_ramp.elements[-1].color = colors[-1]
                if colors_positions:
                    self.nodes['colorRamp'].color_ramp.elements[i].position = colors_positions[0]
                    self.nodes['colorRamp'].color_ramp.elements[-1].position = colors_positions[-1]
            elif i == len(colors)-1:
                break
            else:
                elem = self.nodes['colorRamp'].color_ramp.elements.new(i * 1/(len(colors)-1))
                elem.color = c
                if colors_positions:
                    elem.position = colors_positions[i]
    """
    """
    def setColorAttributeAsEmissionColor(self, attribute_name):
        """
        Link emission color to named float attribute.

        :param attribute_name: Name of the color attribute.

        """
        self._removeAttributeAsEmissionColor()

        self.nodes['emissionColorAttribute'] = self.node_group.nodes.new(type="ShaderNodeAttribute")
        self.links['emissionColorAttributeToPrincipledBSDF'] = self.node_group.links.new(self.nodes['emissionColorAttribute'].outputs['Color'], self.nodes['principledBSDF'].inputs['Emission Color'])

        self.nodes['emissionColorAttribute'].attribute_name = attribute_name
    """
    """
    def setFloatAttributeAsEmissionColor(self, attribute_name, colors = [(0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0)], colors_positions = None):
        """
        Link emission color to named float attribute. The interpolation between colors is linear.

        :param attribute_name: Name of the color attribute.
        :param colors: List of colors (vec4) that represent the color ramp.
        :param colors_positions: List of floats that determine the mapping of each color to a float value.

        """
        self._removeAttributeAsEmissionColor()

        self.nodes['emissionColorAttribute'] = self.node_group.nodes.new(type="ShaderNodeAttribute")
        self.nodes['emissionColorRamp'] = self.node_group.nodes.new(type="ShaderNodeValToRGB")
        self.links['emissionColorAttributeToEmissionColorRamp'] = self.node_group.links.new(self.nodes['emissionColorAttribute'].outputs['Fac'], self.nodes['emissionColorRamp'].inputs['Fac'])
        self.links['emissionColorRampToPrincipledBSDF'] = self.node_group.links.new(self.nodes['emissionColorRamp'].outputs['Color'], self.nodes['principledBSDF'].inputs['Emission Color'])
        
        self.nodes['emissionColorAttribute'].attribute_name = attribute_name

        for i, c in enumerate(colors):
            if i == 0:
                self.nodes['emissionColorRamp'].color_ramp.elements[i].color = c
                self.nodes['emissionColorRamp'].color_ramp.elements[-1].color = colors[-1]
                if colors_positions:
                    self.nodes['emissionColorRamp'].color_ramp.elements[i].position = colors_positions[0]
                    self.nodes['emissionColorRamp'].color_ramp.elements[-1].position = colors_positions[-1]
            elif i == len(colors)-1:
                break
            else:
                elem = self.nodes['emissionColorRamp'].color_ramp.elements.new(i * 1/(len(colors)-1))
                elem.color = c
                if colors_positions:
                    elem.position = colors_positions[i]

    def setFloatAttributeAsEmissionStrength(self, attribute_name):
        """
        Link emission strength to named float attribute.

        :param attribute_name: Name of the color attribute.

        """
        self._removeAttributeAsEmissionStrength()

        self.nodes['emissionStrengthAttribute'] = self.node_group.nodes.new(type="ShaderNodeAttribute")
        self.links['emissionStrengthAttributeToPrincipledBSDF'] = self.node_group.links.new(self.nodes['emissionStrengthAttribute'].outputs['Fac'], self.nodes['principledBSDF'].inputs['Emission Strength'])

        self.nodes['emissionStrengthAttribute'].attribute_name = attribute_name

    def _removeAttributeAsColor(self):
        self._removeLink('colorAttributeToPrincipledBSDF')
        self._removeLink('colorAttributeToColorRamp')
        self._removeLink('colorRampToPrincipledBSDF')
        self._removeNode('colorAttribute')
        self._removeNode('colorRamp')

    def _removeAttributeAsEmissionColor(self):
        self._removeLink('emissionColorAttributeToPrincipledBSDF')
        self._removeLink('emissionColorAttributeToEmissionColorRamp')
        self._removeLink('emissionColorRampToPrincipledBSDF')
        self._removeNode('emissionColorAttribute')
        self._removeNode('emissionColorRamp')

    def _removeAttributeAsEmissionStrength(self):
        self._removeLink('emissionStrengthAttributeToPrincipledBSDF')
        self._removeNode('emissionStrengthAttribute')

    def _removeLink(self, link_name):
        if link_name in self.links.keys():
            self.node_group.links.remove(self.links[link_name])
            self.links.pop(link_name)

    def _removeNode(self, node_name):
        if node_name in self.nodes.keys():
            self.node_group.nodes.remove(self.nodes[node_name])
            self.nodes.pop(node_name)
    """