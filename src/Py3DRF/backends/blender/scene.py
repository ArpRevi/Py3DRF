"""
SceneBlender: the Blender (bpy) backend for Py3DRF.

This is the only module in Py3DRF.backends.blender that imports bpy. It
translates Py3DRF.core objects (Camera, Mesh, SunLight, Material,
PointCloudSettings, WireFrameSettings) into actual Blender data-blocks.

Do not import this module directly -- go through the Scene facade
(Py3DRF.Scene(backend="blender", ...)), which loads it lazily.
"""

import bpy
import numpy as np

from ...core.camera import Camera
from ...core.lights import SunLight
from ...core.materials import Material
from ...core.mesh import Mesh
from ...core.pointcloud import PointCloudSettings
from ...core.wireframe import WireFrameSettings
from ..base import SceneBackend
from ...core.types import Location, Rotation, Scale


class SceneBlender(SceneBackend):

    def __init__(self, name="Scene", engine="CYCLES", device="GPU", resolution=(1920, 1080), transparent=True, deafult_sun=False, gamma=1, exposure=0, shadow_catcher_alpha=1.0) -> None:
        self._material_cache = {}

        self.data = bpy.data.scenes.new(name)
        self.setRenderEngine(engine=engine)
        self.setDevice(device=device)
        self.setResolution(resolution)
        self.data.render.film_transparent = transparent
        self.data.view_settings.view_transform = "Raw"

        self.data.view_layers[0].cycles.use_pass_shadow_catcher = True

        # Enable compositor rendering
        self.data.render.use_compositing = True

        # Create a new, modern Blender 5.0+ compositor node tree data-block
        comp_tree = bpy.data.node_groups.new(name + "_Compositor", "CompositorNodeTree")
        self.data.compositing_node_group = comp_tree

        # Instantiate Input and Output nodes
        self.nodes = {
            "input": comp_tree.nodes.new(type="CompositorNodeRLayers"),
            "output": comp_tree.nodes.new(type="NodeGroupOutput")
        }

        # comp_tree was just created above, so it never already has an output socket.
        # (NodeTreeInterface has no .inputs/.outputs attribute to check against -- only
        # .items_tree and .new_socket().)
        comp_tree.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")

        comp_tree.links.clear()
        self.links = {}

        self.nodes['input'].scene = self.data

        # Instantiate unified utility node classes
        self.nodes['invert'] = comp_tree.nodes.new("CompositorNodeInvert")
        self.nodes['add'] = comp_tree.nodes.new("ShaderNodeMix")
        self.nodes['setAlpha'] = comp_tree.nodes.new("CompositorNodeSetAlpha")

        self.nodes['add'].data_type = 'RGBA'
        self.nodes['add'].blend_type = 'ADD'
        # TODO: CompositorNodeSetAlpha's `mode` enum property was removed in Blender 5.0
        # (options-as-sockets migration). The replacement input socket name/value is
        # unconfirmed -- see conversation. Currently using the node's default mode.
        # self.nodes['setAlpha'].mode = 'REPLACE_ALPHA'

        # Link up the updated composite pipeline (Using sockets 'A', 'B', and 'Result')
        self.links['inputToInvert'] = comp_tree.links.new(self.nodes["input"].outputs['Shadow Catcher'], self.nodes["invert"].inputs["Color"])
        self.links['inputToAdd'] = comp_tree.links.new(self.nodes['input'].outputs['Alpha'], self.nodes['add'].inputs['A'])
        self.links['invertToAdd'] = comp_tree.links.new(self.nodes['invert'].outputs['Color'], self.nodes['add'].inputs['B'])
        self.links['inputToSetAlpha'] = comp_tree.links.new(self.nodes['input'].outputs['Image'], self.nodes['setAlpha'].inputs['Image'])
        self.links['addToSetAlpha'] = comp_tree.links.new(self.nodes['add'].outputs['Result'], self.nodes['setAlpha'].inputs['Alpha'])
        self.links['setAlphaToOutput'] = comp_tree.links.new(self.nodes['setAlpha'].outputs['Image'], self.nodes['output'].inputs['Image'])

        self.setShadowCatcherAlpha(shadow_catcher_alpha)
        self.setGamma(gamma)
        self.setExposure(exposure)

        world = bpy.data.worlds.new("World")
        self.data.world = world

        if deafult_sun:
            sun = SunLight(rotation=Rotation(-30 * np.pi/180, 0, -10 * np.pi/180))
            self.addObject(sun)

    def setRenderEngine(self, engine):
        """
        Set Blender rendering engine.

        :param engine: Engine name, either CYCLES or EEVEE

        """
        self.data.render.engine = engine

    def setDevice(self, device):
        """
        Set rendering device.

        :param device: Device to use when using CYCLES rendering engine, either GPU or CPU

        """
        self.data.cycles.device = device

    def setResolution(self, resolution):
        """
        Set rendering resolution.

        :param resolution: Size of the output image.

        """
        self.data.render.resolution_x = resolution[0]
        self.data.render.resolution_y = resolution[1]

    def setSamples(self, samples):
        """
        Set rendering samples.

        :param samples: Number of rendering samples.

        """
        self.data.cycles.samples = samples
        self.data.eevee.taa_render_samples = samples

    def addCamera(self, camera: Camera):
        """
        Build the Blender camera data-block and object for a Camera, link it to the scene and
        set it as the active camera.

        :param camera: Camera object.
        :return: The linked bpy.types.Object wrapping the camera.

        """
        obj = self._buildCameraObject(camera)
        self.data.collection.objects.link(obj)
        self.data.camera = obj
        return obj

    def addObject(self, object):
        """
        Build the Blender data-block(s) and object for a Mesh or SunLight, and link it to the
        scene.

        :param object: Mesh or SunLight object to add to the scene.
        :return: The linked bpy.types.Object.

        """
        if isinstance(object, Mesh):
            obj = self._buildMeshObject(object)
        elif isinstance(object, SunLight):
            obj = self._buildLightObject(object)
        else:
            raise TypeError(f"Cannot add object of type {type(object).__name__} to the scene.")

        self.data.collection.objects.link(obj)
        return obj

    def setGamma(self, gamma):
        """
        Set gamma correction of the image.

        :param gamma: Gamma value to use.

        """
        self.data.view_settings.gamma = gamma

    def setExposure(self, exposure):
        """
        Set rendering exposure.

        :param exposure: Exposure value to use.

        """
        self.data.view_settings.exposure = exposure

    def setShadowCatcherAlpha(self, alpha):
        """
        Set alpha value of the shadow catcher.

        :param alpha: Alpha value to use.

        """
        # ShaderNodeMix maps the mixing factor configuration to 'Factor' instead of 'Fac'
        self.nodes['add'].inputs['Factor'].default_value = alpha

    def renderToFile(self, filepath):
        """
        Render the scene and write the result to filepath.

        :param filepath: Destination filepath for the rendered image.

        """
        self.data.render.filepath = filepath
        bpy.ops.render.render(write_still=True, scene=self.data.name)

    def saveToFile(self, filename):
        """
        Save the current Blender file.

        :param filename: Destination filepath for the saved .blend file.

        """
        bpy.ops.wm.save_mainfile(filepath=filename)

    # -- Builders: translate plain-Python Py3DRF objects into bpy data-blocks --

    def _buildCameraObject(self, camera: Camera):
        """
        Build a bpy.types.Object wrapping a bpy.types.Camera from a Camera.
        """
        data = bpy.data.cameras.new(name=camera.name)
        data.lens = camera.focal_length

        obj = bpy.data.objects.new(camera.name, data)
        obj.location = camera.location.to_tuple()
        obj.rotation_euler = camera.rotation.to_tuple()
        obj.scale = camera.scale.to_tuple()
        return obj

    def _buildLightObject(self, light: SunLight):
        """
        Build a bpy.types.Object wrapping a bpy.types.Light (SUN) from a SunLight.
        """
        data = bpy.data.lights.new(name=light.name, type='SUN')
        data.energy = light.strength

        obj = bpy.data.objects.new(light.name, data)
        obj.location = light.location.to_tuple()
        obj.rotation_euler = light.rotation.to_tuple()
        obj.scale = light.scale.to_tuple()
        return obj

    def _buildMeshObject(self, mesh: Mesh):
        """
        Build a bpy.types.Object wrapping a bpy.types.Mesh from a Mesh, including its
        attributes, material, shading and optional point-cloud/wireframe modifier.
        """
        data = bpy.data.meshes.new(name=mesh.name)
        data.from_pydata(mesh.vertices, mesh.edges, mesh.faces)
        data.update()
        data.validate()

        for attr_name, (values, domain) in mesh.float_attributes.items():
            attribute = data.attributes.new(name=attr_name, type='FLOAT', domain=domain)
            attribute.data.foreach_set('value', np.asarray(values).ravel())

        for attr_name, (values, domain) in mesh.color_attributes.items():
            attribute = data.attributes.new(name=attr_name, type='FLOAT_COLOR', domain=domain)
            attribute.data.foreach_set('color', np.asarray(values).ravel())

        bl_material = self._buildMaterial(mesh.material)
        data.materials.clear()
        data.materials.append(bl_material)

        if mesh.shade_smooth:
            data.shade_smooth()
        else:
            data.shade_flat()

        obj = bpy.data.objects.new(mesh.name, data)
        obj.location = mesh.location.to_tuple()
        obj.rotation_euler = mesh.rotation.to_tuple()
        obj.scale = mesh.scale.to_tuple()
        obj.is_shadow_catcher = mesh.is_shadow_catcher

        if mesh.point_cloud_settings is not None:
            node_tree = self._buildPointCloudNodeTree(mesh.point_cloud_settings)
            modifier = obj.modifiers.new(mesh.point_cloud_settings.name, 'NODES')
            modifier.node_group = node_tree

        if mesh.wireframe_settings is not None:
            node_tree = self._buildWireframeNodeTree(mesh.wireframe_settings)
            modifier = obj.modifiers.new(mesh.wireframe_settings.name, 'NODES')
            modifier.node_group = node_tree

        return obj

    def _buildMaterial(self, material: Material):
        """
        Build (or fetch from cache) the bpy.types.Material corresponding to a Material,
        wiring up a Principled BSDF node and any attribute-driven inputs it describes.
        """
        cached = self._material_cache.get(id(material))
        if cached is not None:
            return cached

        bl_material = bpy.data.materials.new(name=material.name)
        bl_material.use_nodes = True
        node_tree = bl_material.node_tree
        node_tree.nodes.clear()

        nodes = {}
        links = {}

        nodes['output'] = node_tree.nodes.new(type="ShaderNodeOutputMaterial")
        nodes['principledBSDF'] = node_tree.nodes.new(type="ShaderNodeBsdfPrincipled")
        links['principledBSDFToOutput'] = node_tree.links.new(nodes['principledBSDF'].outputs['BSDF'], nodes['output'].inputs['Surface'])

        nodes['principledBSDF'].inputs['Base Color'].default_value = material.color
        nodes['principledBSDF'].inputs['Roughness'].default_value = material.roughness
        nodes['principledBSDF'].inputs['Emission Color'].default_value = material.emission_color
        nodes['principledBSDF'].inputs['Emission Strength'].default_value = material.emission_strength

        if material.color_attribute is not None:
            self._wireAttribute(
                node_tree, nodes, links,
                attribute_name=material.color_attribute,
                colors=material.color_attribute_colors,
                colors_positions=material.color_attribute_positions,
                prefix='color',
                target_node='principledBSDF', target_socket='Base Color'
            )

        if material.emission_color_attribute is not None:
            self._wireAttribute(
                node_tree, nodes, links,
                attribute_name=material.emission_color_attribute,
                colors=material.emission_color_attribute_colors,
                colors_positions=material.emission_color_attribute_positions,
                prefix='emissionColor',
                target_node='principledBSDF', target_socket='Emission Color'
            )

        if material.emission_strength_attribute is not None:
            nodes['emissionStrengthAttribute'] = node_tree.nodes.new(type="ShaderNodeAttribute")
            nodes['emissionStrengthAttribute'].attribute_name = material.emission_strength_attribute
            links['emissionStrengthAttributeToPrincipledBSDF'] = node_tree.links.new(
                nodes['emissionStrengthAttribute'].outputs['Fac'],
                nodes['principledBSDF'].inputs['Emission Strength']
            )

        self._material_cache[id(material)] = bl_material
        return bl_material

    def _wireAttribute(self, node_tree, nodes, links, attribute_name, colors, colors_positions, prefix, target_node, target_socket):
        """
        Wire a named geometry attribute into a shader node input: either directly (when colors
        is None, i.e. attribute_name refers to a color-type attribute) or through a color ramp
        (when colors is provided, i.e. attribute_name refers to a float-type attribute driving a
        color gradient).
        """
        attr_key = f'{prefix}Attribute'
        nodes[attr_key] = node_tree.nodes.new(type="ShaderNodeAttribute")
        nodes[attr_key].attribute_name = attribute_name

        if colors is None:
            links[f'{attr_key}To{target_node}'] = node_tree.links.new(nodes[attr_key].outputs['Color'], nodes[target_node].inputs[target_socket])
            return

        ramp_key = f'{prefix}Ramp'
        nodes[ramp_key] = node_tree.nodes.new(type="ShaderNodeValToRGB")
        links[f'{attr_key}To{ramp_key}'] = node_tree.links.new(nodes[attr_key].outputs['Fac'], nodes[ramp_key].inputs['Fac'])
        links[f'{ramp_key}To{target_node}'] = node_tree.links.new(nodes[ramp_key].outputs['Color'], nodes[target_node].inputs[target_socket])

        for i, c in enumerate(colors):
            if i == 0:
                nodes[ramp_key].color_ramp.elements[i].color = c
                nodes[ramp_key].color_ramp.elements[-1].color = colors[-1]
                if colors_positions:
                    nodes[ramp_key].color_ramp.elements[i].position = colors_positions[0]
                    nodes[ramp_key].color_ramp.elements[-1].position = colors_positions[-1]
            elif i == len(colors) - 1:
                break
            else:
                elem = nodes[ramp_key].color_ramp.elements.new(i * 1 / (len(colors) - 1))
                elem.color = c
                if colors_positions:
                    elem.position = colors_positions[i]

    def _buildPointCloudNodeTree(self, settings: PointCloudSettings):
        """
        Build the bpy.types.GeometryNodeTree (Mesh to Points -> Instance icospheres on points)
        corresponding to a PointCloudSettings.
        """
        node_group = bpy.data.node_groups.new(type="GeometryNodeTree", name=settings.name)
        node_group.nodes.clear()
        nodes = {}
        links = {}

        nodes['input'] = node_group.nodes.new(type="NodeGroupInput")
        nodes['output'] = node_group.nodes.new(type="NodeGroupOutput")
        nodes['meshToPoints'] = node_group.nodes.new(type="GeometryNodeMeshToPoints")
        nodes['instanceOnPoints'] = node_group.nodes.new(type="GeometryNodeInstanceOnPoints")
        nodes['icoSphere'] = node_group.nodes.new(type="GeometryNodeMeshIcoSphere")
        nodes['setShadeSmooth'] = node_group.nodes.new(type="GeometryNodeSetShadeSmooth")
        nodes['setMaterial'] = node_group.nodes.new(type="GeometryNodeSetMaterial")
        nodes['realizeInstances'] = node_group.nodes.new(type="GeometryNodeRealizeInstances")

        node_group.interface.new_socket(name="Geometry", description="", in_out="INPUT", socket_type="NodeSocketGeometry")
        node_group.interface.new_socket(name="Geometry", description="", in_out="OUTPUT", socket_type="NodeSocketGeometry")

        links['inputToMeshToPoints'] = node_group.links.new(nodes['input'].outputs['Geometry'], nodes['meshToPoints'].inputs['Mesh'])
        links['meshToPointsToInstanceOnPoints'] = node_group.links.new(nodes['meshToPoints'].outputs['Points'], nodes['instanceOnPoints'].inputs['Points'])
        links['icoSphereToSetShadeSmooth'] = node_group.links.new(nodes['icoSphere'].outputs['Mesh'], nodes['setShadeSmooth'].inputs['Geometry'])
        links['setShadeSmoothToInstanceOnPoints'] = node_group.links.new(nodes['setShadeSmooth'].outputs['Geometry'], nodes['instanceOnPoints'].inputs['Instance'])
        links['instanceOnPointsToSetMaterial'] = node_group.links.new(nodes['instanceOnPoints'].outputs['Instances'], nodes['setMaterial'].inputs['Geometry'])
        links['setMaterialToRealizeInstances'] = node_group.links.new(nodes['setMaterial'].outputs['Geometry'], nodes['realizeInstances'].inputs['Geometry'])
        links['realizeInstancesToOutput'] = node_group.links.new(nodes['realizeInstances'].outputs['Geometry'], nodes['output'].inputs['Geometry'])

        nodes['icoSphere'].inputs['Radius'].default_value = 1
        nodes['icoSphere'].inputs['Subdivisions'].default_value = settings.subdivison

        if settings.radius_attribute is not None:
            nodes['radiusAttribute'] = node_group.nodes.new(type="GeometryNodeInputNamedAttribute")
            nodes['radiusAttribute'].data_type = 'FLOAT'
            nodes['radiusAttribute'].inputs['Name'].default_value = settings.radius_attribute
            links['radiusAttributeToInstanceOnPoints'] = node_group.links.new(nodes['radiusAttribute'].outputs['Attribute'], nodes['instanceOnPoints'].inputs['Scale'])
        else:
            radius = settings.radius
            nodes['instanceOnPoints'].inputs['Scale'].default_value = (radius, radius, radius)

        if settings.material is not None:
            bl_material = self._buildMaterial(settings.material)
            nodes['setMaterial'].inputs['Material'].default_value = bl_material

        return node_group


    def _buildWireframeNodeTree(self, settings: WireFrameSettings):
        """
        Build the bpy.types.GeometryNodeTree (Mesh to Curve -> Curve to Mesh along a circle
        profile) corresponding to a WireFrameSettings.

        Mesh to Curve keeps only the mesh's edges, discarding its faces, and Curve to Mesh
        then sweeps a circular profile along each of them -- so every edge comes out as a
        real 3D cylinder and the surface is gone. That "faces are hidden" is a consequence
        of how the geometry is built, not a material trick: nothing is left to render but
        the tubes.

        The tube radius is the product of two things Curve to Mesh multiplies together: the
        profile circle's own radius, fixed at 0.5 here (i.e. a unit-diameter profile), and
        Curve to Mesh's 'Scale' input, which is where settings.thickness is applied. That
        way thickness reads as the cylinder's diameter -- its girth -- and the same socket
        serves both the constant and the attribute-driven case, since 'Scale' is a field
        and so accepts a named attribute just as well as a constant.

        ('Scale' is deliberately the socket used here rather than a Set Curve Radius node
        feeding the curve's radius attribute: as of Blender 4.0 Curve to Mesh scales the
        profile by this input, and no longer picks the radius attribute up implicitly, so
        the Set Curve Radius route silently produces tubes of the wrong girth.)

        Note that a previous version of this method rendered the wireframe with Blender's
        Freestyle NPR system instead, which draws screen-space lines whose thickness is in
        pixels. These cylinders are ordinary geometry: they shade, cast and receive
        shadows, and shrink with distance, and their thickness is in the mesh's local units
        (so the parent mesh's scale applies to them, as it does to the mesh itself).

        :param settings: WireFrameSettings describing the wireframe.
        :return: The created bpy.types.GeometryNodeTree.

        """
        node_group = bpy.data.node_groups.new(type="GeometryNodeTree", name=settings.name)
        node_group.nodes.clear()
        nodes = {}
        links = {}

        nodes['input'] = node_group.nodes.new(type="NodeGroupInput")
        nodes['output'] = node_group.nodes.new(type="NodeGroupOutput")
        nodes['meshToCurve'] = node_group.nodes.new(type="GeometryNodeMeshToCurve")
        nodes['profileCircle'] = node_group.nodes.new(type="GeometryNodeCurvePrimitiveCircle")
        nodes['curveToMesh'] = node_group.nodes.new(type="GeometryNodeCurveToMesh")
        nodes['setShadeSmooth'] = node_group.nodes.new(type="GeometryNodeSetShadeSmooth")
        nodes['setMaterial'] = node_group.nodes.new(type="GeometryNodeSetMaterial")

        node_group.interface.new_socket(name="Geometry", description="", in_out="INPUT", socket_type="NodeSocketGeometry")
        node_group.interface.new_socket(name="Geometry", description="", in_out="OUTPUT", socket_type="NodeSocketGeometry")

        links['inputToMeshToCurve'] = node_group.links.new(nodes['input'].outputs['Geometry'], nodes['meshToCurve'].inputs['Mesh'])
        links['meshToCurveToCurveToMesh'] = node_group.links.new(nodes['meshToCurve'].outputs['Curve'], nodes['curveToMesh'].inputs['Curve'])
        links['profileCircleToCurveToMesh'] = node_group.links.new(nodes['profileCircle'].outputs['Curve'], nodes['curveToMesh'].inputs['Profile Curve'])
        links['curveToMeshToSetShadeSmooth'] = node_group.links.new(nodes['curveToMesh'].outputs['Mesh'], nodes['setShadeSmooth'].inputs['Geometry'])
        links['setShadeSmoothToSetMaterial'] = node_group.links.new(nodes['setShadeSmooth'].outputs['Geometry'], nodes['setMaterial'].inputs['Geometry'])

        # A unit-diameter profile, so the 'Scale' input below carries the thickness on its
        # own (see the docstring).
        nodes['profileCircle'].inputs['Radius'].default_value = 0.5
        nodes['profileCircle'].inputs['Resolution'].default_value = settings.resolution

        # Close the ends of the tubes. Mesh to Curve breaks the edge network into separate
        # splines wherever more than two edges meet at a vertex, so on anything but a
        # simple closed loop the tubes genuinely have open ends to cap.
        nodes['curveToMesh'].inputs['Fill Caps'].default_value = True

        if settings.thickness_attribute is not None:
            nodes['thicknessAttribute'] = node_group.nodes.new(type="GeometryNodeInputNamedAttribute")
            nodes['thicknessAttribute'].data_type = 'FLOAT'
            nodes['thicknessAttribute'].inputs['Name'].default_value = settings.thickness_attribute
            links['thicknessAttributeToCurveToMesh'] = node_group.links.new(nodes['thicknessAttribute'].outputs['Attribute'], nodes['curveToMesh'].inputs['Scale'])
        else:
            nodes['curveToMesh'].inputs['Scale'].default_value = settings.thickness

        if settings.material is not None:
            bl_material = self._buildMaterial(settings.material)
            nodes['setMaterial'].inputs['Material'].default_value = bl_material

        if settings.hide_surface:
            links['setMaterialToOutput'] = node_group.links.new(nodes['setMaterial'].outputs['Geometry'], nodes['output'].inputs['Geometry'])
        else:
            # Keep the shaded surface as well, by adding the untouched incoming mesh back
            # alongside the tubes. It keeps the material assigned to the mesh data-block;
            # only the tubes get settings.material.
            nodes['joinGeometry'] = node_group.nodes.new(type="GeometryNodeJoinGeometry")
            links['setMaterialToJoinGeometry'] = node_group.links.new(nodes['setMaterial'].outputs['Geometry'], nodes['joinGeometry'].inputs['Geometry'])
            links['inputToJoinGeometry'] = node_group.links.new(nodes['input'].outputs['Geometry'], nodes['joinGeometry'].inputs['Geometry'])
            links['joinGeometryToOutput'] = node_group.links.new(nodes['joinGeometry'].outputs['Geometry'], nodes['output'].inputs['Geometry'])

        return node_group
