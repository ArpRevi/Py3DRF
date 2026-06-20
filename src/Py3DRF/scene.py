import bpy
import numpy as np

from b3Dv.camera import Camera
from b3Dv.lights import SunLight

class Scene:

    def __init__(self, name="Scene", engine="CYCLES", device="GPU", resolution=(1920, 1080), transparent=True, deafult_sun=False, gamma=1, exposure=0, shadow_catcher_alpha=1.0) -> None:
        self.data = bpy.data.scenes.new(name)
        self.setRenderEngine(engine=engine)
        self.setDevice(device=device)
        self.setResolution(resolution)
        self.data.render.film_transparent = transparent
        self.data.view_settings.view_transform = "Raw"

        self.data.view_layers[0].cycles.use_pass_shadow_catcher = True

        self.data.use_nodes = True
        self.nodes = {"input" : self.data.node_tree.nodes["Render Layers"], 
                      "output" : self.data.node_tree.nodes["Composite"]}
        self.data.node_tree.links.clear()
        self.links = {}

        self.nodes['input'].scene = self.data

        self.nodes['invert'] = self.data.node_tree.nodes.new("CompositorNodeInvert")
        self.nodes['add'] = self.data.node_tree.nodes.new("CompositorNodeMixRGB")
        self.nodes['setAlpha'] = self.data.node_tree.nodes.new("CompositorNodeSetAlpha")

        self.nodes['add'].blend_type = 'ADD'
        self.nodes['setAlpha'].mode = 'REPLACE_ALPHA'

        self.links['inputToInvert'] = self.data.node_tree.links.new(self.nodes["input"].outputs['Shadow Catcher'], self.nodes["invert"].inputs["Color"])
        self.links['inputToAdd'] = self.data.node_tree.links.new(self.nodes['input'].outputs['Alpha'], self.nodes['add'].inputs['Image'])
        self.links['invertToAdd'] = self.data.node_tree.links.new(self.nodes['invert'].outputs['Color'], self.nodes['add'].inputs['Image_001'])
        self.links['inputToSetAlpha'] = self.data.node_tree.links.new(self.nodes['input'].outputs['Image'], self.nodes['setAlpha'].inputs['Image'])
        self.links['addToSetAlpha'] = self.data.node_tree.links.new(self.nodes['add'].outputs['Image'], self.nodes['setAlpha'].inputs['Alpha'])
        self.links['setAlphaToOutput'] = self.data.node_tree.links.new(self.nodes['setAlpha'].outputs['Image'], self.nodes['output'].inputs['Image'])

        self.setShadowCatcherAlpha(shadow_catcher_alpha)
        self.setGamma(gamma)
        self.setExposure(exposure)

        world = bpy.data.worlds.new("World")
        self.data.world = world

        if deafult_sun:
            sun = SunLight(rotation=(-30 * np.pi/180, 0, -10 * np.pi/180))
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

    def addCamera(self, camera:Camera):
        """
        Add camera object and set as active camera.

        :param camera: Camera object.

        """
        self.data.collection.objects.link(camera.object)
        self.data.camera = camera.object

    def addObject(self, object):
        """
        Add object to the scene.

        :param object: Object to add to the scene.

        """
        self.data.collection.objects.link(object.object)

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
        self.nodes['add'].inputs['Fac'].default_value = alpha

    def renderToFile(self, filepath):
        self.data.render.filepath = filepath
        bpy.ops.render.render(write_still=True, scene=self.data.name)