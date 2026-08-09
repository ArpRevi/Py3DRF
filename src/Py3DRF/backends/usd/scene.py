"""
SceneUSD: the USD (Universal Scene Description) backend for Py3DRF.

This is the only module in Py3DRF.backends.usd that imports pxr. It translates
Py3DRF.core objects (Camera, Mesh, SunLight, Material) into prims on a
pxr.Usd.Stage, and writes that stage out as ASCII USD (.usda).

Do not import this module directly -- go through the Scene facade
(Py3DRF.Scene(backend="usd", ...)), which loads it lazily.

Unlike the other backends, USD is a scene-description format, not a renderer:
there is no rasterized/path-traced image to produce. So this backend has no
renderToFile (see the explicit override below) and instead implements
exportToFile, which writes the scene description itself.

Honest limitations of this backend (deliberately left unimplemented rather
than faked -- calling them raises NotSupportedByBackendError):

  * renderToFile, show, setRenderEngine, setDevice, setResolution,
    setSamples, setGamma, setExposure, setShadowCatcherAlpha, saveToFile:
    all rendering/viewer concepts this backend, being a pure scene-description
    writer, has no equivalent for.
  * Mesh.point_cloud_settings / Mesh.wireframe_settings: not yet mapped to a
    USD prim type (e.g. UsdGeom.Points / BasisCurves); addObject rejects a
    mesh with either set, rather than silently ignoring the setting.
  * Material.color_attribute / emission_color_attribute /
    emission_strength_attribute: not yet mapped to a USD shading graph (e.g.
    a primvar reader feeding UsdPreviewSurface); only Material's constant
    color/roughness/emission are exported.
"""

import numpy as np
from pxr import Usd, UsdGeom, UsdLux, UsdShade, Sdf, Gf, Tf

from ...core.camera import Camera
from ...core.lights import SunLight
from ...core.materials import Material
from ...core.mesh import Mesh
from ..base import SceneBackend, NotSupportedByBackendError


class SceneUSD(SceneBackend):

    def __init__(self, name="Scene") -> None:
        self.name = name
        self._material_cache = {}
        self._used_names = set()

        self.stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)

        self._root_path = Sdf.Path(f"/{Tf.MakeValidIdentifier(name)}")
        root_prim = UsdGeom.Xform.Define(self.stage, self._root_path)
        self.stage.SetDefaultPrim(root_prim.GetPrim())

        self._materials_path = self._root_path.AppendChild("Materials")
        UsdGeom.Scope.Define(self.stage, self._materials_path)

    def addCamera(self, camera: Camera):
        """
        Build a UsdGeom.Camera prim from a Camera.

        :param camera: Camera object.
        :return: The created UsdGeom.Camera schema object.

        """
        path = self._childPath(camera.name)
        usd_camera = UsdGeom.Camera.Define(self.stage, path)
        usd_camera.CreateFocalLengthAttr(float(camera.focal_length))
        self._setTransform(usd_camera, camera.location, camera.rotation, camera.scale)
        return usd_camera

    def addObject(self, object):
        """
        Build a UsdGeom.Mesh or UsdLux.DistantLight prim from a Mesh or SunLight.

        :param object: Mesh or SunLight object to add to the scene.
        :return: The created USD schema object.

        """
        if isinstance(object, SunLight):
            return self._addSunLight(object)
        if isinstance(object, Mesh):
            return self._addMesh(object)
        raise TypeError(f"Cannot add object of type {type(object).__name__} to the scene.")

    def renderToFile(self, filepath):
        raise NotSupportedByBackendError(
            "SceneUSD does not implement 'renderToFile': USD is a scene-description "
            "format, not a renderer, so there's no rasterized image to write. Use "
            "exportToFile to write the scene description as a .usda file instead."
        )

    def exportToFile(self, filepath):
        """
        Export the stage to filepath as ASCII USD.

        :param filepath: Destination filepath for the exported scene description.
            Must end in '.usda' -- this backend only ever writes the ASCII text
            format, never the binary .usd/.usdc crate formats.

        """
        filepath = str(filepath)
        if not filepath.lower().endswith(".usda"):
            raise ValueError(
                f"SceneUSD.exportToFile only writes the ASCII .usda format; got "
                f"{filepath!r}. Use a filepath ending in '.usda'."
            )
        self.stage.GetRootLayer().Export(filepath)

    # -- Builders: translate plain-Python Py3DRF objects into USD prims --

    def _childPath(self, name):
        """
        Build a unique, valid Sdf.Path for a new child prim of the scene's root, from a
        Py3DRF object's (possibly non-identifier, possibly duplicate) name.
        """
        base = Tf.MakeValidIdentifier(name)
        candidate = base
        suffix = 1
        while candidate in self._used_names:
            suffix += 1
            candidate = f"{base}_{suffix}"
        self._used_names.add(candidate)
        return self._root_path.AppendChild(candidate)

    def _setTransform(self, schema, location, rotation, scale):
        """
        Apply a Py3DRF location/rotation/scale to a USD prim via XformCommonAPI, using the
        same XYZ Euler rotation order Mesh._localToWorldMatrix and SceneBlender's
        rotation_euler assume.
        """
        xform_api = UsdGeom.XformCommonAPI(schema)
        xform_api.SetTranslate(Gf.Vec3d(*location.to_tuple()))
        degrees = np.degrees(rotation.to_array())
        xform_api.SetRotate(Gf.Vec3f(*degrees), UsdGeom.XformCommonAPI.RotationOrderXYZ)
        xform_api.SetScale(Gf.Vec3f(*scale.to_tuple()))

    def _addSunLight(self, light: SunLight):
        """
        Build a UsdLux.DistantLight prim from a SunLight.
        """
        path = self._childPath(light.name)
        usd_light = UsdLux.DistantLight.Define(self.stage, path)
        usd_light.CreateIntensityAttr(float(light.strength))
        self._setTransform(usd_light, light.location, light.rotation, light.scale)
        return usd_light

    def _addMesh(self, mesh: Mesh):
        """
        Build a UsdGeom.Mesh prim from a Mesh's world-space geometry and material.

        World-space vertices are baked directly into the prim's points (as ScenePlotly
        does for its traces), leaving the prim's own transform at identity, rather than
        re-deriving Mesh's rotation convention as a USD xform.
        """
        if mesh.point_cloud_settings is not None:
            raise NotSupportedByBackendError(
                "SceneUSD cannot export a point cloud: Mesh.point_cloud_settings has no "
                "mapping to a USD prim type in this backend yet. Export the plain mesh "
                "instead, or use a backend that supports point clouds directly."
            )
        if mesh.wireframe_settings is not None:
            raise NotSupportedByBackendError(
                "SceneUSD cannot export a wireframe: Mesh.wireframe_settings has no "
                "mapping to a USD prim type in this backend yet."
            )

        path = self._childPath(mesh.name)
        usd_mesh = UsdGeom.Mesh.Define(self.stage, path)

        vertices = mesh._worldVertices()
        usd_mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in vertices])

        counts = [len(face) for face in mesh.faces]
        indices = [int(i) for face in mesh.faces for i in face]
        usd_mesh.CreateFaceVertexCountsAttr(counts)
        usd_mesh.CreateFaceVertexIndicesAttr(indices)
        usd_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)

        material = self._buildMaterial(mesh.material)
        UsdShade.MaterialBindingAPI.Apply(usd_mesh.GetPrim())
        UsdShade.MaterialBindingAPI(usd_mesh.GetPrim()).Bind(material)

        return usd_mesh

    def _buildMaterial(self, material: Material):
        """
        Build (or fetch from cache) the UsdShade.Material corresponding to a Material,
        wiring up a UsdPreviewSurface shader from its constant color/roughness/emission.
        """
        cached = self._material_cache.get(id(material))
        if cached is not None:
            return cached

        if (
            material.color_attribute is not None
            or material.emission_color_attribute is not None
            or material.emission_strength_attribute is not None
        ):
            raise NotSupportedByBackendError(
                "SceneUSD does not map attribute-driven materials (color_attribute / "
                "emission_color_attribute / emission_strength_attribute) to a USD shading "
                "graph yet -- only Material's constant color/roughness/emission are exported."
            )

        material_name = Tf.MakeValidIdentifier(f"{material.name}_{len(self._material_cache)}")
        material_path = self._materials_path.AppendChild(material_name)

        usd_material = UsdShade.Material.Define(self.stage, material_path)
        shader = UsdShade.Shader.Define(self.stage, material_path.AppendChild("PreviewSurface"))
        shader.CreateIdAttr("UsdPreviewSurface")

        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*material.color[:3]))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(material.roughness))

        if len(material.color) > 3 and material.color[3] < 1.0:
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(material.color[3]))

        emission = tuple(c * material.emission_strength for c in material.emission_color[:3])
        if any(emission):
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emission))

        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        usd_material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        self._material_cache[id(material)] = usd_material
        return usd_material
