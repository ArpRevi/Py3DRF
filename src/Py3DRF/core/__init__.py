"""
Py3DRF.core: the backend-agnostic scene-description data model.

Every class in this subpackage (Camera, Mesh, SunLight, Material,
PointCloudSettings) is a plain Python/numpy object. None of them import any
rendering backend (bpy, plotly, polyscope, ...). Translating these objects
into a backend-native scene is the sole responsibility of the backends
subpackage, selected at runtime through the Scene facade.
"""

from .camera import Camera
from .lights import SunLight
from .materials import Material
from .mesh import Mesh
from .pointcloud import PointCloudSettings, MeshToPointCloudNodeTree

__all__ = [
    "Camera",
    "SunLight",
    "Material",
    "Mesh",
    "PointCloudSettings",
    "MeshToPointCloudNodeTree",
]
