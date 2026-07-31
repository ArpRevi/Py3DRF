"""
Py3DRF: a backend-agnostic facade for 3D scene description and rendering.

Public API:
    Scene                                        -- choose a backend, render
    Camera, Mesh, SunLight, Material, PointCloudSettings  -- backend-agnostic data model

Importing Py3DRF never imports any backend's third-party dependency (bpy,
plotly, polyscope, pxr, ...). Only Scene(backend=...) triggers loading the
one backend you asked for; see Py3DRF.backends for details.
"""

from .core import Camera, SunLight, Material, Mesh, PointCloudSettings, MeshToPointCloudNodeTree, WireFrameSettings
from .scene import Scene
from .backends import available_backends, BackendConflictError, BackendConflictWarning
from .backends.base import NotSupportedByBackendError
from .core.types import Location, Rotation, Scale

__all__ = [
    "Scene",
    "Camera",
    "SunLight",
    "Material",
    "Mesh",
    "PointCloudSettings",
    "MeshToPointCloudNodeTree",
    "WireFrameSettings",
    "available_backends",
    "NotSupportedByBackendError",
    "BackendConflictError",
    "BackendConflictWarning",
    "Location",
    "Rotation",
    "Scale"
]
