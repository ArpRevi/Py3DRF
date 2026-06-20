from .camera import Camera
from .mesh import Mesh
from .lights import SunLight
from .materials import Material

__all__ = ['Camera', 'Mesh', 'SunLight', 'Material']

try:
    # Scene is the only class that depends on bpy (Blender's Python API), since it is
    # responsible for turning Camera/Mesh/SunLight/Material objects into actual Blender
    # data-blocks. Everything else in Py3DRF can be imported and used outside of Blender
    # (e.g. to build up a scene graph for testing, or for a future non-Blender backend).
    from .scene import Scene
    __all__.append('Scene')
except ImportError:
    Scene = None
