import bpy"""

from b3Dv.scene import Scene

"""bpy.ops.object.select_all(action = 'SELECT')
bpy.ops.object.delete()"""

from . scene import Scene
from . camera import Camera
from . mesh import Mesh
from . lights import SunLight
from . materials import Material

"""
def saveToFile(filename):
    bpy.ops.wm.save_mainfile(filepath = filename)
"""

__all__ = ['Scene', 'Camera', 'Mesh', 'SunLight', 'Material', 'saveToFile']