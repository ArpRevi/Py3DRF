import pytest

pxr = pytest.importorskip("pxr")
from pxr import Usd

from Py3DRF import Camera, Location, Material, Mesh, NotSupportedByBackendError, SunLight
from Py3DRF.backends.usd.scene import SceneUSD

TRIANGLE = dict(
    vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0)],
    faces=[(0, 1, 2)],
)


def test_export_round_trips_mesh_geometry(tmp_path):
    scene = SceneUSD()
    mesh = Mesh("Triangle", **TRIANGLE)
    scene.addObject(mesh)

    out_path = tmp_path / "out.usda"
    scene.exportToFile(str(out_path))
    assert out_path.exists()

    stage = Usd.Stage.Open(str(out_path))
    mesh_prim = stage.GetPrimAtPath("/Scene/Triangle")
    assert mesh_prim.IsValid()

    points_attr = mesh_prim.GetAttribute("points")
    assert len(points_attr.Get()) == 3

    counts_attr = mesh_prim.GetAttribute("faceVertexCounts")
    assert list(counts_attr.Get()) == [3]


def test_export_requires_usda_extension(tmp_path):
    scene = SceneUSD()
    scene.addObject(Mesh("Triangle", **TRIANGLE))

    with pytest.raises(ValueError):
        scene.exportToFile(str(tmp_path / "out.usd"))


def test_render_to_file_is_not_supported():
    scene = SceneUSD()
    with pytest.raises(NotSupportedByBackendError):
        scene.renderToFile("out.png")


def test_point_cloud_mesh_is_not_supported():
    scene = SceneUSD()
    mesh = Mesh("Triangle", **TRIANGLE)
    mesh.asPointCloud()

    with pytest.raises(NotSupportedByBackendError):
        scene.addObject(mesh)


def test_wireframe_mesh_is_not_supported():
    scene = SceneUSD()
    mesh = Mesh("Triangle", **TRIANGLE)
    mesh.asWireframe()

    with pytest.raises(NotSupportedByBackendError):
        scene.addObject(mesh)


def test_attribute_driven_material_is_not_supported():
    scene = SceneUSD()
    mesh = Mesh("Triangle", **TRIANGLE)
    mesh.addFloatAttribute([0.0, 0.5, 1.0], "value")
    mesh.material.setFloatAttributeAsColor("value")

    with pytest.raises(NotSupportedByBackendError):
        scene.addObject(mesh)


def test_add_camera_and_sun_light(tmp_path):
    scene = SceneUSD()
    scene.addCamera(Camera(location=Location(0, -5, 2)))
    scene.addObject(SunLight())

    out_path = tmp_path / "out.usda"
    scene.exportToFile(str(out_path))

    stage = Usd.Stage.Open(str(out_path))
    assert stage.GetPrimAtPath("/Scene/Camera").IsValid()
    assert stage.GetPrimAtPath("/Scene/Sun").IsValid()
