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


def test_material_cache_does_not_bypass_the_unsupported_guard():
    # Regression: the cache lookup ran BEFORE the attribute-driven-material guard,
    # so a Material cached while it still had only constant values kept returning
    # that stale entry after attributes were added -- silently exporting geometry
    # this backend can't shade, instead of raising.
    scene = SceneUSD()
    material = Material()
    scene.addObject(Mesh("A", material=material, **TRIANGLE))

    material.setFloatAttributeAsColor("v", colors=[(0, 0, 0, 1), (1, 1, 1, 1)])
    mesh_b = Mesh("B", material=material, **TRIANGLE)
    mesh_b.addFloatAttribute([0.0, 0.5, 1.0], "v")

    with pytest.raises(NotSupportedByBackendError):
        scene.addObject(mesh_b)


def test_material_shared_between_meshes_is_written_once(tmp_path):
    scene = SceneUSD()
    shared = Material(name="Shared", color=(1.0, 0.0, 0.0, 1.0))
    scene.addObject(Mesh("A", material=shared, **TRIANGLE))
    scene.addObject(Mesh("B", material=shared, **TRIANGLE))

    out = tmp_path / "out.usda"
    scene.exportToFile(str(out))

    assert out.read_text().count('def Material "') == 1


def test_distinct_materials_are_not_confused_after_one_is_released(tmp_path):
    # Regression: the cache was keyed on id(material), which keeps nothing alive.
    # CPython reuses freed addresses constantly, so a released Material could hand
    # its cached appearance to an unrelated one allocated at the same address.
    scene = SceneUSD()
    first = Material(name="RED", color=(1.0, 0.0, 0.0, 1.0))
    scene.addObject(Mesh("A", material=first, **TRIANGLE))
    del first

    second = Material(name="BLUE", color=(0.0, 0.0, 1.0, 1.0))
    scene.addObject(Mesh("B", material=second, **TRIANGLE))

    out = tmp_path / "out.usda"
    scene.exportToFile(str(out))
    text = out.read_text()

    assert "BLUE" in text and "RED" in text


def test_add_camera_and_sun_light(tmp_path):
    scene = SceneUSD()
    scene.addCamera(Camera(location=Location(0, -5, 2)))
    scene.addObject(SunLight())

    out_path = tmp_path / "out.usda"
    scene.exportToFile(str(out_path))

    stage = Usd.Stage.Open(str(out_path))
    assert stage.GetPrimAtPath("/Scene/Camera").IsValid()
    assert stage.GetPrimAtPath("/Scene/Sun").IsValid()
