import numpy as np
import pytest

pytest.importorskip("pyvista")

from Py3DRF import Camera, Location, Material, Mesh, NotSupportedByBackendError, Rotation, SunLight
from Py3DRF.backends.pyvista.scene import ScenePyVista, _buildLookupTable, _rotationToDirection

TRIANGLE = dict(vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0)], faces=[(0, 1, 2)])
QUAD = dict(vertices=[(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], faces=[(0, 1, 2, 3)])


def test_default_is_headless():
    scene = ScenePyVista()
    assert scene.plotter.off_screen is True


def test_quad_is_kept_as_a_single_ngon_cell():
    # Unlike Plotly, PyVista takes n-gons natively -- no triangulation needed.
    scene = ScenePyVista()
    actor = scene.addObject(Mesh("Quad", **QUAD))
    assert actor.mapper.dataset.n_cells == 1


def test_mixed_face_lengths_do_not_raise():
    mesh = Mesh(
        "Mixed",
        vertices=[(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (2, 0, 0)],
        faces=[(0, 1, 2, 3), (1, 4, 2)],
    )
    scene = ScenePyVista()
    scene.addObject(mesh)  # should not raise


def test_lookup_table_reproduces_ramp_exactly_at_stop_positions():
    lut = _buildLookupTable([(0.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0)], positions=[0.0, 0.5, 1.0])

    np.testing.assert_array_equal(lut.values[0], [0, 0, 0, 255])
    np.testing.assert_array_equal(lut.values[-1], [255, 255, 255, 255])
    # midpoint (position 0.5) should land exactly on pure red
    mid = lut.values[len(lut.values) // 2]
    np.testing.assert_allclose(mid, [255, 0, 0, 255], atol=2)


def test_lookup_table_defaults_to_evenly_spaced_positions():
    lut = _buildLookupTable([(0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0)])
    np.testing.assert_array_equal(lut.values[0], [0, 0, 0, 255])
    np.testing.assert_array_equal(lut.values[-1], [255, 255, 255, 255])


def test_float_attribute_drives_the_custom_ramp_visibly():
    mesh = Mesh("T", **TRIANGLE)
    mesh.addFloatAttribute([0.0, 0.5, 1.0], "v")
    mesh.material.setFloatAttributeAsColor("v", colors=[(0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0)])

    scene = ScenePyVista(resolution=(80, 80), transparent=False)
    scene.addObject(mesh)
    scene.addCamera(Camera(location=Location(0.3, 0.3, 3), target=Location(0.3, 0.3, 0)))

    image = scene.plotter.screenshot()
    assert image[..., :3].std() > 5  # not a flat/uniform image


def test_camera_up_is_not_degenerate_when_looking_straight_down_z():
    # Regression: hardcoding up=(0,0,1) makes the camera basis degenerate
    # (cross(view_direction, up) == 0) whenever the view direction is itself
    # parallel to Z -- an entirely ordinary axial top-down view. VTK then
    # renders a blank frame with no error at all.
    scene = ScenePyVista(resolution=(100, 100), transparent=False)
    scene.addObject(Mesh("Quad", vertices=QUAD["vertices"], faces=QUAD["faces"], material=Material(color=(0.9, 0.1, 0.1, 1))))
    scene.addCamera(Camera(location=Location(0.5, 0.5, 3), target=Location(0.5, 0.5, 0)))

    assert abs(np.dot(scene.plotter.camera.up, (0, 0, 1))) < 0.999

    image = scene.plotter.screenshot()
    assert image[..., :3].std() > 5


def test_camera_up_is_still_z_for_an_ordinary_view():
    scene = ScenePyVista()
    scene.addCamera(Camera(location=Location(2, 2, 2), target=Location(0, 0, 0)))
    np.testing.assert_allclose(scene.plotter.camera.up, (0.0, 0.0, 1.0))


def test_rotation_to_direction_matches_blenders_unrotated_sun():
    # An unrotated Blender Sun points along its local -Z.
    np.testing.assert_allclose(_rotationToDirection(Rotation(0, 0, 0)), [0.0, 0.0, -1.0], atol=1e-10)


def test_sun_light_is_positional_false_directional():
    scene = ScenePyVista()
    light = scene.addObject(SunLight(strength=3.0))
    assert light.positional is False
    assert light.intensity == 3.0


def test_point_cloud_glyphs_one_icosphere_per_vertex():
    mesh = Mesh("PC", vertices=[(0, 0, 0), (1, 0, 0), (2, 0, 0)], faces=[])
    mesh.asPointCloud(radius=0.1, subdivison=1)

    scene = ScenePyVista()
    actor = scene.addObject(mesh)

    # 3 vertices, each glyphed with the same icosphere -> 3x the single-icosphere point count
    single = __import__("pyvista").Icosphere(radius=0.1, nsub=1)
    assert actor.mapper.dataset.n_points == 3 * single.n_points


def test_point_cloud_radius_attribute_varies_glyph_size():
    mesh = Mesh("PC", vertices=[(0, 0, 0), (1, 0, 0)], faces=[])
    mesh.addFloatAttribute([0.1, 0.5], "rad")
    mesh.asPointCloud()
    mesh.point_cloud_settings.setFloatAttributeAsRadius("rad")

    scene = ScenePyVista()
    actor = scene.addObject(mesh)

    bounds = actor.mapper.dataset.bounds
    # the second, larger-radius sphere (centered at x=1) should push the x-max well past 1
    assert bounds[1] > 1.3


def test_wireframe_builds_tube_geometry():
    mesh = Mesh("WF", **TRIANGLE)
    mesh.asWireframe(thickness=0.05)

    scene = ScenePyVista()
    actor = scene.addObject(mesh)

    # a tube has real volume (many points per edge), unlike a flat line
    assert actor.mapper.dataset.n_points > 6


def test_wireframe_thickness_attribute_varies_tube_radius():
    mesh = Mesh("WF", **TRIANGLE)
    mesh.addFloatAttribute([0.01, 0.01, 0.3], "thick")
    mesh.asWireframe(thickness=0.01)
    mesh.wireframe_settings.setFloatAttributeAsThickness("thick")

    scene = ScenePyVista()
    actor = scene.addObject(mesh)

    bounds = actor.mapper.dataset.bounds
    # the vertex with thickness=0.3 should make the tube noticeably fatter there
    # than a uniformly-thin (thickness=0.01) tube would be
    span = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
    assert span > 1.1  # triangle itself spans 1.0 in X/Y; a fat tube end adds to it


def test_wireframe_hide_surface_false_also_adds_the_surface():
    mesh = Mesh("WF", **TRIANGLE)
    mesh.asWireframe(thickness=0.02, hide_surface=False)

    scene = ScenePyVista()
    scene.addObject(mesh)

    assert len(scene.plotter.actors) == 2


def test_wireframe_with_no_edges_does_not_raise():
    # Regression: a mesh with no faces and no explicit edges makes
    # Mesh._edgePairs() return a (0, 2) array. tube() on a PolyData built from
    # that has points but no line cells, and pyvista's add_mesh then raised its
    # own "Empty meshes cannot be plotted" -- unlike Blender/Plotly, which
    # silently render nothing for the same input. Match them instead of raising.
    mesh = Mesh("Empty", vertices=[(0, 0, 0), (1, 0, 0)], faces=[])
    mesh.asWireframe(thickness=0.02)

    scene = ScenePyVista()
    actor = scene.addObject(mesh)  # should not raise

    assert actor is None
    assert len(scene.plotter.actors) == 0


def test_wireframe_with_no_edges_and_hide_surface_false_still_adds_the_surface():
    mesh = Mesh("Empty", vertices=[(0, 0, 0), (1, 0, 0)], faces=[])
    mesh.asWireframe(thickness=0.02, hide_surface=False)

    scene = ScenePyVista()
    actor = scene.addObject(mesh)

    assert actor is None
    assert len(scene.plotter.actors) == 1


def test_shadow_catcher_is_approximated_as_reduced_opacity():
    mesh = Mesh("SC", **TRIANGLE)
    mesh.is_shadow_catcher = True

    scene = ScenePyVista()
    actor = scene.addObject(mesh)

    assert actor.GetProperty().GetOpacity() == pytest.approx(0.3)


@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.setSamples(64),
        lambda s: s.setGamma(1.0),
        lambda s: s.setExposure(0.0),
        lambda s: s.setShadowCatcherAlpha(0.5),
        lambda s: s.setRenderEngine("CYCLES"),
        lambda s: s.setDevice("GPU"),
        lambda s: s.saveToFile("x.vtkjs"),
    ],
)
def test_unsupported_methods_raise(call):
    scene = ScenePyVista()
    with pytest.raises(NotSupportedByBackendError):
        call(scene)


def test_render_to_file(tmp_path):
    scene = ScenePyVista(resolution=(80, 80), transparent=False)
    scene.addObject(Mesh("T", material=Material(color=(0.9, 0.1, 0.1, 1)), **TRIANGLE))
    scene.addCamera(Camera(location=Location(2, 2, 2), target=Location(0.3, 0.3, 0)))

    out = tmp_path / "out.png"
    scene.renderToFile(str(out))

    assert out.exists() and out.stat().st_size > 0
