import numpy as np
import pytest

from Py3DRF import Location, Mesh, Scale


def test_set_location_keeps_the_location_type():
    # Regression: setLocation used to store location.to_array(), a numpy array,
    # while the constructor and every reader expect a Location dataclass -- so
    # any call to it left the mesh permanently broken (every bounds query, and
    # adding it to a Blender/USD scene, raised AttributeError on .to_tuple()).
    mesh = Mesh("T", vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0)], faces=[(0, 1, 2)])
    mesh.setLocation(Location(1, 2, 3))

    assert isinstance(mesh.location, Location)
    assert mesh.getMinZ() == 3.0
    np.testing.assert_allclose(mesh._worldVertices()[0], [1, 2, 3])


def test_triangles_passes_triangles_through():
    mesh = Mesh("T", vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0)], faces=[(0, 1, 2)])
    np.testing.assert_array_equal(mesh._triangles(), [[0, 1, 2]])


def test_triangles_fans_a_quad_into_two():
    # Regression: a quad must survive as two triangles. Slicing the first three
    # indices instead dropped the 4th corner and lost half the surface.
    mesh = Mesh("Q", vertices=[(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], faces=[(0, 1, 2, 3)])
    np.testing.assert_array_equal(mesh._triangles(), [[0, 1, 2], [0, 2, 3]])


def test_triangles_handles_mixed_face_lengths():
    mesh = Mesh(
        "M",
        vertices=[(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (2, 0, 0)],
        faces=[(0, 1, 2, 3), (1, 4, 2)],
    )
    triangles = mesh._triangles()

    assert triangles.shape == (3, 3)
    np.testing.assert_array_equal(triangles, [[0, 1, 2], [0, 2, 3], [1, 4, 2]])


def test_triangles_fans_a_pentagon():
    mesh = Mesh("P", vertices=[(i, 0, 0) for i in range(5)], faces=[(0, 1, 2, 3, 4)])
    # An n-gon becomes n-2 triangles, all sharing the first vertex.
    np.testing.assert_array_equal(mesh._triangles(), [[0, 1, 2], [0, 2, 3], [0, 3, 4]])


def test_triangles_of_a_faceless_mesh_is_empty_not_an_error():
    mesh = Mesh("E", vertices=[(0, 0, 0), (1, 0, 0)], edges=[(0, 1)])
    assert mesh._triangles().shape == (0, 3)


def test_point_cloud_and_wireframe_stay_mutually_exclusive():
    mesh = Mesh("T", vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0)], faces=[(0, 1, 2)])

    mesh.asPointCloud()
    assert mesh.point_cloud_settings is not None and mesh.wireframe_settings is None

    mesh.asWireframe()
    assert mesh.wireframe_settings is not None and mesh.point_cloud_settings is None

    mesh.asPointCloud()
    assert mesh.point_cloud_settings is not None and mesh.wireframe_settings is None


def test_world_vertices_applies_scale_and_location():
    mesh = Mesh(
        "S",
        vertices=[(1, 1, 1)],
        faces=[],
        location=Location(10, 20, 30),
        scale=Scale(2, 3, 4),
    )
    np.testing.assert_allclose(mesh._worldVertices()[0], [12, 23, 34])
