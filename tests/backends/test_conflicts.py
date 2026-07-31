import sys
import types

import pytest

from Py3DRF.backends import (
    BackendConflictError,
    BackendConflictWarning,
    _check_conflicts,
    load_backend_class,
)


@pytest.fixture
def fake_module(monkeypatch):
    """Register a bare module object under `name` in sys.modules, then remove it."""
    def _install(name):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    return _install


def test_check_conflicts_raises_by_default(fake_module):
    fake_module("pxr")
    with pytest.raises(BackendConflictError):
        _check_conflicts("blender", allow_backend_conflict=False)


def test_check_conflicts_symmetric(fake_module):
    fake_module("bpy")
    with pytest.raises(BackendConflictError):
        _check_conflicts("usd", allow_backend_conflict=False)


def test_check_conflicts_allowed_warns_instead_of_raising(fake_module):
    fake_module("pxr")
    with pytest.warns(BackendConflictWarning):
        _check_conflicts("blender", allow_backend_conflict=True)


def test_check_conflicts_no_conflict_present_is_a_no_op():
    # Neither "pxr" nor "bpy" is faked into sys.modules here (they may or may
    # not be genuinely installed, but that's not what's under test).
    _check_conflicts("plotly", allow_backend_conflict=False)
    _check_conflicts("polyscope", allow_backend_conflict=False)


def test_load_backend_class_raises_before_importing_the_conflicting_dependency(fake_module):
    # The conflict check must run before load_backend_class attempts to import
    # bpy/pxr, so this raises BackendConflictError even when bpy genuinely
    # isn't installed on this machine.
    fake_module("pxr")
    with pytest.raises(BackendConflictError):
        load_backend_class("blender")
