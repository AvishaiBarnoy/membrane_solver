# tests/test_constraint_manager.py

import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json

from geometry.geom_io import load_data, parse_geometry
from runtime.constraint_manager import ConstraintModuleManager


def test_constraint_manager_init(monkeypatch):
    # Mock constraint module names
    constraint_names = ["dummy_constraint", "pin_to_plane"]

    # Mock importlib to simulate loading modules
    import importlib
    mock_import_module = MagicMock()
    sys.modules["modules.constraints.dummy_constraint"] = MagicMock()
    sys.modules["modules.constraints.pin_to_plane"] = MagicMock()

    monkeypatch.setattr(importlib, "import_module", mock_import_module)

    # Initialize the ConstraintModuleManager
    constraint_manager = ConstraintModuleManager(constraint_names)

    # Check that the correct modules were loaded
    assert "dummy_constraint" in constraint_manager.modules
    assert "pin_to_plane" in constraint_manager.modules
    mock_import_module.assert_any_call("modules.constraints.dummy_constraint")
    mock_import_module.assert_any_call("modules.constraints.pin_to_plane")

def test_get_module():
    # Mock constraint module names
    constraint_names = ["dummy_constraint", "pin_to_plane"]

    # Mock modules
    mock_dummy_module = MagicMock()
    mock_pin_to_plane_module = MagicMock()

    # Initialize the ConstraintModuleManager
    constraint_manager = ConstraintModuleManager(constraint_names)
    constraint_manager.modules = {
        "dummy_constraint": mock_dummy_module,
        "pin_to_plane": mock_pin_to_plane_module
    }

    # Test retrieving loaded modules
    for name in constraint_names:
        module = constraint_manager.get_module(name)
        assert module is not None, f"Module '{name}' should be loaded"
        assert module == constraint_manager.modules[name], f"Module '{name}' should match the loaded module"

    # Test retrieving a non-existent module
    with pytest.raises(KeyError, match="Constraint module 'non_existent' not found."):
        constraint_manager.get_module("non_existent")

def test_constraints_loaded_from_file(monkeypatch, tmp_path):
    # Mock importlib to simulate loading modules
    import importlib
    mock_import_module = MagicMock()
    sys.modules["modules.constraints.pin_to_plane"] = MagicMock()
    sys.modules["modules.constraints.fix_facet_area"] = MagicMock()

    monkeypatch.setattr(importlib, "import_module", mock_import_module)

    # Create a temporary geometry file
    sample_geometry_path = tmp_path / "testing_geometry.json"
    sample_geometry_content = {
        "vertices": [
            [0, 0, 0, {"fixed": "true"}],
            [1, 0, 0, {"constraints": ["pin_to_plane"]}],
            [1, 1, 0, {"constraints": ["fixed"]}],
            [0, 1, 0]
        ],
        "edges": [
            [0, 1, {"constraints": "pin_to_plane"}],
            [1, 2],
            [2, 3],
            [3, 0]
        ],
        "faces": [
            [0, 1, 2, 3, {"constraints": ["fix_facet_area"]}]
        ],
        "bodies": {
            "faces": [[0]],
            "target_volume": [1.0]
        },
        "global_parameters": {},
        "instructions": []
    }
    with open(sample_geometry_path, "w") as f:
        json.dump(sample_geometry_content, f)

    # Load and parse the geometry
    data = load_data(str(sample_geometry_path))
    mesh = parse_geometry(data)

    # Initialize the ConstraintModuleManager
    constraint_manager = ConstraintModuleManager(mesh.constraint_modules)
    print(mesh.constraint_modules)

    # Check that the correct constraint modules were loaded
    assert "pin_to_plane" in constraint_manager.modules
    assert "fix_facet_area" in constraint_manager.modules
    mock_import_module.assert_any_call("modules.constraints.pin_to_plane")
    mock_import_module.assert_any_call("modules.constraints.fix_facet_area")

    # Check that constraints are correctly assigned to vertices and facets
    assert "constraints" in mesh.vertices[1].options
    assert mesh.vertices[1].options["constraints"] == ["pin_to_plane"]

    assert "constraints" in mesh.facets[0].options
    assert mesh.facets[0].options["constraints"] == ["fix_facet_area"]
