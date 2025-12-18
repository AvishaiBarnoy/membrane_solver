from unittest.mock import MagicMock

import pytest

from runtime.energy_manager import EnergyModuleManager


@pytest.fixture
def energy_manager():
    # Mock importlib.import_module inside EnergyModuleManager
    with pytest.MonkeyPatch.context() as m:
        # Create dummy modules
        surface_mod = MagicMock()
        surface_mod.calculate_surface_energy = MagicMock(return_value=10.0)

        volume_mod = MagicMock()
        volume_mod.calculate_volume_energy = MagicMock(return_value=5.0)

        # Define side_effect for import_module
        def mock_import(name):
            if name.endswith("surface"):
                return surface_mod
            elif name.endswith("volume"):
                return volume_mod
            raise ImportError(f"No module named {name}")

        m.setattr("importlib.import_module", mock_import)

        manager = EnergyModuleManager(["surface", "volume"])
        return manager


def test_load_modules(energy_manager):
    assert "surface" in energy_manager.modules
    assert "volume" in energy_manager.modules


def test_get_module(energy_manager):
    mod = energy_manager.get_module("surface")
    assert mod is not None

    with pytest.raises(KeyError):
        energy_manager.get_module("nonexistent")
