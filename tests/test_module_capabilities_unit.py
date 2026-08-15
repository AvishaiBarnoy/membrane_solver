from __future__ import annotations

from runtime.module_capabilities import (
    resolve_module_names,
    supports_array_energy,
    supports_array_energy_gradient,
    uses_leaflet_tilts,
    uses_tilt,
)


class _ArrayModule:
    def compute_energy_and_gradient_array(self):
        pass

    def compute_energy_array(self):
        pass


def test_array_gradient_capability_matches_legacy_attribute_contract() -> None:
    assert supports_array_energy_gradient(_ArrayModule())
    assert supports_array_energy(_ArrayModule())
    assert not supports_array_energy_gradient(object())


def test_tilt_capabilities_default_to_false_and_preserve_truthiness() -> None:
    class SingleTilt:
        USES_TILT = 1

    class LeafletTilt:
        USES_TILT_LEAFLETS = "enabled"

    assert uses_tilt(SingleTilt())
    assert not uses_leaflet_tilts(SingleTilt())
    assert uses_leaflet_tilts(LeafletTilt())
    assert not uses_tilt(object())


def test_module_name_resolution_preserves_aligned_names_and_legacy_fallback():
    names = ["configured"]
    assert resolve_module_names([object()], names) is names

    class NamedModule:
        pass

    def function_module():
        pass

    assert resolve_module_names([NamedModule(), function_module], []) == [
        "NamedModule",
        "function_module",
    ]
