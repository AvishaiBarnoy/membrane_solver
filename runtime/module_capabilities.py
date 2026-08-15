"""Explicit capability predicates for runtime module dispatch."""

from __future__ import annotations


def supports_array_energy_gradient(module: object) -> bool:
    """Preserve legacy array-capability detection used by evaluation dispatch."""
    return hasattr(module, "compute_energy_and_gradient_array")


def supports_array_energy(module: object) -> bool:
    """Preserve legacy energy-only array capability detection."""
    return hasattr(module, "compute_energy_array")


def uses_tilt(module: object) -> bool:
    """Return the legacy single-field tilt capability flag."""
    return bool(getattr(module, "USES_TILT", False))


def uses_leaflet_tilts(module: object) -> bool:
    """Return the legacy leaflet-tilt capability flag."""
    return bool(getattr(module, "USES_TILT_LEAFLETS", False))


def resolve_module_names(modules: list[object], names: list[str]) -> list[str]:
    """Return supplied names, or the legacy fallback names when misaligned."""
    if len(names) == len(modules):
        return names
    return [
        getattr(module, "__name__", module.__class__.__name__) for module in modules
    ]
