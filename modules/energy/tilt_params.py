"""Parameter resolution helpers for leaflet-specific tilt energy modules."""

from __future__ import annotations


def _resolve_tilt_modulus(param_resolver, leaflet: str) -> float:
    """Resolve tilt magnitude modulus for the specified leaflet."""
    k = param_resolver.get(None, f"tilt_modulus_{leaflet}")
    if k is None:
        # Legacy typo fallback
        k = param_resolver.get(None, f"tilt_modolus_{leaflet}")
    return float(k or 0.0)


def _resolve_tilt_mass_mode(param_resolver, leaflet: str) -> str:
    """Resolve tilt mass mode (lumped vs consistent) for the specified leaflet."""
    mode = param_resolver.get(None, f"tilt_mass_mode_{leaflet}")
    if mode is None:
        mode = param_resolver.get(None, "tilt_mass_mode")
    txt = str(mode or "lumped").strip().lower()
    if txt not in {"lumped", "consistent"}:
        raise ValueError(f"tilt_mass_mode_{leaflet} must be 'lumped' or 'consistent'.")
    return txt


def _resolve_exclude_shared_rim_outer_rows(param_resolver, leaflet: str) -> bool:
    """Resolve whether to exclude shared-rim outer rows for the specified leaflet."""
    raw = param_resolver.get(None, f"tilt_{leaflet}_exclude_shared_rim_outer_rows")
    if raw is None:
        if leaflet == "out":
            raw = param_resolver.get(None, "tilt_out_exclude_shared_rim_rows")
        else:
            raw = param_resolver.get(None, "tilt_in_exclude_shared_rim_outer_rows")
    if raw is None:
        raw = param_resolver.get(None, f"tilt_exclude_shared_rim_rows_{leaflet}")

    if raw is None:
        return False
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(raw)
