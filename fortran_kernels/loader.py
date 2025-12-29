"""Helpers for loading optional compiled kernels.

This module centralizes the logic for importing f2py-built extension modules
and determining which calling convention they expose. Energy modules should
use these helpers instead of re-implementing import heuristics.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class KernelSpec:
    """Resolved kernel callable with metadata."""

    func: Callable
    expects_transpose: bool


_SURFACE_KERNEL: KernelSpec | None | bool = None


def get_surface_energy_kernel() -> KernelSpec | None:
    """Return the compiled surface-energy kernel, if available.

    If the env var `MEMBRANE_DISABLE_FORTRAN_SURFACE` is set to a truthy value
    ("1", "true"), returns None.

    The wrapper detects whether the compiled module expects the old f2py
    signature (pos bounds (3,nv), tri bounds (3,nf)) or the newer native-layout
    signature (pos bounds (nv,3), tri bounds (nf,3)).
    """

    global _SURFACE_KERNEL
    if _SURFACE_KERNEL is False:
        return None
    if isinstance(_SURFACE_KERNEL, KernelSpec):
        return _SURFACE_KERNEL

    if os.environ.get("MEMBRANE_DISABLE_FORTRAN_SURFACE") in {"1", "true", "TRUE"}:
        _SURFACE_KERNEL = False
        return None

    candidates = []
    try:
        import fortran_kernels.surface_energy as fe  # type: ignore

        candidates.append(fe)
    except Exception:
        pass

    try:
        import surface_energy as fe  # type: ignore

        candidates.append(fe)
    except Exception:
        pass

    for mod in candidates:
        fn = getattr(mod, "surface_energy_and_gradient", None)
        if not callable(fn):
            submod = getattr(mod, "surface_energy_mod", None)
            fn = (
                getattr(submod, "surface_energy_and_gradient", None) if submod else None
            )
        if not callable(fn):
            continue

        doc = getattr(fn, "__doc__", "") or ""
        expects_transpose = "bounds (3,nv)" in doc or "bounds (3, nv)" in doc
        _SURFACE_KERNEL = KernelSpec(func=fn, expects_transpose=expects_transpose)
        return _SURFACE_KERNEL

    _SURFACE_KERNEL = False
    return None
