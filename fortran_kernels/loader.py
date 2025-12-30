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
_BENDING_GRAD_COTAN: KernelSpec | None | bool = None
_BENDING_LAPLACIAN: KernelSpec | None | bool = None


def _fortran_opt_in_enabled() -> bool:
    """Return True if compiled kernels are allowed to load.

    Compiled kernels are opt-in to keep default installs pure-Python and avoid
    surprising performance differences or import issues on systems without a
    working compiler toolchain.
    """
    return os.environ.get("MEMBRANE_ENABLE_FORTRAN") in {"1", "true", "TRUE"}


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

    if not _fortran_opt_in_enabled():
        _SURFACE_KERNEL = False
        return None

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


def get_bending_grad_cotan_kernel() -> KernelSpec | None:
    """Return the compiled bending `grad_cotan_batch` kernel, if available.

    Disabled when `MEMBRANE_DISABLE_FORTRAN_BENDING=1` or
    `MEMBRANE_DISABLE_FORTRAN_BENDING_GRAD_COTAN=1`.
    """
    global _BENDING_GRAD_COTAN
    if _BENDING_GRAD_COTAN is False:
        return None
    if isinstance(_BENDING_GRAD_COTAN, KernelSpec):
        return _BENDING_GRAD_COTAN

    if not _fortran_opt_in_enabled():
        _BENDING_GRAD_COTAN = False
        return None

    if os.environ.get("MEMBRANE_DISABLE_FORTRAN_BENDING") in {"1", "true", "TRUE"}:
        _BENDING_GRAD_COTAN = False
        return None
    if os.environ.get("MEMBRANE_DISABLE_FORTRAN_BENDING_GRAD_COTAN") in {
        "1",
        "true",
        "TRUE",
    }:
        _BENDING_GRAD_COTAN = False
        return None

    candidates = []
    try:
        import fortran_kernels.bending_kernels as bk  # type: ignore

        candidates.append(bk)
    except Exception:
        pass
    try:
        import bending_kernels as bk  # type: ignore

        candidates.append(bk)
    except Exception:
        pass

    for mod in candidates:
        fn = getattr(mod, "grad_cotan_batch", None)
        if not callable(fn):
            submod = getattr(mod, "bending_kernels_mod", None)
            fn = getattr(submod, "grad_cotan_batch", None) if submod else None
        if not callable(fn):
            continue

        doc = getattr(fn, "__doc__", "") or ""
        expects_transpose = "bounds (3," in doc
        _BENDING_GRAD_COTAN = KernelSpec(func=fn, expects_transpose=expects_transpose)
        return _BENDING_GRAD_COTAN

    _BENDING_GRAD_COTAN = False
    return None


def get_bending_laplacian_kernel() -> KernelSpec | None:
    """Return the compiled bending `apply_beltrami_laplacian` kernel, if available.

    Disabled when `MEMBRANE_DISABLE_FORTRAN_BENDING=1` or
    `MEMBRANE_DISABLE_FORTRAN_BENDING_LAPLACIAN=1`.
    """
    global _BENDING_LAPLACIAN
    if _BENDING_LAPLACIAN is False:
        return None
    if isinstance(_BENDING_LAPLACIAN, KernelSpec):
        return _BENDING_LAPLACIAN

    if not _fortran_opt_in_enabled():
        _BENDING_LAPLACIAN = False
        return None

    if os.environ.get("MEMBRANE_DISABLE_FORTRAN_BENDING") in {"1", "true", "TRUE"}:
        _BENDING_LAPLACIAN = False
        return None
    if os.environ.get("MEMBRANE_DISABLE_FORTRAN_BENDING_LAPLACIAN") in {
        "1",
        "true",
        "TRUE",
    }:
        _BENDING_LAPLACIAN = False
        return None

    candidates = []
    try:
        import fortran_kernels.bending_kernels as bk  # type: ignore

        candidates.append(bk)
    except Exception:
        pass
    try:
        import bending_kernels as bk  # type: ignore

        candidates.append(bk)
    except Exception:
        pass

    for mod in candidates:
        fn = getattr(mod, "apply_beltrami_laplacian", None)
        if not callable(fn):
            submod = getattr(mod, "bending_kernels_mod", None)
            fn = getattr(submod, "apply_beltrami_laplacian", None) if submod else None
        if not callable(fn):
            continue

        doc = getattr(fn, "__doc__", "") or ""
        expects_transpose = "bounds (3," in doc
        _BENDING_LAPLACIAN = KernelSpec(func=fn, expects_transpose=expects_transpose)
        return _BENDING_LAPLACIAN

    _BENDING_LAPLACIAN = False
    return None
