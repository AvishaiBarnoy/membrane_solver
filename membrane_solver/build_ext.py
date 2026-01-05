"""Build optional compiled kernels (Fortran / f2py).

This module never compiles on import. Building is only performed when:
- you run `python -m membrane_solver.build_ext`, or
- you set `MEMBRANE_SOLVER_BUILD_EXT=1` during `pip install .` / `pip install -e .`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class BuildResult:
    """Result of building a single kernel."""

    name: str
    output_dir: Path


def _run(cmd: Sequence[str], *, cwd: Path) -> None:
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout.strip() or f"Command failed: {' '.join(cmd)}")


def build_extensions(
    *,
    source_dir: Path,
    target_dir: Path,
    kernels: Sequence[str] = ("surface_energy", "bending_kernels"),
) -> list[BuildResult]:
    """Build selected f2py kernels into `target_dir`.

    Parameters
    ----------
    source_dir:
        Directory containing Fortran sources like `surface_energy.f90`.
    target_dir:
        Output directory where the compiled extension modules should be placed.
        For in-place builds, pass the installed `fortran_kernels/` directory.
    kernels:
        Iterable of kernel module names to build (without file extension).

    Returns
    -------
    list[BuildResult]
        A list of successful build results.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    results: list[BuildResult] = []
    for kernel in kernels:
        source_path = source_dir / f"{kernel}.f90"
        if not source_path.exists():
            raise FileNotFoundError(f"Missing Fortran source: {source_path}")

        cmd = [
            sys.executable,
            "-m",
            "numpy.f2py",
            "-c",
            "-m",
            kernel,
            str(source_path),
        ]
        _run(cmd, cwd=target_dir)
        results.append(BuildResult(name=kernel, output_dir=target_dir))

    return results


def _resolve_installed_fortran_dir() -> Path:
    try:
        import fortran_kernels  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Could not import `fortran_kernels`. Run this command from the repo root "
            "or install the package first."
        ) from exc

    package_path = Path(fortran_kernels.__file__).resolve()
    return package_path.parent


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=None,
        help=(
            "Where to write the compiled extension modules. Defaults to the installed "
            "`fortran_kernels/` package directory."
        ),
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help=(
            "Where to read the Fortran sources from. Defaults to the same directory as "
            "`--target-dir`."
        ),
    )
    parser.add_argument(
        "--kernels",
        nargs="*",
        default=["surface_energy", "bending_kernels"],
        help="Kernels to build (default: surface_energy bending_kernels).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    target_dir = args.target_dir or _resolve_installed_fortran_dir()
    source_dir = args.source_dir or target_dir

    try:
        build_extensions(
            source_dir=source_dir,
            target_dir=target_dir,
            kernels=tuple(args.kernels),
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Built kernels into {target_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
