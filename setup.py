from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


class build_py(_build_py):
    """Optionally build f2py kernels during packaging.

    Set `MEMBRANE_SOLVER_BUILD_EXT=1` to attempt building the optional Fortran
    kernels as part of `pip install .` / `pip install -e .`. Build failures do
    not abort installation; the runtime always falls back to NumPy kernels.
    """

    def run(self) -> None:
        super().run()

        if not _truthy_env("MEMBRANE_SOLVER_BUILD_EXT"):
            return

        try:
            from membrane_solver.build_ext import build_extensions
        except Exception as exc:
            print(
                f"membrane-solver: could not import build helper; skipping optional kernels: {exc}",
                file=sys.stderr,
            )
            return

        try:
            repo_root = Path(__file__).resolve().parent
            source_dir = repo_root / "fortran_kernels"
            target_dir = Path(self.build_lib) / "fortran_kernels"
            build_extensions(source_dir=source_dir, target_dir=target_dir)
        except Exception as exc:
            print(
                "membrane-solver: optional kernel build failed; continuing without compiled extensions: "
                f"{exc}",
                file=sys.stderr,
            )


setup(cmdclass={"build_py": build_py})
