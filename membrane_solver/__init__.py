"""Package utilities for membrane-solver.

The solver core currently lives in top-level packages like `geometry/`,
`modules/`, and `runtime/`. This package exists to provide stable helper entry
points such as `python -m membrane_solver.build_ext`.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("membrane-solver")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"
