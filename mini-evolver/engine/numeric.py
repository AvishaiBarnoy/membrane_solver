"""Numeric backend selection."""

try:
    import numpy as np  # type: ignore

    HAVE_NUMPY = True
except Exception:
    np = None  # type: ignore[assignment]
    HAVE_NUMPY = False

USE_NUMPY = HAVE_NUMPY


def set_use_numpy(flag: bool) -> None:
    global USE_NUMPY
    USE_NUMPY = bool(flag and HAVE_NUMPY)
