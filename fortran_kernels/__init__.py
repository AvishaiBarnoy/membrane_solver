"""Optional compiled kernels.

This package is intended to hold compiled extensions (e.g. Fortran via f2py)
that accelerate the NumPy hot loops. The pure-Python engine always remains the
fallback when these kernels are not available.
"""
