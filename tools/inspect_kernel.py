import os
import sys

# Ensure we can import the module
sys.path.append(os.getcwd())

try:
    from fortran_kernels import bending_kernels

    print("Docstring for apply_beltrami_laplacian:")
    print(bending_kernels.apply_beltrami_laplacian.__doc__)
except ImportError:
    print("Could not import bending_kernels. Make sure they are compiled.")
except AttributeError:
    print("bending_kernels found, but apply_beltrami_laplacian not found.")
    try:
        print(bending_kernels.bending_kernels_mod.apply_beltrami_laplacian.__doc__)
    except AttributeError:
        print("Could not find apply_beltrami_laplacian in bending_kernels_mod either.")
