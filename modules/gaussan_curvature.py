# gaussian_curvature.py
import numpy as np


def compute_energy_and_gradient(mesh, gp, resolver):
    κbar = resolver.get(None, 'gaussian_modulus')
    E = 0.0
    shape_grad = {i: np.zeros(3) for i in mesh.vertices}
    # tilt_grad zeros, no tilt dependence
    tilt_grad  = {i: np.zeros(2) for i in mesh.vertices}

    for v in mesh.vertices.values():
        # TODO: implement discrete_gauss_curvature voronoi_area
        K = v.discrete_gauss_curvature()
        A = v.voronoi_area()
        E_loc = κbar * K * A
        E += E_loc

        # ∂E/∂x comes from ∂K/∂x and ∂A/∂x
        dK_dx = v.dK_dvertex()
        dA_dx = v.dA_dvertex()
        shape_grad[v.index] += κbar * (K * dA_dx + A * dK_dx)

    return E, shape_grad, tilt_grad
