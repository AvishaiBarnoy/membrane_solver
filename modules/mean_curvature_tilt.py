import numpy as np


def compute_energy_and_gradient(mesh, gp, resolver):
    E = 0.0
    shape_grad = {i: np.zeros(3) for i in mesh.vertices}
    tilt_grad  = {i: np.zeros(2) for i in mesh.vertices}

    for facet in mesh.facets.values():
        # 1. Evaluate discrete mean curvature J at this facet
        J = facet.compute_mean_curvature()

        # 2. Evaluate discrete divergence of tilt: ∇·t on this facet
        div_t = facet.compute_divergence_of_tilt()

        # 3. Parameter J0 = resolver.get(facet, 'rest_curvature')
        J0 = resolver.get(facet, 'spontaneous_curvature')

        κ = resolver.get(facet, 'bending_rigidity')

        # local energy density: ½κ (J − J0 + div_t)^2
        delta = J - J0 + div_t
        A = facet.area()
        E_loc = 0.5 * κ * (delta**2) * A
        E += E_loc

        # now get ∂E/∂J, ∂E/∂(div_t):
        dE_dJ     = κ * delta * A
        dE_d_divt = κ * delta * A

        # decorate shape_grad at each of the 3 vertices
        for vidx in facet.vertex_indices:
            # discrete ∂J/∂x_i  → a 3-vector
            dJ_dxi = facet.dJ_dvertex(vidx)
            shape_grad[vidx] += dE_dJ * dJ_dxi

            # discrete ∂(div_t)/∂x_i  → also contributes to shape_grad if tilt‐gradient coupling
            ddivt_dxi = facet.dDivT_dvertex(vidx)
            shape_grad[vidx] += dE_d_divt * ddivt_dxi

            # discrete ∂(div_t)/∂t_i  → a 2-vector in the tangent plane
            ddivt_dti = facet.dDivT_dtilt(vidx)
            tilt_grad[vidx] += dE_d_divt * ddivt_dti

    return E, shape_grad, tilt_grad
