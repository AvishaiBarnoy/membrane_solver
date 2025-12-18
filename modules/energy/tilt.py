# tilt.py
import numpy as np


def compute_energy_and_gradient(mesh, gp, resolver):
    κt = resolver.get(None, "tilt_rigidity")
    E = 0.0
    shape_grad = {i: np.zeros(3) for i in mesh.vertices}
    tilt_grad = {i: np.zeros(2) for i in mesh.vertices}

    for v in mesh.vertices.values():
        t = v.tilt
        # TODO: implement vornoi_area
        A = v.voronoi_area()
        E_loc = 0.5 * κt * np.dot(t, t) * A
        E += E_loc

        # ∂E/∂t = κt * t * A
        tilt_grad[v.index] += κt * t * A

        # optionally ∂A/∂x contributes to shape_grad, but often small—omit or include if you wish
    return E, shape_grad, tilt_grad
