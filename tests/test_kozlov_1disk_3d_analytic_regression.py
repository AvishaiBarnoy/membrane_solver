import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


def _annulus_data(
    *,
    n_theta: int = 32,
    n_rings: int = 10,
    r_in: float = 1.0,
    r_out: float = 6.0,
    rim_source_strength: float = 50.0,
) -> dict:
    """Return a bilayer annulus configured to match `docs/tex/1_disk_3d.pdf`.

    For the tensionless case (surface_tension=0), the analytic Euler-Lagrange
    equations imply the exact identity θ^p(r)=θ^d(r) for the outer membrane
    (r >= R). We test this by driving the inner rim with *the same* soft
    boundary source on both leaflets and checking that the relaxed radial tilt
    components match.
    """
    if n_theta < 8:
        raise ValueError("n_theta must be >= 8")
    if n_rings < 3:
        raise ValueError("n_rings must be >= 3")
    if r_out <= r_in:
        raise ValueError("r_out must be > r_in")

    radii = np.linspace(float(r_in), float(r_out), int(n_rings) + 1)

    definitions = {
        "inner_rim": {
            "constraints": ["pin_to_circle"],
            "pin_to_circle_group": "inner",
            "pin_to_circle_radius": float(r_in),
            "pin_to_circle_normal": [0.0, 0.0, 1.0],
            "pin_to_circle_point": [0.0, 0.0, 0.0],
            "pin_to_circle_mode": "fixed",
        },
        "outer_rim": {
            "constraints": ["pin_to_circle", "pin_to_plane"],
            "pin_to_circle_group": "outer",
            "pin_to_circle_radius": float(r_out),
            "pin_to_circle_normal": [0.0, 0.0, 1.0],
            "pin_to_circle_point": [0.0, 0.0, 0.0],
            "pin_to_circle_mode": "fixed",
            "tilt_fixed_in": True,
            "tilt_fixed_out": True,
        },
    }

    vertices: list[list] = []
    for j, r in enumerate(radii):
        for i in range(n_theta):
            theta = 2.0 * np.pi * i / float(n_theta)
            x = float(r * np.cos(theta))
            y = float(r * np.sin(theta))
            z = 0.0
            if j == 0:
                vertices.append([x, y, z, {"preset": "inner_rim"}])
            elif j == len(radii) - 1:
                vertices.append(
                    [
                        x,
                        y,
                        z,
                        {
                            "preset": "outer_rim",
                            "tilt_in": [0.0, 0.0],
                            "tilt_out": [0.0, 0.0],
                        },
                    ]
                )
            else:
                vertices.append([x, y, z])

    def vid(ring: int, k: int) -> int:
        return int(ring) * int(n_theta) + int(k)

    edges: list[list[int]] = []

    # Ring edges for every ring.
    for ring in range(int(n_rings) + 1):
        for k in range(int(n_theta)):
            edges.append([vid(ring, k), vid(ring, (k + 1) % n_theta)])

    # Radial edges between successive rings.
    for ring in range(int(n_rings)):
        for k in range(int(n_theta)):
            edges.append([vid(ring, k), vid(ring + 1, k)])

    # Diagonal edges to triangulate each quad.
    for ring in range(int(n_rings)):
        for k in range(int(n_theta)):
            edges.append([vid(ring, k), vid(ring + 1, (k + 1) % n_theta)])

    edge_index_by_pair: dict[tuple[int, int], int] = {}
    for idx, (tail, head, *_rest) in enumerate(edges):
        edge_index_by_pair[(int(tail), int(head))] = int(idx)

    def edge_ref(tail: int, head: int) -> int | str:
        forward = edge_index_by_pair.get((int(tail), int(head)))
        if forward is not None:
            return forward
        reverse = edge_index_by_pair.get((int(head), int(tail)))
        if reverse is not None:
            return f"r{reverse}"
        raise KeyError(f"Missing edge for face: {tail}->{head}")

    faces: list[list] = []
    for ring in range(int(n_rings)):
        for k in range(int(n_theta)):
            k1 = (k + 1) % n_theta
            v00 = vid(ring, k)
            v01 = vid(ring, k1)
            v10 = vid(ring + 1, k)
            v11 = vid(ring + 1, k1)

            faces.append([edge_ref(v00, v01), edge_ref(v01, v11), edge_ref(v11, v00)])
            faces.append([edge_ref(v00, v11), edge_ref(v11, v10), edge_ref(v10, v00)])

    return {
        "global_parameters": {
            "surface_tension": 0.0,
            "bending_energy_model": "helfrich",
            "spontaneous_curvature": 0.0,
            # Match the analytic normalization in docs/tex/1_disk_3d.pdf:
            # the continuum integrand uses κ(J±Dθ)^2 + κ_t θ^2 (no 1/2 factors),
            # while our modules include 1/2. Using 2.0 here keeps units aligned.
            "bending_modulus_in": 2.0,
            "bending_modulus_out": 2.0,
            "tilt_modulus_in": 2.0,
            "tilt_modulus_out": 2.0,
            "tilt_rim_source_center": [0.0, 0.0, 0.0],
            "tilt_rim_source_group_in": "inner",
            "tilt_rim_source_strength_in": float(rim_source_strength),
            "tilt_rim_source_group_out": "inner",
            "tilt_rim_source_strength_out": float(rim_source_strength),
            # Relax only tilts on the flat annulus (outer rim clamps the far field).
            "tilt_solve_mode": "nested",
            "tilt_step_size": 0.05,
            "tilt_inner_steps": 800,
            "tilt_tol": 1e-10,
            "step_size": 0.0,
            "step_size_mode": "fixed",
            "pin_to_plane_normal": [0.0, 0.0, 1.0],
            "pin_to_plane_point": [0.0, 0.0, 0.0],
        },
        "constraint_modules": ["pin_to_plane", "pin_to_circle"],
        "definitions": definitions,
        "energy_modules": [
            "bending_tilt_in",
            "bending_tilt_out",
            "tilt_in",
            "tilt_out",
            "tilt_rim_source_in",
            "tilt_rim_source_out",
        ],
        "vertices": vertices,
        "edges": edges,
        "faces": faces,
        "instructions": [],
    }


def _build_minimizer(mesh) -> Minimizer:
    return Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )


def _radial_components(mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = mesh.positions_view()
    r = np.linalg.norm(positions[:, :2], axis=1)
    r_hat = np.zeros_like(positions)
    good = r > 1e-12
    r_hat[good, 0] = positions[good, 0] / r[good]
    r_hat[good, 1] = positions[good, 1] / r[good]

    th_in = np.einsum("ij,ij->i", mesh.tilts_in_view(), r_hat)
    th_out = np.einsum("ij,ij->i", mesh.tilts_out_view(), r_hat)
    return r, th_in, th_out


def test_kozlov_1disk_3d_tensionless_equal_leaflet_tilts():
    """Regression for docs/tex/1_disk_3d.pdf (γ=0 ⇒ θ^p(r)=θ^d(r))."""
    r_in = 1.0
    r_out = 6.0
    mesh = parse_geometry(_annulus_data(r_in=r_in, r_out=r_out))
    minim = _build_minimizer(mesh)

    # Only relax tilts (shape step is disabled via step_size=0).
    minim._relax_leaflet_tilts(positions=mesh.positions_view(), mode="nested")

    r, th_in, th_out = _radial_components(mesh)
    outer_rows = [
        mesh.vertex_index_to_row[int(vid)]
        for vid in mesh.vertex_ids
        if mesh.vertices[int(vid)].options.get("pin_to_circle_group") == "outer"
    ]
    is_outer = np.zeros(len(mesh.vertex_ids), dtype=bool)
    is_outer[np.asarray(outer_rows, dtype=int)] = True
    free = ~is_outer

    assert float(np.max(np.abs(th_in[free]))) > 1e-3
    assert float(np.max(np.abs(th_out[free]))) > 1e-3

    diff = th_out[free] - th_in[free]
    rel = float(np.linalg.norm(diff) / (np.linalg.norm(th_in[free]) + 1e-12))
    assert rel < 2e-2

    # Ensure the large annulus approximates the "decay to 0 at infinity" regime:
    # the last free ring should be essentially relaxed.
    dr = (r_out - r_in) / 10.0
    near_outer = free & (r > (r_out - 1.5 * dr))
    max_in = float(np.max(np.abs(th_in[free])))
    max_out = float(np.max(np.abs(th_out[free])))
    assert float(np.max(np.abs(th_in[near_outer]))) < 1e-2 * max_in
    assert float(np.max(np.abs(th_out[near_outer]))) < 1e-2 * max_out
