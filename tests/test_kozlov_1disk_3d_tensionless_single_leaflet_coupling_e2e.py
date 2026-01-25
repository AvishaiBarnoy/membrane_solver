import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


def _annulus_single_leaflet_drive_data(
    *,
    n_theta: int = 12,
    n_rings: int = 6,
    r_in: float = 1.0,
    r_out: float = 6.0,
    symmetry_break_z: float = 1e-3,
    hard_source_amplitude: float = 4.0,
    soft_source_strength: float = 40.0,
    drive: str,
) -> dict:
    """Return a tensionless (γ=0) annulus for single-leaflet coupling checks.

    This is the "Milestone D" mechanism in the symmetric regime of
    `docs/tex/1_disk_3d.pdf`:
      - drive only the inner leaflet on the inner rim,
      - allow midplane shape relaxation,
      - verify that opposite-leaflet tilt develops via the `bending_tilt_*`
        coupling through curvature.
    """
    if drive not in ("hard", "soft"):
        raise ValueError("drive must be 'hard' or 'soft'")
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
            z = float(symmetry_break_z) if j == 1 else 0.0

            if j == 0:
                opts = {"preset": "inner_rim"}
                if drive == "hard":
                    opts["tilt_fixed_in"] = True
                    opts["tilt_in"] = [
                        float(hard_source_amplitude * np.cos(theta)),
                        float(hard_source_amplitude * np.sin(theta)),
                    ]
                vertices.append([x, y, z, opts])
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
    for ring in range(int(n_rings) + 1):
        for k in range(int(n_theta)):
            edges.append([vid(ring, k), vid(ring, (k + 1) % n_theta)])

    for ring in range(int(n_rings)):
        for k in range(int(n_theta)):
            edges.append([vid(ring, k), vid(ring + 1, k)])

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

    energy_modules = [
        "bending_tilt_in",
        "bending_tilt_out",
        "tilt_in",
        "tilt_out",
    ]
    global_parameters: dict = {
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
        "tilt_solve_mode": "nested",
        "tilt_step_size": 0.15,
        "tilt_inner_steps": 25,
        "tilt_tol": 1e-10,
        "step_size": 0.006,
        "step_size_mode": "fixed",
        "pin_to_plane_normal": [0.0, 0.0, 1.0],
        "pin_to_plane_point": [0.0, 0.0, 0.0],
    }
    if drive == "soft":
        energy_modules.append("tilt_rim_source_in")
        global_parameters.update(
            {
                "tilt_rim_source_center": [0.0, 0.0, 0.0],
                "tilt_rim_source_group_in": "inner",
                "tilt_rim_source_strength_in": float(soft_source_strength),
            }
        )

    return {
        "global_parameters": global_parameters,
        "constraint_modules": ["pin_to_plane", "pin_to_circle"],
        "definitions": definitions,
        "energy_modules": energy_modules,
        "vertices": vertices,
        "edges": edges,
        "faces": faces,
        "instructions": [],
    }


def _build_minimizer(mesh) -> Minimizer:
    """Return a minimizer configured for quick e2e checks."""
    return Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )


def _radial_components(mesh) -> tuple[np.ndarray, np.ndarray]:
    """Return radial components (θ_in, θ_out) using XY radial unit vectors."""
    positions = mesh.positions_view()
    r = np.linalg.norm(positions[:, :2], axis=1)
    r_hat = np.zeros_like(positions)
    good = r > 1e-12
    r_hat[good, 0] = positions[good, 0] / r[good]
    r_hat[good, 1] = positions[good, 1] / r[good]
    th_in = np.einsum("ij,ij->i", mesh.tilts_in_view(), r_hat)
    th_out = np.einsum("ij,ij->i", mesh.tilts_out_view(), r_hat)
    return th_in, th_out


def _free_vertex_mask(mesh) -> np.ndarray:
    """Return a boolean mask for vertices not on the outer clamped rim."""
    outer_rows = [
        mesh.vertex_index_to_row[int(vid)]
        for vid in mesh.vertex_ids
        if mesh.vertices[int(vid)].options.get("pin_to_circle_group") == "outer"
    ]
    is_outer = np.zeros(len(mesh.vertex_ids), dtype=bool)
    is_outer[np.asarray(outer_rows, dtype=int)] = True
    return ~is_outer


def test_kozlov_1disk_3d_tensionless_single_leaflet_hard_source_couples_leaflets() -> (
    None
):
    """Hard inner-leaflet drive should induce tilt in the opposite leaflet via curvature."""
    mesh = parse_geometry(_annulus_single_leaflet_drive_data(drive="hard"))
    minim = _build_minimizer(mesh)
    minim.minimize(n_steps=15)

    th_in, th_out = _radial_components(mesh)
    free = _free_vertex_mask(mesh)

    max_in = float(np.max(np.abs(th_in[free])))
    max_out = float(np.max(np.abs(th_out[free])))
    assert max_in > 0.5
    assert max_out > 1e-3
    assert max_out / (max_in + 1e-12) > 0.1

    z = mesh.positions_view()[:, 2]
    assert float(np.ptp(z)) > 1e-2


def test_kozlov_1disk_3d_tensionless_single_leaflet_soft_source_couples_leaflets() -> (
    None
):
    """Soft inner-leaflet source should induce tilt in the opposite leaflet via curvature."""
    mesh = parse_geometry(_annulus_single_leaflet_drive_data(drive="soft"))
    minim = _build_minimizer(mesh)
    minim.minimize(n_steps=15)

    th_in, th_out = _radial_components(mesh)
    free = _free_vertex_mask(mesh)

    max_in = float(np.max(np.abs(th_in[free])))
    max_out = float(np.max(np.abs(th_out[free])))
    assert max_in > 1e-2
    assert max_out > 1e-3

    z = mesh.positions_view()[:, 2]
    assert float(np.ptp(z)) > 1e-3


def test_kozlov_1disk_3d_without_bending_tilt_out_keeps_opposite_leaflet_zeroish() -> (
    None
):
    """Control: without bending_tilt_out there is no shape-mediated coupling into θ_out."""
    data = _annulus_single_leaflet_drive_data(drive="hard")
    data["energy_modules"] = [
        m for m in data["energy_modules"] if m != "bending_tilt_out"
    ]
    mesh = parse_geometry(data)
    minim = _build_minimizer(mesh)
    minim.minimize(n_steps=15)

    _th_in, th_out = _radial_components(mesh)
    free = _free_vertex_mask(mesh)
    assert float(np.max(np.abs(th_out[free]))) < 1e-6
