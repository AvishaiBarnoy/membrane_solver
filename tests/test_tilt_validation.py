import numpy as np
import pytest
from sample_meshes import cube_soft_volume_input

from core.parameters.global_parameters import GlobalParameters
from geometry.entities import Edge, Facet, Mesh, Vertex
from geometry.geom_io import parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.refinement import refine_triangle_mesh
from runtime.steppers.conjugate_gradient import ConjugateGradient
from runtime.steppers.gradient_descent import GradientDescent


def _planar_grid_mesh(n: int) -> tuple[Mesh, dict[tuple[int, int], int]]:
    mesh = Mesh()
    mesh.global_parameters = GlobalParameters({"surface_tension": 0.0})
    mesh.energy_modules = []
    mesh.constraint_modules = []

    vid: dict[tuple[int, int], int] = {}
    idx = 0
    for j in range(n + 1):
        for i in range(n + 1):
            vid[(i, j)] = idx
            mesh.vertices[idx] = Vertex(
                idx,
                np.array([i / n, j / n, 0.0], dtype=float),
                fixed=True,
            )
            idx += 1

    edge_map: dict[tuple[int, int], int] = {}
    next_eid = 1
    next_fid = 0

    def add_triangle(a: int, b: int, c: int) -> None:
        nonlocal next_eid, next_fid
        e_ids: list[int] = []
        for u, v in ((a, b), (b, c), (c, a)):
            key = (min(u, v), max(u, v))
            if key not in edge_map:
                edge_map[key] = next_eid
                mesh.edges[next_eid] = Edge(next_eid, u, v)
                next_eid += 1
            eid = edge_map[key]
            edge = mesh.edges[eid]
            e_ids.append(eid if edge.tail_index == u else -eid)

        mesh.facets[next_fid] = Facet(next_fid, e_ids)
        next_fid += 1

    for j in range(n):
        for i in range(n):
            v00 = vid[(i, j)]
            v10 = vid[(i + 1, j)]
            v01 = vid[(i, j + 1)]
            v11 = vid[(i + 1, j + 1)]
            add_triangle(v00, v10, v11)
            add_triangle(v00, v11, v01)

    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()
    mesh.build_position_cache()
    return mesh, vid


def _run_tilt_only_relaxation(mesh: Mesh) -> float:
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        tol=0.0,
        quiet=True,
    )
    e0 = minim.compute_energy()
    minim.minimize(n_steps=1)
    e1 = minim.compute_energy()
    assert e1 <= e0 + 1e-12
    return float(e1)


def test_tilt_source_decay_on_planar_patch():
    mesh, vid = _planar_grid_mesh(10)

    for vertex in mesh.vertices.values():
        vertex.tilt = np.zeros(3, dtype=float)
        vertex.tilt_fixed = False

    # Boundary condition: clamp tilt to zero on the boundary.
    for (i, j), v_id in vid.items():
        if i in (0, 10) or j in (0, 10):
            mesh.vertices[v_id].tilt_fixed = True
            mesh.vertices[v_id].tilt = np.zeros(3, dtype=float)

    # Single interior source.
    source_vid = vid[(5, 5)]
    mesh.vertices[source_vid].tilt_fixed = True
    mesh.vertices[source_vid].tilt = np.array([1.0, 0.0, 0.0], dtype=float)
    mesh.touch_tilts()

    mesh.energy_modules = ["tilt_smoothness", "tilt"]
    mesh.global_parameters.update(
        {
            "tilt_smoothness_rigidity": 1.0,
            "tilt_rigidity": 0.05,
            "tilt_solve_mode": "nested",
            "tilt_step_size": 0.35,
            "tilt_inner_steps": 300,
            "tilt_tol": 1e-12,
        }
    )

    _run_tilt_only_relaxation(mesh)

    tilts = mesh.tilts_view()
    assert float(np.max(np.abs(tilts[:, 2]))) < 1e-12  # stays tangent to z=0 plane

    positions = mesh.positions_view()
    dist = np.linalg.norm(positions[:, :2] - np.array([0.5, 0.5], dtype=float), axis=1)
    mags = np.linalg.norm(tilts[:, :2], axis=1)
    mask = dist > 1e-12

    near = mags[mask & (dist <= 0.25)]
    mid = mags[mask & (dist > 0.25) & (dist <= 0.5)]
    far = mags[mask & (dist > 0.5)]

    assert near.size > 0 and mid.size > 0 and far.size > 0
    assert float(near.mean()) > float(mid.mean()) > float(far.mean())


def test_tilt_tangency_projection_on_closed_mesh():
    """B0.1: projecting tilt on a closed mesh removes normal components."""
    data = cube_soft_volume_input(volume_mode="lagrange")
    mesh = parse_geometry(data)

    rng = np.random.default_rng(0)
    for vid in mesh.vertices:
        mesh.vertices[vid].tilt = rng.normal(size=3)
    mesh.touch_tilts()

    mesh.project_tilts_to_tangent()

    normals = mesh.vertex_normals()
    tilts = mesh.tilts_view()
    dot = np.einsum("ij,ij->i", normals, tilts)
    assert float(np.max(np.abs(dot))) < 1e-12


def test_tilt_opposite_sources_cancel_at_midpoint():
    mesh_single, vid = _planar_grid_mesh(10)
    mesh_dipole = mesh_single.copy()

    def setup(mesh: Mesh, *, sources: dict[tuple[int, int], np.ndarray]) -> None:
        for vertex in mesh.vertices.values():
            vertex.tilt = np.zeros(3, dtype=float)
            vertex.tilt_fixed = False
            vertex.fixed = True

        for (i, j), v_id in vid.items():
            if i in (0, 10) or j in (0, 10):
                mesh.vertices[v_id].tilt_fixed = True
                mesh.vertices[v_id].tilt = np.zeros(3, dtype=float)

        for ij, vec in sources.items():
            v_id = vid[ij]
            mesh.vertices[v_id].tilt_fixed = True
            mesh.vertices[v_id].tilt = np.asarray(vec, dtype=float)

        mesh.touch_tilts()
        mesh.energy_modules = ["tilt_smoothness", "tilt"]
        mesh.global_parameters.update(
            {
                "tilt_smoothness_rigidity": 1.0,
                "tilt_rigidity": 0.1,
                "tilt_solve_mode": "nested",
                "tilt_step_size": 0.35,
                "tilt_inner_steps": 350,
                "tilt_tol": 1e-12,
            }
        )

    setup(mesh_single, sources={(4, 5): np.array([1.0, 0.0, 0.0])})
    setup(
        mesh_dipole,
        sources={
            (4, 5): np.array([1.0, 0.0, 0.0]),
            (6, 5): np.array([-1.0, 0.0, 0.0]),
        },
    )

    _run_tilt_only_relaxation(mesh_single)
    _run_tilt_only_relaxation(mesh_dipole)

    center_row = mesh_single.vertex_index_to_row[vid[(5, 5)]]
    single_mag = float(np.linalg.norm(mesh_single.tilts_view()[center_row]))
    dipole_mag = float(np.linalg.norm(mesh_dipole.tilts_view()[center_row]))

    assert single_mag > 1e-3
    assert dipole_mag < 0.35 * single_mag
    assert dipole_mag < 0.15


def _tilted_cube_mesh(*, rng: np.random.Generator) -> Mesh:
    data = cube_soft_volume_input(volume_mode="lagrange")
    data["energy_modules"] = ["bending_tilt", "tilt"]
    data["global_parameters"].update(
        {
            "bending_modulus": 0.2,
            "spontaneous_curvature": 0.0,
            "bending_energy_model": "helfrich",
            "tilt_rigidity": 0.01,
            "tilt_solve_mode": "nested",
            "tilt_step_size": 0.1,
            "tilt_inner_steps": 30,
            "tilt_tol": 1e-10,
            "step_size": 2e-3,
            "step_size_mode": "fixed",
        }
    )
    mesh = parse_geometry(data)

    mesh.build_position_cache()
    tilts = 0.05 * rng.normal(size=mesh.positions_view().shape)
    mesh.set_tilts_from_array(tilts)
    mesh.project_tilts_to_tangent()
    return mesh


def test_tilt_minimization_preserves_volume_constraint():
    rng = np.random.default_rng(0)
    mesh = _tilted_cube_mesh(rng=rng)

    target_volume = mesh.bodies[0].target_volume
    assert target_volume is not None

    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        ConjugateGradient(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        tol=0.0,
        quiet=True,
    )
    minim.minimize(n_steps=6)

    final_volume = minim.mesh.compute_total_volume()
    rel = abs(final_volume - target_volume) / max(abs(target_volume), 1.0)
    assert rel < 5e-6

    normals = minim.mesh.vertex_normals()
    tilts = minim.mesh.tilts_view()
    dot = np.einsum("ij,ij->i", normals, tilts)
    assert float(np.max(np.abs(dot))) < 1e-10


def test_tilt_pipeline_stable_after_triangle_refinement():
    rng = np.random.default_rng(1)
    mesh = _tilted_cube_mesh(rng=rng)

    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        ConjugateGradient(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        tol=0.0,
        quiet=True,
    )
    minim.minimize(n_steps=3)

    refined = refine_triangle_mesh(minim.mesh)
    refined_min = Minimizer(
        refined,
        refined.global_parameters,
        ConjugateGradient(),
        EnergyModuleManager(refined.energy_modules),
        ConstraintModuleManager(refined.constraint_modules),
        tol=0.0,
        quiet=True,
    )
    refined_min.minimize(n_steps=2)

    target_volume = refined_min.mesh.bodies[0].target_volume
    assert target_volume is not None
    final_volume = refined_min.mesh.compute_total_volume()
    rel = abs(final_volume - target_volume) / max(abs(target_volume), 1.0)
    assert rel < 5e-6

    assert refined_min.mesh.validate_edge_indices()
    assert not np.any(np.isnan(refined_min.mesh.positions_view()))
    assert not np.any(np.isnan(refined_min.mesh.tilts_view()))


def test_refinement_inherits_tilt_fixed_on_boundary_loop_midpoints() -> None:
    """Refinement: midpoints on a fixed-tilt loop keep fixed_tilt and average tilt."""
    mesh = Mesh()
    mesh.global_parameters = GlobalParameters({"surface_tension": 0.0})
    mesh.energy_modules = []
    mesh.constraint_modules = []

    mesh.vertices = {
        0: Vertex(
            0,
            np.array([0.0, 0.0, 0.0], dtype=float),
            fixed=True,
            tilt=np.array([1.0, 0.0, 0.0], dtype=float),
            tilt_fixed=True,
        ),
        1: Vertex(
            1,
            np.array([1.0, 0.0, 0.0], dtype=float),
            fixed=True,
            tilt=np.array([0.0, 1.0, 0.0], dtype=float),
            tilt_fixed=True,
        ),
        2: Vertex(
            2,
            np.array([1.0, 1.0, 0.0], dtype=float),
            fixed=True,
            tilt=np.array([-1.0, 0.0, 0.0], dtype=float),
            tilt_fixed=True,
        ),
        3: Vertex(
            3,
            np.array([0.0, 1.0, 0.0], dtype=float),
            fixed=True,
            tilt=np.array([0.0, -1.0, 0.0], dtype=float),
            tilt_fixed=True,
        ),
    }

    # Boundary edges + diagonal; edges are *not* marked fixed to ensure
    # tilt_fixed inheritance depends only on parent vertices.
    mesh.edges = {
        1: Edge(1, 0, 1),
        2: Edge(2, 1, 2),
        3: Edge(3, 2, 0),  # diagonal (2 -> 0)
        4: Edge(4, 2, 3),
        5: Edge(5, 3, 0),
    }
    mesh.facets = {
        0: Facet(0, [1, 2, 3]),  # (0,1,2)
        1: Facet(1, [-3, 4, 5]),  # (0,2,3)
    }
    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()
    mesh.build_position_cache()
    mesh.touch_tilts()

    refined = refine_triangle_mesh(mesh)

    def find_vertex_by_position(pos: np.ndarray) -> Vertex:
        for v in refined.vertices.values():
            if np.allclose(v.position, pos, atol=1e-12, rtol=0.0):
                return v
        raise AssertionError(
            f"Could not find refined vertex at position {pos.tolist()}"
        )

    checks = [
        # edge (0,1)
        (
            np.array([0.5, 0.0, 0.0], dtype=float),
            0.5 * (mesh.vertices[0].tilt + mesh.vertices[1].tilt),
        ),
        # edge (1,2)
        (
            np.array([1.0, 0.5, 0.0], dtype=float),
            0.5 * (mesh.vertices[1].tilt + mesh.vertices[2].tilt),
        ),
        # edge (2,3)
        (
            np.array([0.5, 1.0, 0.0], dtype=float),
            0.5 * (mesh.vertices[2].tilt + mesh.vertices[3].tilt),
        ),
        # edge (3,0)
        (
            np.array([0.0, 0.5, 0.0], dtype=float),
            0.5 * (mesh.vertices[3].tilt + mesh.vertices[0].tilt),
        ),
    ]

    for midpoint_pos, expected_tilt in checks:
        v = find_vertex_by_position(midpoint_pos)
        assert v.tilt_fixed is True
        assert v.tilt == pytest.approx(expected_tilt, rel=0.0, abs=1e-12)


def test_refinement_midpoint_tilt_is_averaged_when_one_parent_is_not_tilt_fixed() -> (
    None
):
    """Refinement: midpoint averages tilt but only inherits tilt_fixed if both parents are fixed."""
    mesh = Mesh()
    mesh.global_parameters = GlobalParameters({"surface_tension": 0.0})
    mesh.energy_modules = []
    mesh.constraint_modules = []

    mesh.vertices = {
        0: Vertex(
            0,
            np.array([0.0, 0.0, 0.0], dtype=float),
            fixed=True,
            tilt=np.array([1.0, 0.0, 0.0], dtype=float),
            tilt_fixed=True,
        ),
        1: Vertex(
            1,
            np.array([1.0, 0.0, 0.0], dtype=float),
            fixed=True,
            tilt=np.array([0.0, 1.0, 0.0], dtype=float),
            tilt_fixed=False,
        ),
        2: Vertex(
            2,
            np.array([0.0, 1.0, 0.0], dtype=float),
            fixed=True,
            tilt=np.array([0.0, 0.0, 0.0], dtype=float),
            tilt_fixed=False,
        ),
    }

    mesh.edges = {
        1: Edge(1, 0, 1),
        2: Edge(2, 1, 2),
        3: Edge(3, 2, 0),
    }
    mesh.facets = {0: Facet(0, [1, 2, 3])}
    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()
    mesh.build_position_cache()
    mesh.touch_tilts()

    refined = refine_triangle_mesh(mesh)

    midpoint = None
    for v in refined.vertices.values():
        if np.allclose(v.position, np.array([0.5, 0.0, 0.0], dtype=float), atol=1e-12):
            midpoint = v
            break
    assert midpoint is not None
    assert midpoint.tilt_fixed is False
    assert midpoint.tilt == pytest.approx(
        0.5 * (mesh.vertices[0].tilt + mesh.vertices[1].tilt),
        rel=0.0,
        abs=1e-12,
    )
