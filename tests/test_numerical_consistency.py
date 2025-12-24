import numpy as np
import pytest

from geometry.entities import Body, Edge, Facet, Mesh, Vertex
from modules.energy import bending, surface, volume
from parameters.global_parameters import GlobalParameters
from parameters.resolver import ParameterResolver


def create_random_mesh():
    """Creates a small, non-symmetric mesh for numerical testing."""
    mesh = Mesh()
    # 4 vertices forming a non-regular tetrahedron
    pts = np.array(
        [[0.1, 0.2, 0.05], [1.1, -0.1, 0.3], [0.4, 1.2, -0.2], [0.5, 0.4, 1.5]]
    )
    for i, p in enumerate(pts):
        mesh.vertices[i] = Vertex(i, p)

    # Triangles
    indices = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]
    edge_map = {}
    next_eid = 1
    for i, tri in enumerate(indices):
        e_ids = []
        for pair in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            key = tuple(sorted(pair))
            if key not in edge_map:
                edge_map[key] = next_eid
                mesh.edges[next_eid] = Edge(next_eid, pair[0], pair[1])
                next_eid += 1
            eid = edge_map[key]
            e_ids.append(eid if mesh.edges[eid].tail_index == pair[0] else -eid)
        mesh.facets[i] = Facet(i, e_ids)

    mesh.bodies[0] = Body(0, list(mesh.facets.keys()), target_volume=0.5)
    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()
    return mesh


def check_gradient_consistency(module, mesh, eps=1e-6, tol=1e-5):
    """
    Perform a central difference check:
    dE/dx approx (E(x+h) - E(x-h)) / 2h
    """
    gp = GlobalParameters()
    gp.surface_tension = 1.0
    gp.volume_stiffness = 10.0
    gp.bending_modulus = 1.0
    resolver = ParameterResolver(gp)

    # 1. Analytical Gradient
    positions = mesh.positions_view().copy()
    idx_map = mesh.vertex_index_to_row
    grad_analytical = np.zeros_like(positions)

    module.compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=idx_map,
        grad_arr=grad_analytical,
    )

    # 2. Numerical Gradient
    grad_numerical = np.zeros_like(positions)

    for i in range(len(positions)):  # for each vertex
        for d in range(3):  # for each dimension x, y, z
            # Shift forward
            positions[i, d] += eps
            # Force entity update (simulating mesh state change)
            for vid, row in idx_map.items():
                if row == i:
                    mesh.vertices[vid].position = positions[i]

            E_plus = module.compute_energy_and_gradient_array(
                mesh,
                gp,
                resolver,
                positions=positions,
                index_map=idx_map,
                grad_arr=np.zeros_like(positions),
            )

            # Shift backward
            positions[i, d] -= 2 * eps
            for vid, row in idx_map.items():
                if row == i:
                    mesh.vertices[vid].position = positions[i]

            E_minus = module.compute_energy_and_gradient_array(
                mesh,
                gp,
                resolver,
                positions=positions,
                index_map=idx_map,
                grad_arr=np.zeros_like(positions),
            )

            # Reset
            positions[i, d] += eps
            for vid, row in idx_map.items():
                if row == i:
                    mesh.vertices[vid].position = positions[i]

            grad_numerical[i, d] = (E_plus - E_minus) / (2 * eps)

    # Compare
    # Note: Analytical gradient in our modules is dE/dx
    diff = np.abs(grad_analytical - grad_numerical)
    max_diff = np.max(diff)

    # Check relative error for non-zero values
    denom = np.maximum(np.abs(grad_analytical), 1.0)
    rel_diff = diff / denom
    max_rel_diff = np.max(rel_diff)

    print(
        f"Module {module.__name__}: Max Abs Diff: {max_diff:.2e}, Max Rel Diff: {max_rel_diff:.2e}"
    )
    assert max_rel_diff < tol


def test_surface_gradient_consistency():
    mesh = create_random_mesh()
    check_gradient_consistency(surface, mesh)


def test_volume_gradient_consistency():
    mesh = create_random_mesh()
    # Ensure we are in penalty mode for volume energy
    mesh.global_parameters.set("volume_constraint_mode", "penalty")
    check_gradient_consistency(volume, mesh)


def test_invariance_under_translation():
    """Energy must be identical if we move the whole mesh."""
    mesh = create_random_mesh()
    gp = GlobalParameters()

    E1 = surface.calculate_surface_energy(mesh, gp)

    # Move
    for v in mesh.vertices.values():
        v.position += np.array([10.0, -5.0, 3.0])

    E2 = surface.calculate_surface_energy(mesh, gp)
    assert E1 == pytest.approx(E2)


def test_invariance_under_rotation():
    """Energy must be identical if we rotate the mesh."""
    mesh = create_random_mesh()
    gp = GlobalParameters()

    E1 = surface.calculate_surface_energy(mesh, gp)

    # Rotate 90 deg around Z
    c, s = 0, 1
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    for v in mesh.vertices.values():
        v.position = R @ v.position

    E2 = surface.calculate_surface_energy(mesh, gp)
    assert E1 == pytest.approx(E2)


def test_bending_flow_reduces_energy():
    """Verify that taking a small step along the bending force reduces bending energy."""
    mesh = create_random_mesh()
    gp = GlobalParameters()
    gp.bending_modulus = 1.0
    resolver = ParameterResolver(gp)

    positions = mesh.positions_view().copy()
    idx_map = mesh.vertex_index_to_row
    grad = np.zeros_like(positions)

    E_initial = bending.compute_energy_and_gradient_array(
        mesh, gp, resolver, positions=positions, index_map=idx_map, grad_arr=grad
    )

    # Step along -gradient (force direction)
    step_size = 1e-4
    new_positions = positions - step_size * grad

    # Update mesh
    for vid, row in idx_map.items():
        mesh.vertices[vid].position = new_positions[row]

    E_final = bending.compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=new_positions,
        index_map=idx_map,
        grad_arr=np.zeros_like(grad),
    )

    print(f"Bending Energy: {E_initial:.6f} -> {E_final:.6f}")
    assert E_final < E_initial


def test_line_tension_gradient_consistency():
    from modules.energy import line_tension

    mesh = create_random_mesh()
    # Add line tension energy to edges
    for edge in mesh.edges.values():
        edge.options["energy"] = ["line_tension"]
    check_gradient_consistency(line_tension, mesh)
