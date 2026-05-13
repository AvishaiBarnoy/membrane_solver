"""Physics symmetry tests for inner/outer leaflet tilt modules."""

import numpy as np
import pytest

from core.parameters.resolver import ParameterResolver
from geometry.entities import GlobalParameters
from modules.energy import tilt_in, tilt_leaflet, tilt_out


def test_tilt_leaflet_parity():
    """Verify that unified tilt_leaflet matches tilt_in and tilt_out results."""
    mesh = build_curved_mesh()
    mesh.build_position_cache()
    gp = GlobalParameters(
        {
            "tilt_modulus_in": 1.5,
            "tilt_modulus_out": 1.5,
        }
    )
    resolver = ParameterResolver(gp)

    tilts = np.random.rand(len(mesh.vertex_ids), 3)
    mesh.set_tilts_in_from_array(tilts)
    mesh.set_tilts_out_from_array(tilts)

    # 1. Test 'in' leaflet
    e_in_old, sg_in_old, tg_in_old = tilt_in.compute_energy_and_gradient(
        mesh, gp, resolver
    )
    e_in_new, sg_in_new, tg_in_new = tilt_leaflet.compute_energy_and_gradient_leaflet(
        mesh, gp, resolver, leaflet="in"
    )

    assert e_in_old == pytest.approx(e_in_new)
    for vid in mesh.vertices:
        assert sg_in_old[vid] == pytest.approx(sg_in_new[vid])
        assert tg_in_old[vid] == pytest.approx(tg_in_new[vid])

    # 2. Test 'out' leaflet
    e_out_old, sg_out_old, tg_out_old = tilt_out.compute_energy_and_gradient(
        mesh, gp, resolver
    )
    e_out_new, sg_out_new, tg_out_new = (
        tilt_leaflet.compute_energy_and_gradient_leaflet(
            mesh, gp, resolver, leaflet="out"
        )
    )

    assert e_out_old == pytest.approx(e_out_new)
    for vid in mesh.vertices:
        assert sg_out_old[vid] == pytest.approx(sg_out_new[vid])
        assert tg_out_old[vid] == pytest.approx(tg_out_new[vid])


from modules.energy import (
    bending_tilt_in,
    bending_tilt_out,
    tilt_smoothness_in,
    tilt_smoothness_out,
)


def test_tilt_smoothness_symmetry():
    """Verify that tilt_smoothness_in and out are identical for identical fields."""
    mesh = build_curved_mesh()
    mesh.build_position_cache()
    gp = GlobalParameters(
        {
            "bending_modulus_in": 1.5,
            "bending_modulus_out": 1.5,
        }
    )
    resolver = ParameterResolver(gp)

    # Smooth fields have more reliable gradients
    pos = mesh.positions_view()
    tilts = pos * 0.1
    mesh.set_tilts_in_from_array(tilts)
    mesh.set_tilts_out_from_array(tilts)

    e_in, sg_in, tg_in = tilt_smoothness_in.compute_energy_and_gradient(
        mesh, gp, resolver
    )
    e_out, sg_out, tg_out = tilt_smoothness_out.compute_energy_and_gradient(
        mesh, gp, resolver
    )

    assert e_in == pytest.approx(e_out)
    for vid in mesh.vertices:
        assert sg_in.get(vid, np.zeros(3)) == pytest.approx(
            sg_out.get(vid, np.zeros(3))
        )
        assert tg_in.get(vid, np.zeros(3)) == pytest.approx(
            tg_out.get(vid, np.zeros(3))
        )


from geometry.geom_io import load_data, parse_geometry


def build_curved_mesh():
    """Load a hemisphere mesh."""
    return parse_geometry(load_data("meshes/hemisphere_start.yaml"))


def test_tilt_magnitude_symmetry():
    """Verify that tilt_in and tilt_out magnitude energies are identical for identical fields."""
    mesh = build_curved_mesh()
    mesh.build_position_cache()
    gp = GlobalParameters(
        {
            "tilt_modulus_in": 1.5,
            "tilt_modulus_out": 1.5,
        }
    )
    resolver = ParameterResolver(gp)

    # Set up identical tilt fields in 3D
    tilts = np.random.rand(len(mesh.vertex_ids), 3)
    mesh.set_tilts_in_from_array(tilts)
    mesh.set_tilts_out_from_array(tilts)

    e_in, sg_in, tg_in = tilt_in.compute_energy_and_gradient(mesh, gp, resolver)
    e_out, sg_out, tg_out = tilt_out.compute_energy_and_gradient(mesh, gp, resolver)

    assert e_in == pytest.approx(e_out)

    # Gradients should also be identical for the same input field
    for vid in mesh.vertices:
        assert sg_in[vid] == pytest.approx(sg_out[vid])
        assert tg_in[vid] == pytest.approx(tg_out[vid])


def test_bending_tilt_coupling_sign_convention():
    """Verify the sign relationship between inner/outer coupled bending-tilt."""
    mesh = build_curved_mesh()
    mesh.build_position_cache()
    gp = GlobalParameters(
        {
            "bending_modulus_in": 2.0,
            "bending_modulus_out": 2.0,
            "tilt_solve_mode": "coupled",
        }
    )
    resolver = ParameterResolver(gp)

    pos = mesh.positions_view()
    # Tilt pointing outward from (0.5, 0.5)
    tilts = pos - np.array([0.0, 0.0, 0.0])  # Radial from origin
    normals = mesh.vertex_normals()
    dot = np.einsum("ij,ij->i", tilts, normals)
    tilts = tilts - dot[:, None] * normals

    mesh.set_tilts_in_from_array(tilts)
    mesh.set_tilts_out_from_array(tilts)

    e_out_base, _, tg_out = bending_tilt_out.compute_energy_and_gradient(
        mesh, gp, resolver
    )
    e_in_base, _, tg_in = bending_tilt_in.compute_energy_and_gradient(
        mesh, gp, resolver
    )

    # If we flip the tilt field (t -> -t), the divergence flips sign.

    # So (2H + div)^2 for 'out' should match (2H - (-div))^2 for 'out'? No.
    # (2H + div)^2 with t -> (2H - div)^2.
    # This should match the 'in' energy with the original t.

    mesh.set_tilts_out_from_array(-tilts)
    e_out_flipped_t, _, _ = bending_tilt_out.compute_energy_and_gradient(
        mesh, gp, resolver
    )

    assert e_out_flipped_t == pytest.approx(e_in_base)


if __name__ == "__main__":
    pytest.main([__file__])
