from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.special import i1, k1

from geometry.geom_io import parse_geometry
from tools.free_one_disc_convergence import (
    FreeOneDiscCase,
    build_canonical_free_one_disc_fixture,
    default_convergence_cases,
    fixed_theta_field_agreement,
    resample_axisymmetric_rings,
    shape_regular_free_radii,
)

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml"


def _base_doc() -> dict:
    return yaml.safe_load(BASE.read_text(encoding="utf-8")) or {}


def test_shape_regular_free_radii_avoids_terminal_sliver() -> None:
    trace = 0.48
    coarse = 1.0
    radii = shape_regular_free_radii(
        trace_radius=trace,
        first_coarse_radius=coarse,
        target_spacing=0.03,
    )
    points = np.asarray([trace, *radii, coarse], dtype=float)
    spacing = np.diff(points)

    assert np.all(spacing > 0.0)
    assert float(np.max(spacing)) <= 0.03 + 1.0e-12
    assert float(np.max(spacing) / np.min(spacing)) < 1.01


def test_canonical_free_fixture_has_no_constructed_support_shells() -> None:
    case = FreeOneDiscCase(
        label="canonical_free_unit",
        trace_epsilon=0.02,
        near_spacing=0.02,
        outer_radius=8.0,
    )
    doc = build_canonical_free_one_disc_fixture(
        base_doc=_base_doc(), case=case, theta_b=0.18
    )

    assert doc["global_parameters"]["free_one_disc_validation_lane"] is True
    assert "pin_to_plane" not in doc["definitions"]["outer_rim"]["constraints"]
    assert doc["definitions"]["outer_rim"]["pin_to_circle_mode"] == "slide"
    trace_count = 0
    free_count = 0
    for vertex in doc["vertices"]:
        opts = vertex[3] if len(vertex) > 3 and isinstance(vertex[3], dict) else {}
        assert "outer_shell_scaffold_index" not in opts
        if opts.get("pin_to_circle_group") == "trace_layer":
            trace_count += 1
            assert opts["pin_to_circle_mode"] == "slide"
            assert opts.get("constraints") == ["pin_to_circle"]
        if opts.get("outer_shell_free_index") is not None:
            free_count += 1
            assert "pin_to_circle" not in opts.get("constraints", [])
            assert "pin_to_plane" not in opts.get("constraints", [])
        if opts.get("pin_to_circle_group") == "outer":
            assert "pin_to_plane" not in opts.get("constraints", [])
            assert opts["pin_to_circle_mode"] == "slide"

    assert trace_count > 0
    assert free_count > 0


def test_fixed_theta_field_metrics_recover_exact_sampled_profiles() -> None:
    theta = 0.18
    case = FreeOneDiscCase(
        label="exact_profile_unit",
        trace_epsilon=0.02,
        near_spacing=0.02,
        outer_radius=8.0,
    )
    mesh = parse_geometry(
        build_canonical_free_one_disc_fixture(
            base_doc=_base_doc(), case=case, theta_b=theta
        )
    )
    positions = mesh.positions_view().copy()
    radii = np.linalg.norm(positions[:, :2], axis=1)
    radial = np.zeros_like(positions)
    active = radii > 1.0e-12
    radial[active, :2] = positions[active, :2] / radii[active, None]
    radius = float(mesh.global_parameters.get("theory_radius"))
    lam = np.sqrt(
        float(mesh.global_parameters.get("tilt_modulus_out"))
        / float(mesh.global_parameters.get("bending_modulus_out"))
    )
    phi = 0.5 * theta
    outer = radii > radius
    positions[outer, 2] = phi * radius * np.log(radii[outer] / radius)
    for row, vertex_id in enumerate(mesh.vertex_ids):
        mesh.vertices[int(vertex_id)].position[:] = positions[row]
    mesh.increment_version()
    normals = mesh.vertex_normals(positions=mesh.positions_view())
    tangent_radial = radial - np.einsum("ij,ij->i", radial, normals)[:, None] * normals
    tangent_norm = np.linalg.norm(tangent_radial, axis=1)
    tangent_active = tangent_norm > 1.0e-12
    tangent_radial[tangent_active] /= tangent_norm[tangent_active, None]
    tin = np.zeros_like(positions)
    tout = np.zeros_like(positions)
    inner_active = active & ~outer
    tin[inner_active] = (theta * i1(lam * radii[inner_active]) / i1(lam * radius))[
        :, None
    ] * tangent_radial[inner_active]
    expected_outer = phi * k1(lam * radii[outer]) / k1(lam * radius)
    tin[outer] = expected_outer[:, None] * tangent_radial[outer]
    tout[outer] = expected_outer[:, None] * tangent_radial[outer]

    mesh.set_tilts_in_from_array(tin)
    mesh.set_tilts_out_from_array(tout)
    metrics = fixed_theta_field_agreement(mesh, theta_b=theta)

    assert metrics["z"]["relative_l2"] < 1.0e-10
    assert metrics["t_in_outer"]["relative_l2"] < 1.0e-10
    assert metrics["t_out_outer"]["relative_l2"] < 1.0e-10
    assert metrics["t_in_disc"]["relative_l2"] < 1.0e-10
    assert metrics["phi"]["relative_l2"] < 0.03
    assert metrics["vector"]["max_tangential_leak_relative"] < 1.0e-12


def test_axisymmetric_ring_resampling_preserves_topology_and_tags() -> None:
    case = FreeOneDiscCase(
        label="angular_resample_unit",
        trace_epsilon=0.02,
        near_spacing=0.02,
        outer_radius=8.0,
    )
    doc = build_canonical_free_one_disc_fixture(
        base_doc=_base_doc(), case=case, theta_b=0.18
    )
    original_ring_count = len(
        {
            round(float(np.hypot(vertex[0], vertex[1])), 10)
            for vertex in doc["vertices"]
            if np.hypot(vertex[0], vertex[1]) > 1.0e-12
        }
    )

    resampled = resample_axisymmetric_rings(doc, angular_sectors=24)
    mesh = parse_geometry(resampled)
    radii = np.linalg.norm(mesh.positions_view()[:, :2], axis=1)
    counts = [
        int(np.count_nonzero(np.isclose(radii, radius, atol=1.0e-9)))
        for radius in sorted({round(float(value), 10) for value in radii if value > 0})
    ]

    assert len(counts) == original_ring_count
    assert set(counts) == {24}
    assert len(mesh.vertices) == 1 + 24 * original_ring_count
    assert len(mesh.facets) == 24 + 2 * 24 * (original_ring_count - 2)
    assert (
        sum(
            vertex.options.get("rim_slope_match_group") == "disk"
            for vertex in mesh.vertices.values()
        )
        == 24
    )
    assert (
        sum(
            vertex.options.get("pin_to_circle_group") == "trace_layer"
            for vertex in mesh.vertices.values()
        )
        == 24
    )


def test_default_convergence_matrix_separates_three_families() -> None:
    cases = default_convergence_cases()
    labels = {case.label for case in cases}

    assert {"radial_h020", "radial_h010", "radial_h005"} <= labels
    assert {"angular_n12", "angular_n24", "angular_n48"} <= labels
    assert {"domain_r8", "domain_r12", "domain_r16"} <= labels
    assert (
        len({case.outer_radius for case in cases if case.label.startswith("domain_")})
        == 3
    )
    assert (
        max(case.angular_sectors for case in cases if case.label.startswith("angular_"))
        == 48
    )


def test_canonical_free_fixture_rejects_outer_radius_inside_disc() -> None:
    case = FreeOneDiscCase(
        label="invalid_domain",
        trace_epsilon=0.01,
        near_spacing=0.01,
        outer_radius=0.4,
    )
    with pytest.raises(ValueError, match="outer_radius"):
        build_canonical_free_one_disc_fixture(
            base_doc=_base_doc(), case=case, theta_b=0.18
        )
