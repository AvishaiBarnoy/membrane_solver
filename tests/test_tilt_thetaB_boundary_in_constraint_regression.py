import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry
from modules.constraints import tilt_thetaB_boundary_in
from runtime.constraint_manager import ConstraintModuleManager
from runtime.refinement import refine_triangle_mesh


def _fixture_path(name: str) -> str:
    import os

    here = os.path.dirname(__file__)
    return os.path.join(here, "fixtures", name)


def _tangent_radial_directions(mesh, rows: np.ndarray) -> np.ndarray:
    """Return per-row radial directions projected to the local tangent plane."""
    positions = mesh.positions_view()
    center = np.asarray(
        mesh.global_parameters.get("tilt_thetaB_center") or [0, 0, 0], dtype=float
    )
    normal = np.asarray(
        mesh.global_parameters.get("tilt_thetaB_normal") or [0, 0, 1], dtype=float
    )
    nrm = float(np.linalg.norm(normal))
    if nrm > 1e-15:
        normal = normal / nrm
    else:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)

    pts = positions[rows]
    r_vec = pts - center[None, :]
    r_vec = r_vec - np.einsum("ij,j->i", r_vec, normal)[:, None] * normal[None, :]
    r_len = np.linalg.norm(r_vec, axis=1)
    r_hat = np.zeros_like(r_vec)
    good = r_len > 1e-12
    r_hat[good] = r_vec[good] / r_len[good][:, None]

    normals_v = mesh.vertex_normals(positions=positions)[rows]
    r_dir = r_hat - np.einsum("ij,ij->i", r_hat, normals_v)[:, None] * normals_v
    nrm = np.linalg.norm(r_dir, axis=1)
    ok = nrm > 1e-12
    r_dir[ok] = r_dir[ok] / nrm[ok][:, None]
    return r_dir


def test_tilt_thetaB_boundary_in_enforces_radial_component_on_group_ring() -> None:
    # Use a stable, hand-authored coarse disk mesh intended for refinement.
    mesh = parse_geometry(
        load_data(_fixture_path("kozlov_free_disk_coarse_refinable.yaml"))
    )
    gp = mesh.global_parameters

    gp.set("tilt_thetaB_value", 0.07)
    gp.set("tilt_thetaB_center", [0.0, 0.0, 0.0])
    gp.set("tilt_thetaB_normal", [0.0, 0.0, 1.0])
    gp.set("tilt_thetaB_group_in", "disk")

    # Identify the disk boundary ring via the same option used by other disk interface modules.
    rows = []
    for vid in mesh.vertex_ids:
        opts = mesh.vertices[int(vid)].options or {}
        if opts.get("rim_slope_match_group") == "disk":
            rows.append(mesh.vertex_index_to_row[int(vid)])
    rows = np.asarray(rows, dtype=int)
    assert rows.size > 0

    # Start from a tilt field with the wrong radial component (t_in · r_hat = 0).
    positions = mesh.positions_view()
    normals = mesh.vertex_normals(positions=positions)
    tilts_in = mesh.tilts_in_view().copy(order="F")
    tilts_in[:] = 0.0
    # Add a small tangential (non-radial) component so the projection is non-trivial.
    tilts_in[:, 0] = 1e-3
    # Make the tilts tangent to the surface.
    tilts_in = tilts_in - np.einsum("ij,ij->i", tilts_in, normals)[:, None] * normals
    mesh.set_tilts_in_from_array(tilts_in)

    cm = ConstraintModuleManager(["tilt_thetaB_boundary_in"])
    cm.enforce_tilt_constraints(mesh, global_params=gp)

    # Check the enforced radial component on the tagged ring.
    tilts_in = mesh.tilts_in_view()
    r_dir = _tangent_radial_directions(mesh, rows)
    theta_vals = np.einsum("ij,ij->i", tilts_in[rows], r_dir)
    # Use a loose tol; this is an exact projection, but r_dir is discrete/tangent-projected.
    assert float(np.median(np.abs(theta_vals - 0.07))) < 1e-10


def test_tilt_thetaB_boundary_in_respects_tilt_fixed_in() -> None:
    mesh = parse_geometry(
        load_data(_fixture_path("kozlov_free_disk_coarse_refinable.yaml"))
    )
    gp = mesh.global_parameters

    gp.set("tilt_thetaB_value", 0.03)
    gp.set("tilt_thetaB_center", [0.0, 0.0, 0.0])
    gp.set("tilt_thetaB_normal", [0.0, 0.0, 1.0])
    # Exercise the fallback path (group resolved from rim_slope_match_disk_group).
    gp.set("tilt_thetaB_group_in", None)
    gp.set("rim_slope_match_disk_group", "disk")

    rows = []
    for vid in mesh.vertex_ids:
        opts = mesh.vertices[int(vid)].options or {}
        if opts.get("rim_slope_match_group") == "disk":
            rows.append(mesh.vertex_index_to_row[int(vid)])
    rows = np.asarray(rows, dtype=int)
    assert rows.size > 0

    # Fix one boundary vertex's inner tilt and ensure the constraint leaves it unchanged.
    fixed_row = int(rows[0])
    fixed_vid = int(mesh.vertex_ids[fixed_row])
    mesh.vertices[fixed_vid].tilt_fixed_in = True

    positions = mesh.positions_view()
    normals = mesh.vertex_normals(positions=positions)
    tilts_in = mesh.tilts_in_view().copy(order="F")
    tilts_in[:] = 0.0
    tilts_in[:, 1] = 2e-3
    tilts_in = tilts_in - np.einsum("ij,ij->i", tilts_in, normals)[:, None] * normals
    mesh.set_tilts_in_from_array(tilts_in)
    before_fixed = mesh.tilts_in_view()[fixed_row].copy()

    cm = ConstraintModuleManager(["tilt_thetaB_boundary_in"])
    cm.enforce_tilt_constraints(mesh, global_params=gp)

    after_fixed = mesh.tilts_in_view()[fixed_row].copy()
    assert np.allclose(after_fixed, before_fixed, atol=0.0, rtol=0.0)


def test_tilt_thetaB_boundary_in_noops_when_group_missing_or_empty() -> None:
    mesh = parse_geometry(
        load_data(_fixture_path("kozlov_free_disk_coarse_refinable.yaml"))
    )
    gp = mesh.global_parameters

    # Make the tilts non-zero so a buggy constraint that touches everything would be detected.
    positions = mesh.positions_view()
    normals = mesh.vertex_normals(positions=positions)
    tilts_in = mesh.tilts_in_view().copy(order="F")
    tilts_in[:] = 0.0
    tilts_in[:, 0] = 1e-3
    tilts_in = tilts_in - np.einsum("ij,ij->i", tilts_in, normals)[:, None] * normals
    mesh.set_tilts_in_from_array(tilts_in)
    before = mesh.tilts_in_view().copy(order="F")

    cm = ConstraintModuleManager(["tilt_thetaB_boundary_in"])

    # Case 1: group resolves to None -> no-op
    gp.set("tilt_thetaB_group_in", None)
    gp.set("rim_slope_match_disk_group", None)
    cm.enforce_tilt_constraints(mesh, global_params=gp)
    assert np.allclose(mesh.tilts_in_view(), before, atol=0.0, rtol=0.0)

    # Case 2: group resolves but matches no vertices -> no-op
    gp.set("rim_slope_match_disk_group", "does_not_exist")
    cm.enforce_tilt_constraints(mesh, global_params=gp)
    assert np.allclose(mesh.tilts_in_view(), before, atol=0.0, rtol=0.0)


def test_tilt_thetaB_boundary_in_constraint_gradients_tilt_match_fd() -> None:
    mesh = parse_geometry(
        load_data(_fixture_path("kozlov_free_disk_coarse_refinable.yaml"))
    )
    gp = mesh.global_parameters
    gp.set("tilt_thetaB_value", 0.07)
    gp.set("tilt_thetaB_center", [0.0, 0.0, 0.0])
    gp.set("tilt_thetaB_normal", [0.0, 0.0, 1.0])
    gp.set("tilt_thetaB_group_in", "disk")

    mesh.build_position_cache()
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    tilts_in = mesh.tilts_in_view().copy(order="F")
    tilts_out = mesh.tilts_out_view().copy(order="F")

    g_list = tilt_thetaB_boundary_in.constraint_gradients_tilt_array(
        mesh,
        gp,
        positions=positions,
        index_map=index_map,
        tilts_in=tilts_in,
        tilts_out=tilts_out,
    )
    assert g_list is not None and len(g_list) > 0

    g_in, g_out = g_list[0]
    assert g_in is not None
    assert g_out is None
    rows = np.flatnonzero(np.linalg.norm(g_in, axis=1) > 0.0)
    assert rows.size == 1
    row = int(rows[0])
    dvec = g_in[row].copy()

    thetaB = float(gp.get("tilt_thetaB_value") or 0.0)
    t0 = tilts_in[row].copy()

    def residual(tvec: np.ndarray) -> float:
        return float(np.dot(tvec, dvec) - thetaB)

    eps = 1e-7
    fd = np.zeros(3, dtype=float)
    for axis in range(3):
        tp = t0.copy()
        tm = t0.copy()
        tp[axis] += eps
        tm[axis] -= eps
        fd[axis] = (residual(tp) - residual(tm)) / (2.0 * eps)

    assert np.allclose(dvec, fd, atol=1e-8, rtol=1e-6)


def test_tilt_thetaB_boundary_in_shape_gradient_hooks_return_none() -> None:
    mesh = parse_geometry(
        load_data(_fixture_path("kozlov_free_disk_coarse_refinable.yaml"))
    )
    gp = mesh.global_parameters
    mesh.build_position_cache()
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row

    assert tilt_thetaB_boundary_in.constraint_gradients(mesh, gp) is None
    assert (
        tilt_thetaB_boundary_in.constraint_gradients_array(
            mesh, gp, positions=positions, index_map=index_map
        )
        is None
    )


def test_tilt_thetaB_boundary_in_geometric_fallback_covers_unrefined_ring_tags() -> (
    None
):
    mesh = parse_geometry(
        load_data(_fixture_path("kozlov_1disk_3d_free_disk_theory_parity.yaml"))
    )
    mesh = refine_triangle_mesh(mesh)
    gp = mesh.global_parameters
    gp.set("tilt_thetaB_value", 0.05)
    gp.set("tilt_thetaB_center", [0.0, 0.0, 0.0])
    gp.set("tilt_thetaB_normal", [0.0, 0.0, 1.0])
    gp.set("tilt_thetaB_group_in", "disk")

    cm_circle = ConstraintModuleManager(["pin_to_circle"])
    cm_circle.enforce_all(mesh, global_params=gp, context="mesh_operation")

    positions = mesh.positions_view()
    r = np.linalg.norm(positions[:, :2], axis=1)

    tagged_rows = []
    for vid in mesh.vertex_ids:
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if (
            opts.get("rim_slope_match_group") == "disk"
            or opts.get("tilt_thetaB_group_in") == "disk"
            or opts.get("tilt_thetaB_group") == "disk"
        ):
            tagged_rows.append(mesh.vertex_index_to_row[int(vid)])
    tagged_rows = np.asarray(tagged_rows, dtype=int)
    assert tagged_rows.size > 0

    target_radius = float(np.median(r[tagged_rows]))
    rim_rows = np.flatnonzero(np.isclose(r, target_radius, atol=1e-6))
    assert rim_rows.size > 0

    stripped_rows = rim_rows[::2]
    for row in stripped_rows:
        vid = int(mesh.vertex_ids[int(row)])
        opts = getattr(mesh.vertices[vid], "options", None) or {}
        for key in (
            "rim_slope_match_group",
            "tilt_thetaB_group_in",
            "tilt_thetaB_group",
        ):
            opts.pop(key, None)
        mesh.vertices[vid].options = opts

    tagged_rows_after = []
    for vid in mesh.vertex_ids:
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if (
            opts.get("rim_slope_match_group") == "disk"
            or opts.get("tilt_thetaB_group_in") == "disk"
            or opts.get("tilt_thetaB_group") == "disk"
        ):
            tagged_rows_after.append(mesh.vertex_index_to_row[int(vid)])
    tagged_rows_after = np.asarray(tagged_rows_after, dtype=int)
    assert tagged_rows_after.size < rim_rows.size

    data = tilt_thetaB_boundary_in._boundary_directions(mesh, gp, positions=positions)
    assert data is not None
    selected_rows, _ = data
    assert np.intersect1d(selected_rows, rim_rows).size == rim_rows.size
    assert np.intersect1d(selected_rows, stripped_rows).size > 0

    normals = mesh.vertex_normals(positions=positions)
    tilts_in = mesh.tilts_in_view().copy(order="F")
    tilts_in[:] = 0.0
    tilts_in[:, 1] = 2e-3
    tilts_in = tilts_in - np.einsum("ij,ij->i", tilts_in, normals)[:, None] * normals
    mesh.set_tilts_in_from_array(tilts_in)

    cm = ConstraintModuleManager(["tilt_thetaB_boundary_in"])
    cm.enforce_tilt_constraints(mesh, global_params=gp)

    enforced = mesh.tilts_in_view()
    r_dir = _tangent_radial_directions(mesh, rim_rows)
    theta_vals = np.einsum("ij,ij->i", enforced[rim_rows], r_dir)
    assert float(np.max(np.abs(theta_vals - 0.05))) < 1e-10
