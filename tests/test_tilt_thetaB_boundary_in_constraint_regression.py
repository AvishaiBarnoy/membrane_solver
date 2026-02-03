import numpy as np

from geometry.geom_io import load_data, parse_geometry
from runtime.constraint_manager import ConstraintModuleManager


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
        load_data("meshes/caveolin/kozlov_free_disk_coarse_refinable.yaml")
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
        load_data("meshes/caveolin/kozlov_free_disk_coarse_refinable.yaml")
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
