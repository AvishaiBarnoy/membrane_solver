import numpy as np

from core.parameters.global_parameters import GlobalParameters
from core.parameters.resolver import ParameterResolver
from geometry.entities import Mesh, Vertex
from geometry.geom_io import load_data, parse_geometry
from modules.energy import tilt_thetaB_contact_in


def _disk_rows(mesh, group: str) -> np.ndarray:
    rows = []
    for vid in mesh.vertex_ids:
        opts = mesh.vertices[int(vid)].options or {}
        if (
            opts.get("rim_slope_match_group") == group
            or opts.get("tilt_thetaB_group") == group
        ):
            rows.append(mesh.vertex_index_to_row[int(vid)])
    return np.asarray(rows, dtype=int)


def _order_by_angle(
    pts: np.ndarray, *, center: np.ndarray, normal: np.ndarray
) -> np.ndarray:
    # Simple XY-plane ordering (sufficient for the flat coarse meshes we test on).
    rel = pts - center[None, :]
    rel = rel - np.einsum("ij,j->i", rel, normal)[:, None] * normal[None, :]
    angles = np.arctan2(rel[:, 1], rel[:, 0])
    return np.argsort(angles)


def _arc_length_weights(positions: np.ndarray) -> np.ndarray:
    n = positions.shape[0]
    diffs_next = positions[(np.arange(n) + 1) % n] - positions
    diffs_prev = positions - positions[(np.arange(n) - 1) % n]
    l_next = np.linalg.norm(diffs_next, axis=1)
    l_prev = np.linalg.norm(diffs_prev, axis=1)
    return 0.5 * (l_next + l_prev)


def _expected_contact_energy(mesh, *, group: str, gamma: float, thetaB: float) -> float:
    positions = mesh.positions_view()
    rows = _disk_rows(mesh, group)
    assert rows.size > 0

    center = np.array([0.0, 0.0, 0.0], dtype=float)
    normal = np.array([0.0, 0.0, 1.0], dtype=float)

    pts = positions[rows]
    order = _order_by_angle(pts, center=center, normal=normal)
    pts = pts[order]

    weights = _arc_length_weights(pts)
    wsum = float(np.sum(weights))
    assert wsum > 1e-12

    r_vec = pts - center[None, :]
    r_vec = r_vec - np.einsum("ij,j->i", r_vec, normal)[:, None] * normal[None, :]
    r_len = np.linalg.norm(r_vec, axis=1)
    R_eff = float(np.sum(weights * r_len) / wsum)

    return float(-2.0 * np.pi * R_eff * gamma * thetaB)


def _make_circle_ring_mesh(*, n: int, radius: float, group: str) -> Mesh:
    """Create a minimal mesh with only a tagged circular boundary ring."""
    mesh = Mesh()
    mesh.global_parameters = GlobalParameters()
    for i in range(n):
        ang = 2.0 * np.pi * float(i) / float(n)
        x = radius * float(np.cos(ang))
        y = radius * float(np.sin(ang))
        mesh.vertices[i] = Vertex(
            index=i,
            position=np.array([x, y, 0.0], dtype=float),
            options={"rim_slope_match_group": group},
        )
    mesh.increment_version()
    return mesh


def test_tilt_thetaB_contact_in_off_mode_ignores_nonzero_tilts() -> None:
    mesh = parse_geometry(
        load_data("meshes/caveolin/kozlov_free_disk_coarse_refinable.yaml")
    )
    gp = mesh.global_parameters
    resolver = ParameterResolver(gp)

    thetaB = 0.04
    gamma = float(gp.get("tilt_thetaB_contact_strength_in") or 0.0)
    assert gamma != 0.0

    gp.set("tilt_thetaB_value", thetaB)
    gp.set("tilt_thetaB_contact_penalty_mode", "off")

    positions = mesh.positions_view()
    # Construct a non-zero tilt field; off mode should ignore it.
    tilts_in = np.zeros_like(positions)
    tilts_in[:, 0] = 1e-3

    idx_map = mesh.vertex_index_to_row
    grad_arr = np.zeros_like(positions)
    tilt_grad = np.zeros_like(positions)

    E = tilt_thetaB_contact_in.compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=idx_map,
        grad_arr=grad_arr,
        tilts_in=tilts_in,
        tilt_in_grad_arr=tilt_grad,
    )

    expected = _expected_contact_energy(mesh, group="disk", gamma=gamma, thetaB=thetaB)
    assert abs(float(E - expected)) < 1e-10
    assert float(np.max(np.abs(tilt_grad))) == 0.0


def test_tilt_thetaB_contact_in_off_mode_invariant_to_vertex_count_for_circle() -> None:
    group = "disk"
    radius = 0.4666666666666667
    thetaB = 0.03
    gamma = 4.286

    def eval_mesh(mesh: Mesh) -> float:
        gp = mesh.global_parameters
        gp.set("tilt_thetaB_group_in", group)
        gp.set("tilt_thetaB_center", [0.0, 0.0, 0.0])
        gp.set("tilt_thetaB_normal", [0.0, 0.0, 1.0])
        gp.set("tilt_thetaB_value", thetaB)
        gp.set("tilt_thetaB_contact_strength_in", gamma)
        gp.set("tilt_thetaB_contact_penalty_mode", "off")
        resolver = ParameterResolver(gp)

        positions = mesh.positions_view()
        idx_map = mesh.vertex_index_to_row
        grad_arr = np.zeros_like(positions)
        return float(
            tilt_thetaB_contact_in.compute_energy_and_gradient_array(
                mesh,
                gp,
                resolver,
                positions=positions,
                index_map=idx_map,
                grad_arr=grad_arr,
                tilts_in=None,
            )
        )

    m8 = _make_circle_ring_mesh(n=8, radius=radius, group=group)
    m32 = _make_circle_ring_mesh(n=32, radius=radius, group=group)

    e8 = eval_mesh(m8)
    e32 = eval_mesh(m32)
    expected = float(-2.0 * np.pi * radius * gamma * thetaB)
    assert abs(e8 - expected) < 1e-12
    assert abs(e32 - expected) < 1e-12
    assert abs(e8 - e32) < 1e-12


def test_tilt_thetaB_contact_in_reports_pure_contact_work_by_default() -> None:
    mesh = parse_geometry(
        load_data("meshes/caveolin/kozlov_free_disk_coarse_refinable.yaml")
    )
    gp = mesh.global_parameters
    resolver = ParameterResolver(gp)

    thetaB = 0.07
    gamma = float(gp.get("tilt_thetaB_contact_strength_in") or 0.0)
    assert gamma != 0.0

    gp.set("tilt_thetaB_value", thetaB)
    gp.set("tilt_thetaB_contact_penalty_mode", "off")
    # Strength should not matter for pure contact work.
    gp.set("tilt_thetaB_strength_in", 1.0e8)

    positions = mesh.positions_view()
    idx_map = mesh.vertex_index_to_row
    grad_arr = np.zeros_like(positions)
    tilt_grad = np.zeros_like(positions)

    E = tilt_thetaB_contact_in.compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=idx_map,
        grad_arr=grad_arr,
        tilts_in=None,
        tilt_in_grad_arr=tilt_grad,
    )

    expected = _expected_contact_energy(mesh, group="disk", gamma=gamma, thetaB=thetaB)
    assert np.isfinite(E)
    assert abs(float(E - expected)) < 1e-10
    # No penalty => no tilt gradient from this module.
    assert float(np.max(np.abs(tilt_grad))) == 0.0


def test_tilt_thetaB_contact_in_contact_energy_independent_of_penalty_strength() -> (
    None
):
    mesh = parse_geometry(
        load_data("meshes/caveolin/kozlov_free_disk_coarse_refinable.yaml")
    )
    gp = mesh.global_parameters
    resolver = ParameterResolver(gp)

    gp.set("tilt_thetaB_contact_penalty_mode", "off")
    gp.set("tilt_thetaB_value", 0.05)
    gamma = float(gp.get("tilt_thetaB_contact_strength_in") or 0.0)

    positions = mesh.positions_view()
    idx_map = mesh.vertex_index_to_row
    grad_arr = np.zeros_like(positions)

    gp.set("tilt_thetaB_strength_in", 1.0e3)
    E1 = tilt_thetaB_contact_in.compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=idx_map,
        grad_arr=grad_arr,
        tilts_in=None,
    )
    gp.set("tilt_thetaB_strength_in", 1.0e12)
    E2 = tilt_thetaB_contact_in.compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=idx_map,
        grad_arr=grad_arr,
        tilts_in=None,
    )

    expected = _expected_contact_energy(mesh, group="disk", gamma=gamma, thetaB=0.05)
    assert abs(float(E1 - expected)) < 1e-10
    assert abs(float(E2 - expected)) < 1e-10


def test_tilt_thetaB_contact_in_legacy_mode_includes_penalty_and_gradient() -> None:
    mesh = parse_geometry(
        load_data("meshes/caveolin/kozlov_free_disk_coarse_refinable.yaml")
    )
    gp = mesh.global_parameters
    resolver = ParameterResolver(gp)

    gp.set("tilt_thetaB_contact_penalty_mode", "legacy")
    gp.set("tilt_thetaB_value", 0.02)
    gp.set("tilt_thetaB_strength_in", 10.0)
    gamma = float(gp.get("tilt_thetaB_contact_strength_in") or 0.0)

    positions = mesh.positions_view()
    idx_map = mesh.vertex_index_to_row
    grad_arr = np.zeros_like(positions)
    tilt_grad = np.zeros_like(positions)

    # Ensure theta_i != thetaB by leaving tilts at 0.
    E = tilt_thetaB_contact_in.compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=idx_map,
        grad_arr=grad_arr,
        tilts_in=np.zeros_like(positions),
        tilt_in_grad_arr=tilt_grad,
    )
    expected_contact = _expected_contact_energy(
        mesh, group="disk", gamma=gamma, thetaB=0.02
    )
    # Total should differ from pure contact due to penalty > 0.
    assert float(E) > float(expected_contact)
    assert float(np.max(np.abs(tilt_grad))) > 0.0
