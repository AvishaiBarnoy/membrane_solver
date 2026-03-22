import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.parameters.resolver import ParameterResolver
from geometry.geom_io import parse_geometry
from modules.energy import (
    tilt_rim_source_bilayer,
    tilt_rim_source_in,
    tilt_rim_source_out,
)


def _disk_plus_annulus_internal_rim_mesh(*, n: int = 10) -> dict:
    """Return a mesh with a tagged *internal* rim ring (not a boundary)."""
    if n < 6:
        raise ValueError("n must be >= 6")

    vertices: list[list] = []
    # Center vertex (disk).
    vertices.append([0.0, 0.0, 0.0])
    # Rim ring at r=1 (tagged).
    for i in range(n):
        theta = 2.0 * np.pi * i / n
        vertices.append(
            [
                float(np.cos(theta)),
                float(np.sin(theta)),
                0.0,
                {"pin_to_circle_group": "rim"},
            ]
        )
    # Outer ring at r=2.
    for i in range(n):
        theta = 2.0 * np.pi * i / n
        vertices.append([float(2.0 * np.cos(theta)), float(2.0 * np.sin(theta)), 0.0])

    def rim_vid(i: int) -> int:
        return 1 + int(i)

    def out_vid(i: int) -> int:
        return 1 + n + int(i)

    edges: list[list[int]] = []
    # Rim ring edges (internal rim).
    for i in range(n):
        edges.append([rim_vid(i), rim_vid((i + 1) % n)])

    # Outer ring edges (boundary).
    for i in range(n):
        edges.append([out_vid(i), out_vid((i + 1) % n)])

    # Spokes rim<->outer.
    for i in range(n):
        edges.append([rim_vid(i), out_vid(i)])

    # Diagonals rim(i) -> outer(i+1) for annulus triangulation.
    for i in range(n):
        edges.append([rim_vid(i), out_vid((i + 1) % n)])

    # Spokes center<->rim (disk triangulation).
    for i in range(n):
        edges.append([0, rim_vid(i)])

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
    # Disk fan triangles.
    for i in range(n):
        i1 = (i + 1) % n
        faces.append(
            [
                edge_ref(0, rim_vid(i)),
                edge_ref(rim_vid(i), rim_vid(i1)),
                edge_ref(rim_vid(i1), 0),
            ]
        )

    # Annulus quads triangulated.
    for i in range(n):
        i1 = (i + 1) % n
        faces.append(
            [
                edge_ref(rim_vid(i), rim_vid(i1)),
                edge_ref(rim_vid(i1), out_vid(i1)),
                edge_ref(out_vid(i1), rim_vid(i)),
            ]
        )
        faces.append(
            [
                edge_ref(rim_vid(i), out_vid(i1)),
                edge_ref(out_vid(i1), out_vid(i)),
                edge_ref(out_vid(i), rim_vid(i)),
            ]
        )

    return {
        "global_parameters": {
            "tilt_rim_source_center": [0.0, 0.0, 0.0],
            "tilt_rim_source_edge_mode": "all",
            "tilt_rim_source_group": "rim",
            "tilt_rim_source_strength": 1.0,
            "tilt_rim_source_group_in": "rim",
            "tilt_rim_source_strength_in": 1.0,
            "tilt_rim_source_group_out": "rim",
            "tilt_rim_source_strength_out": 1.0,
        },
        "energy_modules": [],
        "vertices": vertices,
        "edges": edges,
        "faces": faces,
        "instructions": [],
    }


def _set_radial_tilts(mesh) -> None:
    positions = mesh.positions_view()
    r = positions.copy()
    r[:, 2] = 0.0
    rn = np.linalg.norm(r, axis=1)
    radial = np.zeros_like(positions)
    good = rn > 1e-12
    radial[good] = r[good] / rn[good][:, None]
    mesh.set_tilts_in_from_array(radial)
    mesh.set_tilts_out_from_array(2.0 * radial)


def test_tilt_rim_source_in_selection_cache_reuses_and_invalidates() -> None:
    mesh = parse_geometry(_disk_plus_annulus_internal_rim_mesh())

    payload = tilt_rim_source_in._rim_selection_payload(mesh, group="rim", mode="all")
    assert payload is not None
    original_edge_count = int(payload["edge_ids"].size)
    assert original_edge_count > 0

    cache = getattr(mesh, "_tilt_rim_source_in_selection_cache")
    sentinel = object()
    cache["value"] = sentinel

    cached = tilt_rim_source_in._rim_selection_payload(mesh, group="rim", mode="all")
    assert cached is sentinel

    mesh.vertices[1].options["pin_to_circle_group"] = "other"
    mesh._vertex_ids_version += 1

    refreshed = tilt_rim_source_in._rim_selection_payload(mesh, group="rim", mode="all")
    assert refreshed is not sentinel
    assert int(refreshed["edge_ids"].size) < original_edge_count


def test_tilt_rim_source_in_gamma_cache_reuses_and_invalidates_on_strength_change() -> (
    None
):
    mesh = parse_geometry(_disk_plus_annulus_internal_rim_mesh())
    resolver = ParameterResolver(mesh.global_parameters)
    payload = tilt_rim_source_in._rim_selection_payload(mesh, group="rim", mode="all")
    assert payload is not None

    gamma_a = tilt_rim_source_in._resolved_gamma(
        mesh, resolver, edge_ids=payload["edge_ids"]
    )
    gamma_b = tilt_rim_source_in._resolved_gamma(
        mesh, resolver, edge_ids=payload["edge_ids"]
    )
    assert gamma_b is gamma_a

    mesh.global_parameters.set("tilt_rim_source_strength_in", 2.0)
    gamma_c = tilt_rim_source_in._resolved_gamma(
        mesh, resolver, edge_ids=payload["edge_ids"]
    )
    assert gamma_c is not gamma_a
    assert np.allclose(gamma_c, 2.0)


def test_tilt_rim_source_in_fixed_frame_cache_reuses_and_invalidates_on_center_change() -> (
    None
):
    mesh = parse_geometry(_disk_plus_annulus_internal_rim_mesh())
    resolver = ParameterResolver(mesh.global_parameters)
    payload = tilt_rim_source_in._rim_selection_payload(mesh, group="rim", mode="all")
    assert payload is not None
    assert not payload["follow"]

    center_a, normal_a = tilt_rim_source_in._fixed_circle_frame(
        mesh, resolver, normal_row=payload["normal_row"]
    )
    center_b, normal_b = tilt_rim_source_in._fixed_circle_frame(
        mesh, resolver, normal_row=payload["normal_row"]
    )
    assert center_b is center_a
    assert normal_b is normal_a

    mesh.global_parameters.set("tilt_rim_source_center", [1.0, 0.0, 0.0])
    center_c, normal_c = tilt_rim_source_in._fixed_circle_frame(
        mesh, resolver, normal_row=payload["normal_row"]
    )
    assert center_c is not center_a
    assert normal_c is not normal_a
    assert np.allclose(center_c, [1.0, 0.0, 0.0])


def test_tilt_rim_source_internal_rim_matches_in_plus_out() -> None:
    """Bilayer internal-rim mode should equal in+out energies and gradients."""
    mesh = parse_geometry(_disk_plus_annulus_internal_rim_mesh())
    _set_radial_tilts(mesh)
    resolver = ParameterResolver(mesh.global_parameters)

    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    grad_dummy = np.zeros_like(positions)

    grad_in_b = np.zeros_like(positions)
    grad_out_b = np.zeros_like(positions)
    e_b = tilt_rim_source_bilayer.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_dummy,
        tilts_in=mesh.tilts_in_view(),
        tilts_out=mesh.tilts_out_view(),
        tilt_in_grad_arr=grad_in_b,
        tilt_out_grad_arr=grad_out_b,
    )

    grad_in = np.zeros_like(positions)
    grad_out = np.zeros_like(positions)
    e_in = tilt_rim_source_in.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_dummy,
        tilts_in=mesh.tilts_in_view(),
        tilt_in_grad_arr=grad_in,
    )
    e_out = tilt_rim_source_out.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_dummy,
        tilts_out=mesh.tilts_out_view(),
        tilt_out_grad_arr=grad_out,
    )

    assert np.isclose(float(e_b), float(e_in + e_out), rtol=1e-12, atol=1e-12)
    assert np.allclose(grad_in_b, grad_in)
    assert np.allclose(grad_out_b, grad_out)
