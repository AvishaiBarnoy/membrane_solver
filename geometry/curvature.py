# geometry/curvature.py

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from geometry.entities import Mesh, _fast_cross


@dataclass(frozen=True)
class CurvatureFields:
    """Vectorized discrete curvature fields on a triangle mesh.

    All arrays are in vertex-row order (see ``Mesh.vertex_ids``).

    Notes
    -----
    - Mean curvature is computed from the cotangent Laplace-Beltrami operator
      using mixed Voronoi dual areas (Meyer et al. 2003).
    - Gaussian curvature is computed from angle defects:
        defect_i = 2π - sum(face angles at vertex i)
      and converted to a pointwise value via K_i = defect_i / A_i.
    - Principal curvatures are derived from H and K via:
        k1,k2 = H ± sqrt(max(H^2 - K, 0))
      Here H is taken as non-negative magnitude to avoid dependence on global
      facet orientation conventions.
    """

    mean_curvature_normal: np.ndarray  # (N, 3)  H * n
    mean_curvature: np.ndarray  # (N,) |H|
    mixed_area: np.ndarray  # (N,) dual area A_i
    angle_defect: np.ndarray  # (N,) integrated Gaussian curvature
    gaussian_curvature: np.ndarray  # (N,) pointwise K
    principal_curvatures: np.ndarray  # (N, 2) (k1,k2)


def _call_fortran_curvature_kernel(
    kernel,
    *,
    positions: np.ndarray,
    tri_rows: np.ndarray,
    k_vecs: np.ndarray,
    vertex_areas: np.ndarray,
    weights: np.ndarray,
    va0: np.ndarray,
    va1: np.ndarray,
    va2: np.ndarray,
    zero_based: int,
) -> bool:
    """Call curvature kernel across legacy and optional-output signatures.

    Returns ``True`` when optional per-triangle area outputs (va0/va1/va2)
    were populated by the kernel call.
    """
    try:
        kernel(
            positions,
            tri_rows,
            k_vecs,
            vertex_areas,
            weights,
            zero_based,
            va0,
            va1,
            va2,
        )
        return True
    except (TypeError, ValueError):
        pass

    try:
        # Some wrappers may place optional outputs before ``zero_based``.
        kernel(
            positions,
            tri_rows,
            k_vecs,
            vertex_areas,
            weights,
            va0,
            va1,
            va2,
            zero_based,
        )
        return True
    except (TypeError, ValueError):
        pass

    kernel(positions, tri_rows, k_vecs, vertex_areas, weights, zero_based)
    return False


def compute_curvature_data(
    mesh: Mesh, positions: np.ndarray, index_map: Dict[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute integrated mean curvature vectors K_i, Mixed Voronoi Areas A_i,
    and the cotangent weights.

    References: Meyer et al. (2003) 'Discrete Differential-Geometry Operators'
    """
    _ = index_map
    n_verts = len(mesh.vertex_ids)
    tri_rows, _ = mesh.triangle_row_cache()

    if tri_rows is None:
        return np.zeros((n_verts, 3)), np.zeros(n_verts), np.array([]), np.array([])

    use_cache = mesh._geometry_cache_active(positions)
    if use_cache and mesh._curvature_version == mesh._version:
        cached = mesh._curvature_cache
        if (
            cached.get("curvature_rows_version") == mesh._facet_loops_version
            and "k_vecs" in cached
            and "vertex_areas" in cached
            and "weights" in cached
            and "tri_rows" in cached
        ):
            return (
                cached["k_vecs"],
                cached["vertex_areas"],
                cached["weights"],
                cached["tri_rows"],
            )

    from fortran_kernels.loader import get_tilt_curvature_kernel

    kernel_spec = get_tilt_curvature_kernel()
    if kernel_spec is not None:
        strict = os.environ.get("MEMBRANE_FORTRAN_STRICT_NOCOPY") in {
            "1",
            "true",
            "TRUE",
        }
        if positions.dtype != np.float64 or tri_rows.dtype != np.int32:
            if strict:
                raise TypeError(
                    "Fortran tilt kernels require float64 positions and int32 tri_rows."
                )
            kernel_spec = None
        elif not (positions.flags["F_CONTIGUOUS"] and tri_rows.flags["F_CONTIGUOUS"]):
            if strict:
                raise ValueError(
                    "Fortran tilt kernels require F-contiguous positions/tri_rows (to avoid hidden copies)."
                )
            kernel_spec = None

    if kernel_spec is not None:
        nf = tri_rows.shape[0]
        va0 = np.zeros(nf, dtype=np.float64, order="F")
        va1 = np.zeros(nf, dtype=np.float64, order="F")
        va2 = np.zeros(nf, dtype=np.float64, order="F")
        if kernel_spec.expects_transpose:
            pos_t = positions.T
            tri_t = tri_rows.T
            k_vecs = np.zeros((3, n_verts), dtype=np.float64, order="F")
            vertex_areas = np.zeros(n_verts, dtype=np.float64, order="F")
            weights = np.zeros((3, nf), dtype=np.float64, order="F")
            has_optional_areas = _call_fortran_curvature_kernel(
                kernel_spec.func,
                positions=pos_t,
                tri_rows=tri_t,
                k_vecs=k_vecs,
                vertex_areas=vertex_areas,
                weights=weights,
                va0=va0,
                va1=va1,
                va2=va2,
                zero_based=1,
            )
            k_vecs = np.asarray(k_vecs).T
            vertex_areas = np.asarray(vertex_areas)
            weights = np.asarray(weights).T
        else:
            k_vecs = np.zeros((n_verts, 3), dtype=np.float64, order="F")
            vertex_areas = np.zeros(n_verts, dtype=np.float64, order="F")
            weights = np.zeros((nf, 3), dtype=np.float64, order="F")
            has_optional_areas = _call_fortran_curvature_kernel(
                kernel_spec.func,
                positions=positions,
                tri_rows=tri_rows,
                k_vecs=k_vecs,
                vertex_areas=vertex_areas,
                weights=weights,
                va0=va0,
                va1=va1,
                va2=va2,
                zero_based=1,
            )

        if use_cache:
            mesh._curvature_cache["k_vecs"] = k_vecs
            mesh._curvature_cache["vertex_areas"] = vertex_areas
            mesh._curvature_cache["weights"] = weights
            mesh._curvature_cache["tri_rows"] = tri_rows
            if has_optional_areas:
                mesh._curvature_cache["va0_raw"] = np.asarray(va0)
                mesh._curvature_cache["va1_raw"] = np.asarray(va1)
                mesh._curvature_cache["va2_raw"] = np.asarray(va2)
            mesh._curvature_cache["curvature_rows_version"] = mesh._facet_loops_version
            mesh._curvature_version = mesh._version
        return k_vecs, vertex_areas, weights, tri_rows

    v0 = positions[tri_rows[:, 0]]
    v1 = positions[tri_rows[:, 1]]
    v2 = positions[tri_rows[:, 2]]

    # Edges
    e0 = v2 - v1  # opposite v0
    e1 = v0 - v2  # opposite v1
    e2 = v1 - v0  # opposite v2

    # Squared edge lengths
    l0_sq = np.einsum("ij,ij->i", e0, e0)
    l1_sq = np.einsum("ij,ij->i", e1, e1)
    l2_sq = np.einsum("ij,ij->i", e2, e2)

    # Triangle area (doubled)
    cross = _fast_cross(e1, e2)
    area_doubled = np.linalg.norm(cross, axis=1)
    area_doubled = np.maximum(area_doubled, 1e-12)

    # Cotangents
    def get_cot(a, b, areas_2):
        return np.einsum("ij,ij->i", a, b) / areas_2

    c0 = get_cot(-e1, e2, area_doubled)
    c1 = get_cot(-e2, e0, area_doubled)
    c2 = get_cot(-e0, e1, area_doubled)

    # 1. Curvature Vectors (Integrated)
    k_vecs = np.zeros((n_verts, 3), dtype=float, order="F")
    np.add.at(k_vecs, tri_rows[:, 0], 0.5 * (c1[:, None] * -e1 + c2[:, None] * e2))
    np.add.at(k_vecs, tri_rows[:, 1], 0.5 * (c2[:, None] * -e2 + c0[:, None] * e0))
    np.add.at(k_vecs, tri_rows[:, 2], 0.5 * (c0[:, None] * -e0 + c1[:, None] * e1))

    # 2. Mixed Voronoi Area
    # Formulas from Meyer et al. 2003
    tri_areas = 0.5 * area_doubled
    vertex_areas = np.zeros(n_verts, dtype=float)

    # Check for obtuse angles
    # cot < 0 means angle > 90 deg
    is_obtuse_v0 = c0 < 0
    is_obtuse_v1 = c1 < 0
    is_obtuse_v2 = c2 < 0
    any_obtuse = is_obtuse_v0 | is_obtuse_v1 | is_obtuse_v2

    # Case A: Non-obtuse triangle (Standard Voronoi)
    # Area contrib to v0 = 1/8 * ( l1^2 * cot_beta + l2^2 * cot_gamma )
    va0 = np.where(~any_obtuse, (l1_sq * c1 + l2_sq * c2) / 8.0, 0.0)
    va1 = np.where(~any_obtuse, (l2_sq * c2 + l0_sq * c0) / 8.0, 0.0)
    va2 = np.where(~any_obtuse, (l0_sq * c0 + l1_sq * c1) / 8.0, 0.0)

    # Case B: Obtuse triangle
    # If angle at v is obtuse, area = T_area / 2
    # If other angle is obtuse, area = T_area / 4
    va0 = np.where(is_obtuse_v0, tri_areas / 2.0, va0)
    va0 = np.where(is_obtuse_v1 | is_obtuse_v2, tri_areas / 4.0, va0)

    va1 = np.where(is_obtuse_v1, tri_areas / 2.0, va1)
    va1 = np.where(is_obtuse_v0 | is_obtuse_v2, tri_areas / 4.0, va1)

    va2 = np.where(is_obtuse_v2, tri_areas / 2.0, va2)
    va2 = np.where(is_obtuse_v0 | is_obtuse_v1, tri_areas / 4.0, va2)

    np.add.at(vertex_areas, tri_rows[:, 0], va0)
    np.add.at(vertex_areas, tri_rows[:, 1], va1)
    np.add.at(vertex_areas, tri_rows[:, 2], va2)

    weights = np.empty((len(tri_rows), 3), dtype=float, order="F")
    weights[:, 0] = c0
    weights[:, 1] = c1
    weights[:, 2] = c2
    if use_cache:
        mesh._curvature_cache["k_vecs"] = k_vecs
        mesh._curvature_cache["vertex_areas"] = vertex_areas
        mesh._curvature_cache["weights"] = weights
        mesh._curvature_cache["tri_rows"] = tri_rows
        mesh._curvature_cache["curvature_rows_version"] = mesh._facet_loops_version
        mesh._curvature_version = mesh._version
    return k_vecs, vertex_areas, weights, tri_rows


def compute_angle_defects(
    mesh: Mesh, positions: np.ndarray, index_map: Dict[int, int]
) -> np.ndarray:
    """Compute per-vertex angle defects (integrated Gaussian curvature).

    For a closed, manifold triangle mesh the sum of all defects equals
    ``2π * chi`` (Gauss–Bonnet), where ``chi`` is the Euler characteristic.

    Parameters
    ----------
    mesh:
        Mesh topology; must have triangle facet loops cached.
    positions:
        Dense positions array in vertex-row order.
    index_map:
        Vertex id -> row index map. Included for interface consistency.

    Returns
    -------
    np.ndarray
        Array of shape ``(N_vertices,)`` with the angle defect at each vertex.
    """
    n_verts = len(mesh.vertex_ids)
    tri_rows, _ = mesh.triangle_row_cache()
    if tri_rows is None or len(tri_rows) == 0:
        return np.zeros(n_verts, dtype=float)

    v0 = positions[tri_rows[:, 0]]
    v1 = positions[tri_rows[:, 1]]
    v2 = positions[tri_rows[:, 2]]

    # Edge lengths for law-of-cosines angles.
    a = np.linalg.norm(v2 - v1, axis=1)  # opposite v0
    b = np.linalg.norm(v0 - v2, axis=1)  # opposite v1
    c = np.linalg.norm(v1 - v0, axis=1)  # opposite v2

    eps = 1e-15
    a = np.maximum(a, eps)
    b = np.maximum(b, eps)
    c = np.maximum(c, eps)

    # Angles at v0, v1, v2.
    cos0 = (b * b + c * c - a * a) / (2.0 * b * c)
    cos1 = (c * c + a * a - b * b) / (2.0 * c * a)
    cos2 = (a * a + b * b - c * c) / (2.0 * a * b)
    cos0 = np.clip(cos0, -1.0, 1.0)
    cos1 = np.clip(cos1, -1.0, 1.0)
    cos2 = np.clip(cos2, -1.0, 1.0)

    ang0 = np.arccos(cos0)
    ang1 = np.arccos(cos1)
    ang2 = np.arccos(cos2)

    angle_sums = np.zeros(n_verts, dtype=float)
    np.add.at(angle_sums, tri_rows[:, 0], ang0)
    np.add.at(angle_sums, tri_rows[:, 1], ang1)
    np.add.at(angle_sums, tri_rows[:, 2], ang2)

    defects = (2.0 * np.pi) - angle_sums

    boundary_vids = getattr(mesh, "boundary_vertex_ids", None) or []
    if boundary_vids:
        boundary_rows = [index_map[vid] for vid in boundary_vids if vid in index_map]
        if boundary_rows:
            defects[boundary_rows] = 0.0

    return defects


def compute_curvature_fields(
    mesh: Mesh, positions: np.ndarray, index_map: Dict[int, int]
) -> CurvatureFields:
    """Compute mean/Gaussian/principal curvature fields (vectorized).

    This is a shared helper intended for energy modules and diagnostics.
    """
    k_vecs, vertex_areas, _, tri_rows = compute_curvature_data(
        mesh, positions, index_map
    )
    n_verts = len(mesh.vertex_ids)

    if tri_rows is None or len(tri_rows) == 0:
        zeros3 = np.zeros((n_verts, 3), dtype=float)
        zeros1 = np.zeros(n_verts, dtype=float)
        return CurvatureFields(
            mean_curvature_normal=zeros3,
            mean_curvature=zeros1,
            mixed_area=zeros1,
            angle_defect=zeros1,
            gaussian_curvature=zeros1,
            principal_curvatures=np.zeros((n_verts, 2), dtype=float),
        )

    safe_areas = np.maximum(vertex_areas, 1e-12)
    mean_curvature_normal = k_vecs / (2.0 * safe_areas[:, None])
    mean_curvature = np.linalg.norm(mean_curvature_normal, axis=1)

    angle_defect = compute_angle_defects(mesh, positions, index_map)
    gaussian_curvature = angle_defect / safe_areas

    disc = np.maximum(mean_curvature * mean_curvature - gaussian_curvature, 0.0)
    root = np.sqrt(disc)
    k1 = mean_curvature + root
    k2 = mean_curvature - root
    principal = np.column_stack([k1, k2])

    return CurvatureFields(
        mean_curvature_normal=mean_curvature_normal,
        mean_curvature=mean_curvature,
        mixed_area=vertex_areas,
        angle_defect=angle_defect,
        gaussian_curvature=gaussian_curvature,
        principal_curvatures=principal,
    )
