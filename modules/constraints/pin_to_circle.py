"""Constraint module to pin selected entities to a circle.

The circle is defined by:

- a plane normal ``pin_to_circle_normal`` (per-entity or global, default ``[0,0,1]``),
- a center point ``pin_to_circle_point`` lying on that plane (default ``[0,0,0]``),
- a radius ``pin_to_circle_radius`` (default ``1.0``).

Attach ``"constraints": ["pin_to_circle"]`` to a vertex or an edge options dict
to have its vertices projected onto the specified circle.

Fit mode
--------
For cases where you want a circular rim but do not want to pin the circle to a
fixed location in space, set ``pin_to_circle_mode`` to ``"fit"`` on the tagged
entities (or in global parameters). In this mode, the constraint computes a
best-fit circle from the currently tagged vertices and projects them back onto
that circle. This allows the circle to translate/rotate with the mesh while
remaining circular.

You can optionally specify:
- ``pin_to_circle_group``: separate multiple fitted circles within one mesh.
- ``pin_to_circle_normal``: keep the plane normal fixed (otherwise it is fitted).
- ``pin_to_circle_radius``: keep radius fixed (otherwise it is fitted).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("membrane_solver")


def _normalize(vec: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-15:
        return None
    return vec / norm


def _default_tangent(normal: np.ndarray) -> np.ndarray:
    # Pick any vector not parallel to the normal, then orthogonalize.
    trial = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(np.dot(trial, normal)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0], dtype=float)
    tangent = trial - np.dot(trial, normal) * normal
    tangent = _normalize(tangent)
    if tangent is None:
        tangent = np.array([1.0, 0.0, 0.0], dtype=float)
    return tangent


def _mode_from_options(mesh, options: dict | None) -> str:
    gp = getattr(mesh, "global_parameters", None)
    raw = None
    if options and options.get("pin_to_circle_mode") is not None:
        raw = options.get("pin_to_circle_mode")
    elif gp is not None and gp.get("pin_to_circle_mode") is not None:
        raw = gp.get("pin_to_circle_mode")
    mode = str(raw or "fixed").lower()
    if mode == "fit":
        return "fit"
    if mode in {"slide", "normal", "normal_only", "slide_normal"}:
        return "slide"
    return "fixed"


def _group_from_options(options: dict | None):
    if not options:
        return "default"
    group = options.get("pin_to_circle_group")
    return "default" if group is None else group


def _resolve_circle(mesh, options: dict | None):
    gp = getattr(mesh, "global_parameters", None)

    def pick(key: str, default):
        if options and options.get(key) is not None:
            return options.get(key)
        if gp is not None and gp.get(key) is not None:
            return gp.get(key)
        return default

    normal_raw = pick("pin_to_circle_normal", [0.0, 0.0, 1.0])
    center_raw = pick("pin_to_circle_point", [0.0, 0.0, 0.0])
    radius = pick("pin_to_circle_radius", 1.0)

    normal = np.asarray(normal_raw, dtype=float)
    normal = _normalize(normal)
    if normal is None:
        logger.warning("pin_to_circle: normal is near zero; skipping projection.")
        return None

    center = np.asarray(center_raw, dtype=float)
    radius = float(radius)
    if radius <= 0.0:
        logger.warning("pin_to_circle: radius must be positive; skipping projection.")
        return None

    return normal, center, radius


def _resolve_fit_params(
    mesh, option_sources: list[dict]
) -> tuple[np.ndarray | None, float | None]:
    gp = getattr(mesh, "global_parameters", None)

    def pick(key: str):
        for opts in option_sources:
            if opts and opts.get(key) is not None:
                return opts.get(key)
        if gp is not None and gp.get(key) is not None:
            return gp.get(key)
        return None

    normal_raw = pick("pin_to_circle_normal")
    radius_raw = pick("pin_to_circle_radius")

    normal = None
    if normal_raw is not None:
        normal = _normalize(np.asarray(normal_raw, dtype=float))
        if normal is None:
            logger.warning(
                "pin_to_circle (fit): normal is near zero; falling back to fitted normal."
            )
            normal = None

    radius = None
    if radius_raw is not None:
        try:
            radius = float(radius_raw)
        except (TypeError, ValueError):
            radius = None
        if radius is not None and radius <= 0.0:
            logger.warning(
                "pin_to_circle (fit): radius must be positive; falling back to fitted radius."
            )
            radius = None

    return normal, radius


def _orthonormal_basis_from_normal(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = _default_tangent(normal)
    v = np.cross(normal, u)
    v = _normalize(v)
    if v is None:
        # Extremely degenerate; pick something arbitrary orthogonal.
        v = np.array([0.0, 0.0, 1.0], dtype=float)
        v = v - np.dot(v, normal) * normal
        v = _normalize(v) or np.array([0.0, 1.0, 0.0], dtype=float)
    return u, v


def _fit_plane_normal(points: np.ndarray) -> np.ndarray | None:
    if points.shape[0] < 3:
        return None
    centroid = np.mean(points, axis=0)
    X = points - centroid
    # PCA normal is the smallest singular vector.
    try:
        _, _, vh = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    normal = vh[-1, :]
    return _normalize(normal)


def _fit_circle_in_plane(
    points_3d: np.ndarray, normal: np.ndarray, radius_fixed: float | None
) -> tuple[np.ndarray, float] | None:
    """Fit a circle to 3D points constrained to a plane with given normal."""
    if points_3d.shape[0] < 3:
        return None

    centroid = np.mean(points_3d, axis=0)
    # Project points onto plane through centroid.
    p = points_3d - np.dot(points_3d - centroid, normal)[:, None] * normal[None, :]

    u, v = _orthonormal_basis_from_normal(normal)
    rel = p - centroid
    x = rel @ u
    y = rel @ v

    A = np.stack([2.0 * x, 2.0 * y, np.ones_like(x)], axis=1)
    b = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    a, b0, d = (float(sol[0]), float(sol[1]), float(sol[2]))
    r_sq = d + a * a + b0 * b0
    if not np.isfinite(r_sq) or r_sq <= 1e-18:
        return None
    radius_fit = float(np.sqrt(r_sq))

    radius = radius_fit if radius_fixed is None else float(radius_fixed)
    if not np.isfinite(radius) or radius <= 0.0:
        return None

    center = centroid + a * u + b0 * v
    return center, radius


def _project_point_to_circle(
    pos: np.ndarray, normal: np.ndarray, center: np.ndarray, radius: float
) -> np.ndarray:
    """Project a 3D point onto the circle defined by normal, center, and radius."""
    # Project onto plane.
    pos_plane = pos - np.dot(pos - center, normal) * normal
    offset = pos_plane - center
    tangent = _normalize(offset)
    if tangent is None:
        tangent = _default_tangent(normal)
    return center + radius * tangent


def _entity_has_constraint(options: dict | None) -> bool:
    if not options:
        return False
    constraints = options.get("constraints")
    if constraints is None:
        return False
    if isinstance(constraints, str):
        return constraints == "pin_to_circle"
    if isinstance(constraints, list):
        return "pin_to_circle" in constraints
    return False


def enforce_constraint(mesh, **_kwargs):
    tagged_vertices = [
        v
        for v in mesh.vertices.values()
        if _entity_has_constraint(getattr(v, "options", None))
    ]
    tagged_edges = [
        e
        for e in mesh.edges.values()
        if _entity_has_constraint(getattr(e, "options", None))
    ]

    # Fast path: pure fixed mode (legacy behavior).
    vertices_are_fixed = all(
        _mode_from_options(mesh, getattr(v, "options", None)) == "fixed"
        for v in tagged_vertices
    )
    edges_are_fixed = all(
        _mode_from_options(mesh, getattr(e, "options", None)) == "fixed"
        for e in tagged_edges
    )
    if vertices_are_fixed and edges_are_fixed:
        for vertex in tagged_vertices:
            params = _resolve_circle(mesh, getattr(vertex, "options", None))
            if params is None:
                continue
            normal, center, radius = params
            vertex.position[:] = _project_point_to_circle(
                vertex.position, normal, center, radius
            )

        for edge in tagged_edges:
            params = _resolve_circle(mesh, getattr(edge, "options", None))
            if params is None:
                continue
            normal, center, radius = params
            for vidx in (int(edge.tail_index), int(edge.head_index)):
                vertex = mesh.vertices.get(vidx)
                if vertex is None:
                    continue
                vertex.position[:] = _project_point_to_circle(
                    vertex.position, normal, center, radius
                )
        return

    # Mixed mode: apply fixed-mode entities directly, and group fit-mode entities.
    fit_groups: dict[object, dict[str, object]] = {}

    def add_fit_vertex(vidx: int, options: dict | None):
        group = _group_from_options(options)
        entry = fit_groups.setdefault(
            group, {"vertex_ids": set(), "options": [], "mode": "fit"}
        )
        entry["vertex_ids"].add(vidx)
        if options:
            entry["options"].append(options)
            entry["mode"] = _mode_from_options(mesh, options)

    for vertex in tagged_vertices:
        options = getattr(vertex, "options", None)
        mode = _mode_from_options(mesh, options)
        if mode in {"fit", "slide"}:
            add_fit_vertex(int(vertex.index), options)
            continue
        params = _resolve_circle(mesh, options)
        if params is None:
            continue
        normal, center, radius = params
        vertex.position[:] = _project_point_to_circle(
            vertex.position, normal, center, radius
        )

    for edge in tagged_edges:
        options = getattr(edge, "options", None)
        mode = _mode_from_options(mesh, options)
        if mode in {"fit", "slide"}:
            add_fit_vertex(int(edge.tail_index), options)
            add_fit_vertex(int(edge.head_index), options)
            continue
        params = _resolve_circle(mesh, options)
        if params is None:
            continue
        normal, center, radius = params
        for vidx in (int(edge.tail_index), int(edge.head_index)):
            vertex = mesh.vertices.get(vidx)
            if vertex is None:
                continue
            vertex.position[:] = _project_point_to_circle(
                vertex.position, normal, center, radius
            )

    for group, spec in fit_groups.items():
        vertex_ids = sorted(spec["vertex_ids"])
        if len(vertex_ids) < 3:
            logger.warning(
                "pin_to_circle (fit): group %r has <3 vertices; skipping.", group
            )
            continue

        points = np.array([mesh.vertices[i].position for i in vertex_ids], dtype=float)
        option_sources = list(spec["options"])
        normal_fixed, radius_fixed = _resolve_fit_params(mesh, option_sources)
        normal = normal_fixed if normal_fixed is not None else _fit_plane_normal(points)
        if normal is None:
            logger.warning("pin_to_circle (fit): could not fit a plane; skipping.")
            continue

        mode = str(spec.get("mode") or "fit").lower()
        if mode == "slide":
            # Slide mode: only allow translation along the fixed normal direction.
            # This is useful for benchmarks where the disk rim can move up/down
            # (kink depth) but should not rotate in space.
            gp = getattr(mesh, "global_parameters", None)
            base_point_raw = None
            for opts in option_sources:
                if opts and opts.get("pin_to_circle_point") is not None:
                    base_point_raw = opts.get("pin_to_circle_point")
                    break
            if base_point_raw is None and gp is not None:
                base_point_raw = gp.get("pin_to_circle_point")
            if base_point_raw is None:
                base_point_raw = [0.0, 0.0, 0.0]
            base_point = np.asarray(base_point_raw, dtype=float).reshape(3)

            offsets = points - base_point[None, :]
            t = float(np.mean(offsets @ normal))
            center = base_point + t * normal

            # Fit radius in the plane if not fixed.
            points_plane = (
                points
                - ((points - center[None, :]) @ normal)[:, None] * normal[None, :]
            )
            radial = points_plane - center[None, :]
            radial = radial - (radial @ normal)[:, None] * normal[None, :]
            r_vals = np.linalg.norm(radial, axis=1)
            radius = (
                float(np.mean(r_vals)) if radius_fixed is None else float(radius_fixed)
            )
            if not np.isfinite(radius) or radius <= 0.0:
                logger.warning(
                    "pin_to_circle (slide): invalid fitted radius; skipping."
                )
                continue
        else:
            fitted = _fit_circle_in_plane(points, normal, radius_fixed)
            if fitted is None:
                logger.warning("pin_to_circle (fit): could not fit a circle; skipping.")
                continue
            center, radius = fitted

        u, _ = _orthonormal_basis_from_normal(normal)
        for vidx in vertex_ids:
            vertex = mesh.vertices.get(int(vidx))
            if vertex is None:
                continue
            pos = vertex.position
            pos_plane = pos - np.dot(pos - center, normal) * normal
            offset = pos_plane - center
            tangent = _normalize(offset)
            if tangent is None:
                tangent = u
            vertex.position[:] = center + radius * tangent


def _collect_pin_to_circle_targets(mesh):
    fixed_targets: list[tuple[int, dict | None]] = []
    fit_groups: dict[object, dict[str, object]] = {}

    def add_fit_vertex(vidx: int, options: dict | None):
        group = _group_from_options(options)
        entry = fit_groups.setdefault(
            group, {"vertex_ids": set(), "options": [], "mode": "fit"}
        )
        entry["vertex_ids"].add(int(vidx))
        if options:
            entry["options"].append(options)
            entry["mode"] = _mode_from_options(mesh, options)

    for vertex in mesh.vertices.values():
        options = getattr(vertex, "options", None)
        if not _entity_has_constraint(options):
            continue
        mode = _mode_from_options(mesh, options)
        if mode in {"fit", "slide"}:
            add_fit_vertex(int(vertex.index), options)
            continue
        fixed_targets.append((int(vertex.index), options))

    for edge in mesh.edges.values():
        options = getattr(edge, "options", None)
        if not _entity_has_constraint(options):
            continue
        mode = _mode_from_options(mesh, options)
        if mode in {"fit", "slide"}:
            add_fit_vertex(int(edge.tail_index), options)
            add_fit_vertex(int(edge.head_index), options)
            continue
        fixed_targets.append((int(edge.tail_index), options))
        fixed_targets.append((int(edge.head_index), options))

    return fixed_targets, fit_groups


def _resolve_fit_circle_for_group(mesh, spec: dict[str, object]):
    vertex_ids = sorted(spec["vertex_ids"])
    if len(vertex_ids) < 3:
        return None
    points = np.array([mesh.vertices[i].position for i in vertex_ids], dtype=float)
    option_sources = list(spec["options"])
    normal_fixed, radius_fixed = _resolve_fit_params(mesh, option_sources)
    normal = normal_fixed if normal_fixed is not None else _fit_plane_normal(points)
    if normal is None:
        return None

    mode = str(spec.get("mode") or "fit").lower()
    if mode == "slide":
        gp = getattr(mesh, "global_parameters", None)
        base_point_raw = None
        for opts in option_sources:
            if opts and opts.get("pin_to_circle_point") is not None:
                base_point_raw = opts.get("pin_to_circle_point")
                break
        if base_point_raw is None and gp is not None:
            base_point_raw = gp.get("pin_to_circle_point")
        if base_point_raw is None:
            base_point_raw = [0.0, 0.0, 0.0]
        base_point = np.asarray(base_point_raw, dtype=float).reshape(3)

        offsets = points - base_point[None, :]
        t = float(np.mean(offsets @ normal))
        center = base_point + t * normal

        points_plane = (
            points - ((points - center[None, :]) @ normal)[:, None] * normal[None, :]
        )
        radial = points_plane - center[None, :]
        radial = radial - (radial @ normal)[:, None] * normal[None, :]
        r_vals = np.linalg.norm(radial, axis=1)
        radius = float(np.mean(r_vals)) if radius_fixed is None else float(radius_fixed)
        if not np.isfinite(radius) or radius <= 0.0:
            return None
    else:
        fitted = _fit_circle_in_plane(points, normal, radius_fixed)
        if fitted is None:
            return None
        center, radius = fitted
    return (
        np.asarray(normal, dtype=float),
        np.asarray(center, dtype=float),
        float(radius),
    )


def _circle_constraint_gradients_for_vertex(
    *,
    pos: np.ndarray,
    normal: np.ndarray,
    center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    g_plane = np.asarray(normal, dtype=float)
    pos_plane = pos - np.dot(pos - center, normal) * normal
    radial = pos_plane - center
    radial_hat = _normalize(radial)
    if radial_hat is None:
        radial_hat = _default_tangent(normal)
    return g_plane, np.asarray(radial_hat, dtype=float)


def constraint_gradients(mesh, _global_params) -> list[dict[int, np.ndarray]] | None:
    """Return shape-constraint gradients for KKT projection."""
    gradients: list[dict[int, np.ndarray]] = []
    fixed_targets, fit_groups = _collect_pin_to_circle_targets(mesh)

    for vidx, options in fixed_targets:
        vertex = mesh.vertices.get(int(vidx))
        if vertex is None or getattr(vertex, "fixed", False):
            continue
        params = _resolve_circle(mesh, options)
        if params is None:
            continue
        normal, center, _radius = params
        g_plane, g_radial = _circle_constraint_gradients_for_vertex(
            pos=vertex.position, normal=normal, center=center
        )
        gradients.append({int(vidx): g_plane})
        gradients.append({int(vidx): g_radial})

    for spec in fit_groups.values():
        resolved = _resolve_fit_circle_for_group(mesh, spec)
        if resolved is None:
            continue
        normal, center, _radius = resolved
        for vidx in sorted(spec["vertex_ids"]):
            vertex = mesh.vertices.get(int(vidx))
            if vertex is None or getattr(vertex, "fixed", False):
                continue
            g_plane, g_radial = _circle_constraint_gradients_for_vertex(
                pos=vertex.position, normal=normal, center=center
            )
            gradients.append({int(vidx): g_plane})
            gradients.append({int(vidx): g_radial})

    return gradients or None


def constraint_gradients_array(
    mesh,
    _global_params,
    *,
    positions: np.ndarray,
    index_map: dict[int, int],
) -> list[np.ndarray] | None:
    """Dense array variant of pin-to-circle shape gradients."""
    _ = positions
    dict_grads = constraint_gradients(mesh, _global_params)
    if not dict_grads:
        return None
    arr_grads: list[np.ndarray] = []
    for gC in dict_grads:
        g_arr = np.zeros((len(index_map), 3), dtype=float)
        for vidx, gvec in gC.items():
            row = index_map.get(int(vidx))
            if row is None:
                continue
            g_arr[row] += np.asarray(gvec, dtype=float)
        arr_grads.append(g_arr)
    return arr_grads or None


__all__ = ["enforce_constraint", "constraint_gradients", "constraint_gradients_array"]
