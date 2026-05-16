from core.exceptions import BodyOrientationError

from .body import Body


def validate_body_orientation(mesh) -> bool:
    """Validate that facets in each body have consistent orientation."""
    if not mesh.bodies:
        return True

    for body in mesh.bodies.values():
        edge_uses: dict[int, list[tuple[int, int]]] = {}
        for fid in body.facet_indices:
            facet = mesh.facets.get(fid)
            if facet is None:
                raise BodyOrientationError(
                    f"Body {body.index} references missing facet {fid}.",
                    body_index=body.index,
                    mesh=mesh,
                )
            for signed_ei in facet.edge_indices:
                ei = abs(int(signed_ei))
                sign = 1 if int(signed_ei) > 0 else -1
                edge_uses.setdefault(ei, []).append((facet.index, sign))

        for ei, uses in edge_uses.items():
            if len(uses) > 2:
                facets = [fid for fid, _ in uses]
                raise BodyOrientationError(
                    f"Body {body.index} is non-manifold: edge {ei} is used by "
                    f"{len(uses)} facets {facets}.",
                    body_index=body.index,
                    edge_index=ei,
                    mesh=mesh,
                )
            if len(uses) == 2:
                (f0, s0), (f1, s1) = uses
                if s0 != -s1:
                    raise BodyOrientationError(
                        f"Body {body.index} has inconsistent facet orientation across "
                        f"edge {ei}: facets {f0} and {f1} traverse it with the same "
                        "direction.",
                        body_index=body.index,
                        edge_index=ei,
                        facet_indices=(f0, f1),
                        mesh=mesh,
                    )

    return True


def _body_edge_uses(mesh, body: Body) -> dict[int, list[tuple[int, int]]]:
    """Return mapping ``edge_id -> [(facet_id, sign), ...]`` within a body."""
    edge_uses: dict[int, list[tuple[int, int]]] = {}
    for fid in body.facet_indices:
        facet = mesh.facets.get(fid)
        if facet is None:
            raise BodyOrientationError(
                f"Body {body.index} references missing facet {fid}.",
                body_index=body.index,
                mesh=mesh,
            )
        for signed_ei in facet.edge_indices:
            ei = abs(int(signed_ei))
            sign = 1 if int(signed_ei) > 0 else -1
            edge_uses.setdefault(ei, []).append((facet.index, sign))
    return edge_uses


def _body_is_closed(mesh, body: Body) -> bool:
    """Return ``True`` when the body's facets form a closed 2-manifold."""
    edge_uses = mesh._body_edge_uses(body)
    return bool(edge_uses) and all(len(uses) == 2 for uses in edge_uses.values())


def validate_body_outwardness(mesh, volume_tol: float = 1e-12) -> bool:
    """Validate that closed bodies have outward (positive) signed volume."""
    if not mesh.bodies:
        return True

    tol = float(volume_tol)
    for body in mesh.bodies.values():
        if not mesh._body_is_closed(body):
            continue
        vol = float(body.compute_volume(mesh))
        if abs(vol) <= tol:
            continue
        if vol < -tol:
            raise BodyOrientationError(
                f"Body {body.index} is inward-oriented (signed volume {vol:.6g} < 0).",
                body_index=body.index,
                mesh=mesh,
            )
    return True


def orient_body_outward(mesh, body_index: int, volume_tol: float = 1e-12) -> int:
    """Flip all facets in ``body_index`` if signed volume is negative."""
    body = mesh.bodies.get(body_index)
    if body is None:
        raise KeyError(f"Body {body_index} not found in mesh.")

    if not mesh._body_is_closed(body):
        return 0

    tol = float(volume_tol)
    vol = float(body.compute_volume(mesh))
    if vol >= -tol:
        return 0

    for fid in body.facet_indices:
        facet = mesh.facets.get(fid)
        if facet is None:
            continue
        facet.edge_indices = [-int(ei) for ei in reversed(facet.edge_indices)]

    mesh.build_facet_vertex_loops()
    mesh.increment_version()
    return len(body.facet_indices)


def orient_body_facets(mesh, body_index: int) -> int:
    """Re-orient facets in a body to make shared-edge orientations consistent."""
    body = mesh.bodies.get(body_index)
    if body is None:
        raise KeyError(f"Body {body_index} not found in mesh.")

    facet_ids = list(body.facet_indices)
    if not facet_ids:
        return 0

    edge_uses: dict[int, list[tuple[int, int]]] = {}
    for fid in facet_ids:
        facet = mesh.facets.get(fid)
        if facet is None:
            raise BodyOrientationError(
                f"Body {body.index} references missing facet {fid}.",
                body_index=body.index,
                mesh=mesh,
            )
        for signed_ei in facet.edge_indices:
            ei = abs(int(signed_ei))
            sign = 1 if int(signed_ei) > 0 else -1
            edge_uses.setdefault(ei, []).append((facet.index, sign))

    adjacency: dict[int, list[tuple[int, int, int]]] = {fid: [] for fid in facet_ids}
    for ei, uses in edge_uses.items():
        if len(uses) > 2:
            facets = [fid for fid, _ in uses]
            raise BodyOrientationError(
                f"Cannot orient body {body.index}: edge {ei} is used by "
                f"{len(uses)} facets {facets}.",
                body_index=body.index,
                edge_index=ei,
                mesh=mesh,
            )
        if len(uses) != 2:
            continue
        (f0, s0), (f1, s1) = uses
        adjacency.setdefault(f0, []).append((f1, s0, s1))
        adjacency.setdefault(f1, []).append((f0, s1, s0))

    flips: dict[int, int] = {}
    queue: list[int] = []
    for start in facet_ids:
        if start in flips:
            continue
        flips[start] = 0
        queue.append(start)
        while queue:
            fid = queue.pop()
            f_flip = flips[fid]
            for nbr, sign_here, sign_nbr in adjacency.get(fid, []):
                nbr_flip = f_flip if sign_here == -sign_nbr else 1 - f_flip
                if nbr in flips:
                    if flips[nbr] != nbr_flip:
                        raise BodyOrientationError(
                            f"Cannot orient body {body.index}: inconsistent parity "
                            f"assignment between facets {fid} and {nbr}.",
                            body_index=body.index,
                            edge_index=None,
                            facet_indices=(fid, nbr),
                            mesh=mesh,
                        )
                    continue
                flips[nbr] = nbr_flip
                queue.append(nbr)

    flipped_count = 0
    for fid, flip in flips.items():
        if not flip:
            continue
        facet = mesh.facets[fid]
        facet.edge_indices = [-int(ei) for ei in reversed(facet.edge_indices)]
        flipped_count += 1

    if flipped_count:
        mesh.build_facet_vertex_loops()
        mesh.increment_version()

    return flipped_count


def validate_triangles(mesh):
    """Validate that all facets are triangles (have exactly 3 oriented edges)."""
    for facet_idx in mesh.facets.keys():
        if len(mesh.facets[facet_idx].edge_indices) != 3:
            raise ValueError(
                f"Facet {facet_idx} does not have 3 edges. Found {len(mesh.facets[facet_idx].edge_indices)}."
            )
    return True


def validate_edge_indices(mesh):
    for facet_idx in mesh.facets.keys():
        for signed_index in mesh.facets[facet_idx].edge_indices:
            edge_index = abs(signed_index)
            if edge_index not in mesh.edges:
                raise ValueError(
                    f"Facet {facet_idx} uses invalid edge index {signed_index} (not found in edge list)."
                )
    return True
