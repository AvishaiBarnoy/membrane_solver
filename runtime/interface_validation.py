"""Validation helpers for mesh interface/topology assumptions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geometry.entities import Mesh


@dataclass(frozen=True)
class DiskInterfaceIssue:
    """A disk-boundary vertex that does not straddle disk↔membrane triangles."""

    vertex_id: int
    incident_presets: tuple[str, ...]


def validate_disk_interface_topology(mesh: Mesh, global_params) -> None:
    """Validate that the configured disk boundary is a true disk↔membrane interface.

    For Kozlov-style "disk embedded in membrane" benchmarks, the ring tagged as
    the disk boundary (e.g. rim_slope_match_disk_group) must be the actual
    topological interface between the disk-covered patch and the outer membrane
    patch. If that ring is instead an internal ring (connected only to disk-side
    triangles), discrete curvature/energy operators can develop large baselines
    and boundary conditions act at the wrong location.

    This validator enforces:
      - every vertex in the disk boundary group has incident triangles that
        include both 'disk' and non-'disk' presets among their vertex presets.

    The validator is only active when rim_slope_match_disk_group is set.

    Raises
    ------
    ValueError
        If the disk boundary group is present but does not straddle disk↔membrane
        connectivity.
    """
    # Opt-in guardrail: many meshes may use rim_slope_match_* groups for other
    # purposes, but the "disk embedded in membrane" interface assumptions are
    # only intended to be enforced when explicitly enabled.
    if not bool(global_params.get("disk_interface_validate", False)):
        return

    group = global_params.get("rim_slope_match_disk_group")
    if group is None:
        return
    group = str(group).strip()
    if not group:
        return

    mesh.build_facet_vertex_loops()

    disk_boundary_vids: list[int] = []
    for vid, vertex in mesh.vertices.items():
        opts = getattr(vertex, "options", None) or {}
        if (
            opts.get("rim_slope_match_group") == group
            or opts.get("tilt_thetaB_group") == group
        ):
            disk_boundary_vids.append(int(vid))

    if not disk_boundary_vids:
        return

    issues: list[DiskInterfaceIssue] = []
    for vid in disk_boundary_vids:
        incident = mesh.vertex_to_facets.get(int(vid)) or set()
        presets: set[str] = set()
        for fid in incident:
            loop = mesh.facet_vertex_loops.get(int(fid))
            if loop is None or len(loop) != 3:
                continue
            for v2 in loop:
                opts = getattr(mesh.vertices[int(v2)], "options", None) or {}
                presets.add(str(opts.get("preset") or ""))

        # Geometry-based straddle check: refinement introduces midpoint vertices that
        # may not inherit 'preset' tags. For Kozlov-style setups we can robustly
        # classify disk-side vs membrane-side by radial distance relative to the
        # disk boundary ring.
        try:
            mesh.build_position_cache()
            center_raw = (
                global_params.get("rim_slope_match_center")
                or global_params.get("tilt_thetaB_center")
                or [0.0, 0.0, 0.0]
            )
            center = np.asarray(center_raw, dtype=float).reshape(3)
        except Exception:  # pragma: no cover - defensive
            center = np.zeros(3, dtype=float)

        # Estimate disk radius R from the tagged ring itself.
        ring_r: list[float] = []
        for rid in disk_boundary_vids:
            try:
                p = mesh.vertices[int(rid)].position
                ring_r.append(float(np.linalg.norm((p - center)[:2])))
            except Exception:
                continue
        R = float(np.median(ring_r)) if ring_r else 0.0
        tol = max(1e-8, 1e-6 * max(1.0, abs(R)))

        r_vals: list[float] = []
        for fid in incident:
            loop = mesh.facet_vertex_loops.get(int(fid))
            if loop is None or len(loop) != 3:
                continue
            for v2 in loop:
                try:
                    p = mesh.vertices[int(v2)].position
                except Exception:
                    continue
                r_vals.append(float(np.linalg.norm((p - center)[:2])))

        has_inner = any(r < R - tol for r in r_vals) if R > 0.0 else False
        has_outer = any(r > R + tol for r in r_vals) if R > 0.0 else False

        # Must include at least one disk-side preset and one non-disk preset.
        # We treat any preset starting with "disk" as disk-side (e.g. "disk_edge").
        has_disk = any(p.startswith("disk") for p in presets if p)
        has_other = any(p and not p.startswith("disk") for p in presets)

        # Accept either explicit tag straddling (has_disk & has_other) or geometric
        # straddling (has_inner & has_outer). The latter keeps the validator stable
        # under refinement where mid-edge vertices may lack presets.
        if not ((has_disk and has_other) or (has_inner and has_outer)):
            issues.append(
                DiskInterfaceIssue(
                    vertex_id=int(vid),
                    incident_presets=tuple(sorted(presets)),
                )
            )

    if issues:
        examples = issues[:5]
        msg = (
            "Disk interface topology invalid: rim_slope_match_disk_group is set, "
            "but the tagged disk boundary vertices do not straddle disk↔membrane "
            "triangles. This usually means the 'disk boundary ring' is an internal "
            "ring inside the disk patch (not the disk↔membrane interface). "
            "The validator expects incident triangles to include both disk-side "
            "presets (prefix 'disk') and non-disk presets (e.g. 'rim'). "
            f"bad_vertices={len(issues)} examples={examples}"
        )
        raise ValueError(msg)


__all__ = ["validate_disk_interface_topology", "DiskInterfaceIssue"]
