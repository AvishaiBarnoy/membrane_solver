"""Validation helpers for mesh interface/topology assumptions."""

from __future__ import annotations

from dataclasses import dataclass

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

    def preset_of(vid: int) -> str:
        opts = getattr(mesh.vertices[vid], "options", None) or {}
        return str(opts.get("preset") or "")

    issues: list[DiskInterfaceIssue] = []
    for vid in disk_boundary_vids:
        incident = mesh.vertex_to_facets.get(int(vid)) or set()
        presets: set[str] = set()
        for fid in incident:
            loop = mesh.facet_vertex_loops.get(int(fid))
            if loop is None or len(loop) != 3:
                continue
            for v2 in loop:
                presets.add(preset_of(int(v2)))

        # Must include at least one disk and one non-disk preset.
        has_disk = "disk" in presets
        has_other = any(p != "disk" and p != "" for p in presets)
        if not (has_disk and has_other):
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
            f"bad_vertices={len(issues)} examples={examples}"
        )
        raise ValueError(msg)


__all__ = ["validate_disk_interface_topology", "DiskInterfaceIssue"]
