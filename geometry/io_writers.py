import json
import logging

import numpy as np
import yaml

from geometry.entities import Mesh, Vertex

logger = logging.getLogger("membrane_solver")


def save_geometry(
    mesh: Mesh,
    path: str = "outputs/temp_output_file.json",
    *,
    compact: bool = False,
):
    def _to_builtin(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {k: _to_builtin(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_to_builtin(v) for v in value]
        return value

    def _sorted_keys(dct):
        return sorted(dct.keys())

    # IMPORTANT: The on-disk format encodes indices implicitly by list order:
    # - vertices are 0..N-1 by position in "vertices" list
    # - edges are 1..E by position in "edges" list (with 0-based references in faces)
    # - faces are 0..F-1 by position in "faces" list
    #
    # In-memory meshes can have sparse/non-contiguous IDs after refinement or
    # equiangulation (e.g. deleting edge 10 and creating edge 500). When saving,
    # reindex all entities to a compact, contiguous numbering so that a
    # save→load roundtrip produces a valid mesh.
    vertex_ids = _sorted_keys(mesh.vertices)
    vertex_id_map = {old: new for new, old in enumerate(vertex_ids)}

    edge_ids = _sorted_keys(mesh.edges)
    edge_id_map = {old: new + 1 for new, old in enumerate(edge_ids)}  # 1-based

    facet_ids = _sorted_keys(mesh.facets)
    facet_id_map = {old: new for new, old in enumerate(facet_ids)}  # 0-based
    pos_view = mesh.positions_view()
    index_map = mesh.vertex_index_to_row

    def export_edge_index(old_signed_edge_index: int):
        sign = -1 if old_signed_edge_index < 0 else 1
        old_abs = abs(int(old_signed_edge_index))
        if old_abs not in edge_id_map:
            raise KeyError(
                f"Cannot save geometry: facet references missing edge {old_signed_edge_index}."
            )
        new_abs = edge_id_map[old_abs]
        new_signed = sign * new_abs
        if new_signed < 0:
            return f"r{new_abs - 1}"  # -1 → "r0"
        return new_abs - 1  # 1 → 0

    def prepare_options(entity):
        opts = entity.options.copy() if entity.options else {}
        if entity.fixed:
            opts["fixed"] = True
        if isinstance(entity, Vertex):
            if hasattr(entity, "tilt") and np.any(entity.tilt):
                opts["tilt"] = entity.tilt.tolist()
            if hasattr(entity, "tilt_fixed") and entity.tilt_fixed:
                opts["tilt_fixed"] = True
            if hasattr(entity, "tilt_in") and np.any(entity.tilt_in):
                opts["tilt_in"] = entity.tilt_in.tolist()
            if hasattr(entity, "tilt_out") and np.any(entity.tilt_out):
                opts["tilt_out"] = entity.tilt_out.tolist()
            if hasattr(entity, "tilt_fixed_in") and entity.tilt_fixed_in:
                opts["tilt_fixed_in"] = True
            if hasattr(entity, "tilt_fixed_out") and entity.tilt_fixed_out:
                opts["tilt_fixed_out"] = True
        return opts if opts else None

    data = {
        "vertices": [
            (
                [
                    *(
                        pos_view[index_map[int(old_vid)]]
                        if int(old_vid) in index_map
                        else mesh.vertices[old_vid].position
                    ).tolist(),
                    prepare_options(mesh.vertices[old_vid]),
                ]
                if prepare_options(mesh.vertices[old_vid])
                else (
                    pos_view[index_map[int(old_vid)]]
                    if int(old_vid) in index_map
                    else mesh.vertices[old_vid].position
                ).tolist()
            )
            for old_vid in vertex_ids
        ],
        "edges": [
            (
                [
                    vertex_id_map[int(mesh.edges[old_eid].tail_index)],
                    vertex_id_map[int(mesh.edges[old_eid].head_index)],
                    prepare_options(mesh.edges[old_eid]),
                ]
                if prepare_options(mesh.edges[old_eid])
                else [
                    vertex_id_map[int(mesh.edges[old_eid].tail_index)],
                    vertex_id_map[int(mesh.edges[old_eid].head_index)],
                ]
            )
            for old_eid in edge_ids
        ],
        "faces": [
            (
                [
                    *map(
                        export_edge_index,
                        mesh.facets[old_fid].edge_indices,
                    ),
                    prepare_options(mesh.facets[old_fid]),
                ]
                if prepare_options(mesh.facets[old_fid])
                else list(map(export_edge_index, mesh.facets[old_fid].edge_indices))
            )
            for old_fid in facet_ids
        ],
        "bodies": {
            "faces": [
                [facet_id_map[int(fid)] for fid in mesh.bodies[b].facet_indices]
                for b in mesh.bodies.keys()
            ],
            "target_volume": [mesh.bodies[b].target_volume for b in mesh.bodies.keys()],
            "target_area": [
                mesh.bodies[b].options.get("target_area") for b in mesh.bodies.keys()
            ],
            "energy": [
                mesh.bodies[b].options.get("energy", {}) for b in mesh.bodies.keys()
            ],
            "constraints": [
                mesh.bodies[b].options.get("constraints", [])
                for b in mesh.bodies.keys()
            ],
        },
        "energy_modules": list(mesh.energy_modules),
        "constraint_modules": list(mesh.constraint_modules),
        "global_parameters": mesh.global_parameters.to_dict(),
        "instructions": mesh.instructions,
    }
    definitions = (
        mesh.definitions.copy()
        if isinstance(getattr(mesh, "definitions", None), dict)
        else {}
    )
    used_presets: set[str] = set()
    for vertex in mesh.vertices.values():
        preset = (vertex.options or {}).get("preset")
        if preset:
            used_presets.add(str(preset))
    for edge in mesh.edges.values():
        preset = (edge.options or {}).get("preset")
        if preset:
            used_presets.add(str(preset))
    for facet in mesh.facets.values():
        preset = (facet.options or {}).get("preset")
        if preset:
            used_presets.add(str(preset))
    if used_presets:
        for preset in used_presets:
            definitions.setdefault(preset, {})
        data["definitions"] = definitions
    elif definitions:
        data["definitions"] = definitions

    filename_str = str(path)
    with open(path, "w") as f:
        if filename_str.endswith((".yaml", ".yml")):
            yaml.safe_dump(
                _to_builtin(data),
                f,
                default_flow_style=bool(compact),
                sort_keys=False,
                allow_unicode=True,
            )
        else:
            if compact:
                json.dump(
                    _to_builtin(data), f, separators=(",", ":"), ensure_ascii=False
                )
            else:
                json.dump(_to_builtin(data), f, indent=4, ensure_ascii=False)
