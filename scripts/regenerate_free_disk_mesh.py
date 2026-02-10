import math
from pathlib import Path
from typing import List, Tuple

from ruamel.yaml import YAML

BASE = "meshes/caveolin/kozlov_1disk_3d_tensionless_single_leaflet_profile_hard_rim_R12_free_disk.yaml"


def _ring_vertices(r: float, n: int) -> List[Tuple[float, float, float]]:
    return [
        (r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), 0.0)
        for i in range(n)
    ]


def _add_faces_between(ring_a: List[int], ring_b: List[int]) -> List[List[int]]:
    faces = []
    n = len(ring_a)
    for i in range(n):
        a0 = ring_a[i]
        a1 = ring_a[(i + 1) % n]
        b0 = ring_b[i]
        b1 = ring_b[(i + 1) % n]
        faces.append([a0, b0, b1])
        faces.append([a0, b1, a1])
    return faces


def _add_faces_center(center_idx: int, ring: List[int]) -> List[List[int]]:
    faces = []
    n = len(ring)
    for i in range(n):
        faces.append([center_idx, ring[i], ring[(i + 1) % n]])
    return faces


def _edges_and_face_refs(
    faces: List[List[int]],
) -> Tuple[List[List[int]], List[List[int | str]]]:
    edge_index = {}
    edges: List[List[int]] = []
    face_refs: List[List[int | str]] = []
    for a, b, c in faces:
        refs = []
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v)
            rev = (v, u)
            if key in edge_index:
                idx = edge_index[key]
                refs.append(idx)
                continue
            if rev in edge_index:
                idx = edge_index[rev]
                refs.append(f"r{idx}")
                continue
            idx = len(edges)
            edge_index[key] = idx
            edges.append([u, v])
            refs.append(idx)
        face_refs.append(refs)
    return edges, face_refs


def main() -> None:
    yaml = YAML()
    yaml.preserve_quotes = True
    path = Path(BASE)
    data = yaml.load(path.read_text())

    definitions = data.get("definitions", {})
    disk_def = definitions.get("disk", {}) if isinstance(definitions, dict) else {}
    R = float(disk_def.get("pin_to_circle_radius", 0.466666666666667))
    R_outer = 12.0

    n_theta = 24
    n_mem = 10
    p = 2.0

    # Disk rings: center, R/2, R
    center = (0.0, 0.0, 0.0)
    disk_inner_r = R * 0.5

    radii_mem = [R + (R_outer - R) * ((i / n_mem) ** p) for i in range(1, n_mem + 1)]

    vertices = []
    ring_indices: List[List[int]] = []

    # Center vertex
    center_idx = len(vertices)
    vertices.append(
        [
            center[0],
            center[1],
            center[2],
            {
                "preset": "disk",
                "rigid_disk_group": "disk",
                "tilt_fixed_in": True,
                "tilt_fixed_out": True,
                "tilt_in": [0.0, 0.0],
                "tilt_out": [0.0, 0.0],
            },
        ]
    )

    # Disk inner ring
    ring = []
    for x, y, z in _ring_vertices(disk_inner_r, n_theta):
        ring.append(len(vertices))
        vertices.append([x, y, z, {"preset": "disk", "rigid_disk_group": "disk"}])
    ring_indices.append(ring)

    # Disk boundary ring (r=R) doubles as rim-matching ring.
    ring = []
    for x, y, z in _ring_vertices(R, n_theta):
        ring.append(len(vertices))
        vertices.append(
            [
                x,
                y,
                z,
                {
                    "preset": "disk",
                    "rim_slope_match_group": "rim",
                    "rigid_disk_group": "disk",
                },
            ]
        )
    ring_indices.append(ring)

    # Membrane rings
    for idx, r in enumerate(radii_mem, start=1):
        ring = []
        for x, y, z in _ring_vertices(r, n_theta):
            opts = {}
            if idx == len(radii_mem):
                opts["preset"] = "outer_rim"
                opts["rim_slope_match_group"] = "outer"
                opts["constraints"] = ["pin_to_plane", "pin_to_circle"]
                opts["pin_to_circle_group"] = "outer"
                opts["pin_to_circle_radius"] = R_outer
                opts["pin_to_circle_normal"] = [0.0, 0.0, 1.0]
                opts["pin_to_circle_point"] = [0.0, 0.0, 0.0]
                opts["pin_to_circle_mode"] = "fixed"
            ring.append(len(vertices))
            vertices.append([x, y, z, opts] if opts else [x, y, z])
        ring_indices.append(ring)

    # Faces
    faces = []
    faces.extend(_add_faces_center(center_idx, ring_indices[0]))
    for a, b in zip(ring_indices, ring_indices[1:]):
        faces.extend(_add_faces_between(a, b))

    edges, face_refs = _edges_and_face_refs(faces)

    data["vertices"] = vertices
    data["edges"] = edges
    data["faces"] = face_refs

    yaml.dump(data, path.open("w"))


if __name__ == "__main__":
    main()
