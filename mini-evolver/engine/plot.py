"""Visualization utilities."""

from __future__ import annotations

from typing import Dict

from .geometry import Vector
from .mesh import Mesh


def visualize(
    mesh: Mesh, positions: Dict[int, Vector], save_path: str | None, show: bool
) -> None:
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except Exception as exc:
        print(f"Matplotlib unavailable: {exc}")
        return
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    polys = [
        [positions[vid] for vid in face.vertex_loop] for face in mesh.faces.values()
    ]
    poly = Poly3DCollection(polys, alpha=0.4, facecolor="#88c0d0", edgecolor="#334155")
    ax.add_collection3d(poly)
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    zs = [p[2] for p in positions.values()]
    margin = 0.2
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.set_zlim(min(zs) - margin, max(zs) + margin)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Minimized surface")
    if save_path:
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


class LivePlotter:
    def __init__(self, mesh: Mesh) -> None:
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        except Exception as exc:
            print(f"Live visualization unavailable: {exc}")
            self.enabled = False
            return
        self.enabled = True
        self.plt = plt
        self.Poly3DCollection = Poly3DCollection
        self.fig = plt.figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.mesh = mesh
        self.poly = None
        plt.ion()

    def update(self, positions: Dict[int, Vector], step: int) -> None:
        if not self.enabled:
            return
        polys = [
            [positions[vid] for vid in face.vertex_loop]
            for face in self.mesh.faces.values()
        ]
        if self.poly is None:
            self.poly = self.Poly3DCollection(
                polys, alpha=0.4, facecolor="#88c0d0", edgecolor="#334155"
            )
            self.ax.add_collection3d(self.poly)
        else:
            self.poly.set_verts(polys)
        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        zs = [p[2] for p in positions.values()]
        margin = 0.2
        self.ax.set_xlim(min(xs) - margin, max(xs) + margin)
        self.ax.set_ylim(min(ys) - margin, max(ys) + margin)
        self.ax.set_zlim(min(zs) - margin, max(zs) + margin)
        self.ax.set_title(f"Step {step}")
        self.plt.pause(0.001)

    def close(self) -> None:
        if not self.enabled:
            return
        self.plt.ioff()
        self.plt.show(block=False)
