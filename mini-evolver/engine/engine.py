"""Minimization runner and reporting."""

from __future__ import annotations

from .commands import run_gogo
from .geometry import Vector
from .mesh import Mesh
from .minimize import minimize
from .plot import LivePlotter, visualize
from .refine import refine_mesh


def format_vec(v: Vector) -> str:
    return f"({v[0]: .6f}, {v[1]: .6f}, {v[2]: .6f})"


def run(
    mesh: Mesh,
    penalty: float,
    max_steps: int,
    enforce_volume: bool,
    plot: bool = False,
    save_plot: str | None = None,
    refine_steps: int = 0,
    use_gogo: bool = True,
    hessian_steps: int = 5,
    macro_name: str = "gogo",
    volume_mode: str = "penalty",
    live_plot: bool = False,
) -> None:
    if volume_mode == "saddle":
        enforce_volume = True
    plotter = LivePlotter(mesh) if live_plot else None
    on_step = plotter.update if plotter else None
    if plotter:
        plotter.update(mesh.current_positions(), 0)
    macro_key = macro_name.lower()
    if use_gogo and macro_key in mesh.macros:
        mesh, positions, step_count = run_gogo(
            mesh,
            penalty=penalty,
            enforce_volume=enforce_volume,
            hessian_steps=hessian_steps,
            macro_name=macro_key,
            volume_mode=volume_mode,
            on_step=on_step,
        )
        energy_penalty = penalty if volume_mode == "penalty" else 0.0
        energy, area, volume, _, grad_norm = mesh.energy(positions, energy_penalty)
        last = (step_count, energy, area, volume, grad_norm)
    else:
        for _ in range(refine_steps):
            mesh = refine_mesh(mesh)
        positions, history = minimize(
            mesh,
            penalty=penalty,
            max_steps=max_steps,
            enforce_volume=enforce_volume,
            volume_mode=volume_mode,
        )
        last = history[-1]
    print("Evolver-style minimization")
    print(f"  steps: {last[0]}  energy: {last[1]:.6f}  area: {last[2]:.6f}")
    if len(mesh.bodies) > 1:
        parts = []
        for body in mesh.bodies:
            vol, _ = mesh.body_volume_and_grads(body, positions)
            parts.append(f"body{body.bid}:{vol:.6f}/{body.target_volume}")
        volume_info = "  volume: " + " ".join(parts)
    else:
        volume_info = f"  volume: {last[3]:.6f} (target {mesh.body.target_volume})"
    print(f"{volume_info}  grad_norm: {last[4]:.3e}")
    print("\nFinal vertex coordinates:")
    for vid in sorted(positions):
        print(f"  v{vid}: {format_vec(positions[vid])}")
    if plot or save_plot:
        visualize(mesh, positions, save_path=save_plot, show=plot)
    if plotter:
        plotter.close()
