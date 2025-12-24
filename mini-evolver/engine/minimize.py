"""Minimization routines and volume constraint handling."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from . import numeric
from .geometry import Vector, v_scale, v_sub
from .mesh import Mesh


def flatten_positions(positions: Dict[int, Vector], order: List[int]) -> List[float]:
    flat: List[float] = []
    for vid in order:
        x, y, z = positions[vid]
        flat.extend([x, y, z])
    return flat


def unflatten_positions(flat: Sequence[float], order: List[int]) -> Dict[int, Vector]:
    positions: Dict[int, Vector] = {}
    for i, vid in enumerate(order):
        idx = 3 * i
        positions[vid] = (flat[idx], flat[idx + 1], flat[idx + 2])
    return positions


def energy_grad_vector(
    mesh: Mesh, positions: Dict[int, Vector], order: List[int], penalty: float
) -> Tuple[float, float, float, List[float], float]:
    energy, area, volume, grads, grad_norm = mesh.energy(positions, penalty)
    grad_vec: List[float] = []
    for vid in order:
        gx, gy, gz = grads.get(vid, (0.0, 0.0, 0.0))
        grad_vec.extend([gx, gy, gz])
    return energy, area, volume, grad_vec, grad_norm


def compute_grads(
    mesh: Mesh, positions: Dict[int, Vector], penalty: float, volume_mode: str
) -> Tuple[float, Dict[int, Vector], float]:
    if volume_mode == "penalty":
        energy, _, _, grads, grad_norm = mesh.energy(positions, penalty)
        return energy, grads, grad_norm
    energy, _, _, grads, _ = mesh.energy(positions, 0.0)
    vol_grads_list: List[Dict[int, Vector]] = []
    for body in mesh.bodies:
        _, body_grads = mesh.body_volume_and_grads(body, positions)
        vol_grads_list.append(body_grads)
    lam = solve_lambda_system(mesh, grads, vol_grads_list)
    for vid in grads:
        if vid in mesh.fixed_ids:
            continue
        total = grads.get(vid, (0.0, 0.0, 0.0))
        for idx, gv_map in enumerate(vol_grads_list):
            gv = gv_map.get(vid, (0.0, 0.0, 0.0))
            if vid in mesh.constraint_vertices:
                axis = mesh.constraint_axes.get(mesh.vertices[vid].constraint, 2)
                if axis == 0:
                    gv = (0.0, gv[1], gv[2])
                elif axis == 1:
                    gv = (gv[0], 0.0, gv[2])
                else:
                    gv = (gv[0], gv[1], 0.0)
            total = (
                total[0] + lam[idx] * gv[0],
                total[1] + lam[idx] * gv[1],
                total[2] + lam[idx] * gv[2],
            )
        grads[vid] = total
    grad_norm = math.sqrt(
        sum(g[0] * g[0] + g[1] * g[1] + g[2] * g[2] for g in grads.values())
    )
    return energy, grads, grad_norm


def solve_lambda_system(
    mesh: Mesh,
    grads_a: Dict[int, Vector],
    vol_grads_list: List[Dict[int, Vector]],
    step: float | None = None,
    volume_errors: List[float] | None = None,
) -> List[float]:
    n = len(vol_grads_list)
    if n == 0:
        return []
    gmat = [[0.0 for _ in range(n)] for _ in range(n)]
    bvec = [0.0 for _ in range(n)]
    for i in range(n):
        for vid, gvi in vol_grads_list[i].items():
            if vid in mesh.fixed_ids:
                continue
            if vid in mesh.constraint_vertices:
                axis = mesh.constraint_axes.get(mesh.vertices[vid].constraint, 2)
                if axis == 0:
                    gvi = (0.0, gvi[1], gvi[2])
                elif axis == 1:
                    gvi = (gvi[0], 0.0, gvi[2])
                else:
                    gvi = (gvi[0], gvi[1], 0.0)
            ga = grads_a.get(vid, (0.0, 0.0, 0.0))
            bvec[i] += gvi[0] * ga[0] + gvi[1] * ga[1] + gvi[2] * ga[2]
            for j in range(n):
                gvj = vol_grads_list[j].get(vid, (0.0, 0.0, 0.0))
                if vid in mesh.constraint_vertices:
                    axis = mesh.constraint_axes.get(mesh.vertices[vid].constraint, 2)
                    if axis == 0:
                        gvj = (0.0, gvj[1], gvj[2])
                    elif axis == 1:
                        gvj = (gvj[0], 0.0, gvj[2])
                    else:
                        gvj = (gvj[0], gvj[1], 0.0)
                gmat[i][j] += gvi[0] * gvj[0] + gvi[1] * gvj[1] + gvi[2] * gvj[2]
    if step is not None and volume_errors is not None:
        for i in range(n):
            bvec[i] = (volume_errors[i] / step) - bvec[i]
    else:
        bvec = [-v for v in bvec]
    return solve_linear_system(gmat, bvec)


def solve_linear_system(mat: List[List[float]], vec: List[float]) -> List[float]:
    n = len(vec)
    if n == 1:
        return [vec[0] / mat[0][0]] if abs(mat[0][0]) > 1e-15 else [0.0]
    a = [row[:] + [vec[i]] for i, row in enumerate(mat)]
    for i in range(n):
        pivot = a[i][i]
        if abs(pivot) < 1e-15:
            return [0.0 for _ in range(n)]
        inv = 1.0 / pivot
        for j in range(i, n + 1):
            a[i][j] *= inv
        for k in range(n):
            if k == i:
                continue
            factor = a[k][i]
            for j in range(i, n + 1):
                a[k][j] -= factor * a[i][j]
    return [a[i][n] for i in range(n)]


def enforce_volume_constraint(
    mesh: Mesh, positions: Dict[int, Vector], tol: float = 1e-9
) -> Tuple[Dict[int, Vector], float]:
    current = positions
    for _ in range(5):
        for body in mesh.bodies:
            target = body.target_volume
            if target == 0.0:
                continue
            volume, vol_grads = mesh.body_volume_and_grads(body, current)
            error = volume - target
            if abs(error) < tol:
                continue
            denom = sum(
                (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]) for g in vol_grads.values()
            )
            if denom < 1e-15:
                continue
            alpha = error / denom
            corrected = {}
            for vid, pos in current.items():
                moved = v_sub(pos, v_scale(vol_grads.get(vid, (0.0, 0.0, 0.0)), alpha))
                corrected[vid] = mesh.project_vertex(vid, moved)
            current = corrected
    final_volume, _ = mesh.signed_volume_and_grads(current)
    return current, final_volume


def backtracking_step(
    mesh: Mesh,
    positions: Dict[int, Vector],
    grads: Dict[int, Vector],
    current_energy: float,
    penalty: float,
    enforce_volume: bool,
    volume_mode: str,
    start_step: float = 0.25,
    shrink: float = 0.5,
    max_trials: int = 12,
) -> Tuple[Dict[int, Vector], float, float, float, Dict[int, Vector], float]:
    step = start_step
    fixed = mesh.fixed_ids
    for _ in range(max_trials):
        trial_positions = {}
        for vid, pos in positions.items():
            if vid in fixed:
                trial_positions[vid] = pos
            else:
                updated = v_sub(pos, v_scale(grads[vid], step))
                trial_positions[vid] = mesh.project_vertex(vid, updated)
        if enforce_volume:
            trial_positions, _ = enforce_volume_constraint(mesh, trial_positions)
        energy_penalty = penalty if volume_mode == "penalty" else 0.0
        energy, area, volume, trial_grads, grad_norm = mesh.energy(
            trial_positions, energy_penalty
        )
        if energy < current_energy:
            return trial_positions, energy, area, volume, trial_grads, grad_norm
        step *= shrink
    return positions, current_energy, None, None, grads, None  # type: ignore


def saddle_step(
    mesh: Mesh,
    positions: Dict[int, Vector],
    penalty: float,
    enforce_volume: bool,
    step_scale: float = 1.0,
    shrink: float = 0.5,
    max_trials: int = 12,
) -> Tuple[Dict[int, Vector], float, Dict[int, Vector], float]:
    energy, _, _, grads_a, _ = mesh.energy(positions, 0.0)
    volume_errors: List[float] = []
    vol_grads_list: List[Dict[int, Vector]] = []
    for body in mesh.bodies:
        vol, body_grads = mesh.body_volume_and_grads(body, positions)
        volume_errors.append(vol - body.target_volume)
        vol_grads_list.append(body_grads)
    step = 0.25 * step_scale
    for _ in range(max_trials):
        lam = solve_lambda_system(mesh, grads_a, vol_grads_list, step, volume_errors)
        trial_positions = {}
        for vid, pos in positions.items():
            if vid in mesh.fixed_ids:
                trial_positions[vid] = pos
                continue
            ga = grads_a.get(vid, (0.0, 0.0, 0.0))
            step_grad = [ga[0], ga[1], ga[2]]
            for idx, gv_map in enumerate(vol_grads_list):
                gv = gv_map.get(vid, (0.0, 0.0, 0.0))
                step_grad[0] += lam[idx] * gv[0]
                step_grad[1] += lam[idx] * gv[1]
                step_grad[2] += lam[idx] * gv[2]
            step_grad_t = (step_grad[0], step_grad[1], step_grad[2])
            updated = v_sub(pos, v_scale(step_grad_t, step))
            trial_positions[vid] = mesh.project_vertex(vid, updated)
        if enforce_volume:
            trial_positions, _ = enforce_volume_constraint(mesh, trial_positions)
        trial_energy, _, _, _, _ = mesh.energy(trial_positions, 0.0)
        if trial_energy < energy:
            energy, grads, grad_norm = compute_grads(
                mesh, trial_positions, penalty, "saddle"
            )
            return trial_positions, energy, grads, grad_norm
        step *= shrink
    energy, grads, grad_norm = compute_grads(mesh, positions, penalty, "saddle")
    return positions, energy, grads, grad_norm


def gradient_steps(
    mesh: Mesh,
    positions: Dict[int, Vector],
    penalty: float,
    steps: int,
    enforce_volume: bool,
    step_scale: float = 1.0,
    volume_mode: str = "penalty",
) -> Dict[int, Vector]:
    energy, grads, grad_norm = compute_grads(mesh, positions, penalty, volume_mode)
    for _ in range(steps):
        if grad_norm < 1e-10:
            break
        if volume_mode == "saddle":
            positions, energy, grads, grad_norm = saddle_step(
                mesh,
                positions,
                penalty,
                enforce_volume=enforce_volume,
                step_scale=step_scale,
            )
        else:
            (
                positions,
                energy,
                _,
                _,
                grads,
                grad_norm,
            ) = backtracking_step(
                mesh,
                positions,
                grads,
                energy,
                penalty,
                enforce_volume=enforce_volume,
                volume_mode=volume_mode,
                start_step=0.25 * step_scale,
            )
        if grad_norm is None:
            break
    return positions


def hessian_bfgs(
    mesh: Mesh,
    positions: Dict[int, Vector],
    penalty: float,
    steps: int = 5,
    volume_mode: str = "penalty",
    enforce_volume: bool = False,
) -> Dict[int, Vector]:
    order = sorted(positions)
    x = flatten_positions(positions, order)
    n = len(x)
    if numeric.USE_NUMPY:
        x_np = numeric.np.array(x, dtype=float)
        h = numeric.np.eye(n, dtype=float)

        def to_positions(vec: "numeric.np.ndarray") -> Dict[int, Vector]:
            arr = vec.reshape((-1, 3))
            return {
                vid: (arr[i, 0], arr[i, 1], arr[i, 2]) for i, vid in enumerate(order)
            }

        for _ in range(steps):
            positions = to_positions(x_np)
            energy_penalty = penalty if volume_mode == "penalty" else 0.0
            energy, _, _, g_list, grad_norm = energy_grad_vector(
                mesh, positions, order, energy_penalty
            )
            if grad_norm < 1e-10:
                break
            g = numeric.np.array(g_list, dtype=float)
            p = -h.dot(g)
            x_old = x_np.copy()
            step = 1.0
            for _ in range(10):
                x_trial = x_old + step * p
                trial_positions = to_positions(x_trial)
                if volume_mode == "saddle" and enforce_volume:
                    trial_positions, _ = enforce_volume_constraint(
                        mesh, trial_positions
                    )
                    x_trial = numeric.np.array(
                        flatten_positions(trial_positions, order)
                    )
                trial_energy, _, _, _, _ = energy_grad_vector(
                    mesh, trial_positions, order, energy_penalty
                )
                if trial_energy < energy:
                    x_np = x_trial
                    break
                step *= 0.5
            positions = to_positions(x_np)
            if volume_mode == "saddle" and enforce_volume:
                positions, _ = enforce_volume_constraint(mesh, positions)
                x_np = numeric.np.array(flatten_positions(positions, order))
            _, _, _, g_new_list, _ = energy_grad_vector(
                mesh, positions, order, energy_penalty
            )
            g_new = numeric.np.array(g_new_list, dtype=float)
            s = x_np - x_old
            y = g_new - g
            ys = float(y.dot(s))
            if ys <= 1e-12:
                continue
            rho = 1.0 / ys
            i_mat = numeric.np.eye(n, dtype=float)
            h = (i_mat - rho * numeric.np.outer(s, y)) @ h @ (
                i_mat - rho * numeric.np.outer(y, s)
            ) + rho * numeric.np.outer(s, s)
        return to_positions(x_np)

    if n > 600:
        steps = min(steps, 1)
    h = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        h[i][i] = 1.0

    def mat_vec(mat: List[List[float]], vec: List[float]) -> List[float]:
        return [sum(row[j] * vec[j] for j in range(n)) for row in mat]

    def dot(a: List[float], b: List[float]) -> float:
        return sum(a[i] * b[i] for i in range(n))

    for _ in range(steps):
        energy_penalty = penalty if volume_mode == "penalty" else 0.0
        energy, _, _, g_list, grad_norm = energy_grad_vector(
            mesh, positions, order, energy_penalty
        )
        if grad_norm < 1e-10:
            break
        p = [-val for val in mat_vec(h, g_list)]
        x_old = x[:]
        step = 1.0
        for _ in range(10):
            x_trial = [x_old[i] + step * p[i] for i in range(n)]
            trial_positions = unflatten_positions(x_trial, order)
            if volume_mode == "saddle" and enforce_volume:
                trial_positions, _ = enforce_volume_constraint(mesh, trial_positions)
                x_trial = flatten_positions(trial_positions, order)
            trial_energy, _, _, _, _ = energy_grad_vector(
                mesh, trial_positions, order, energy_penalty
            )
            if trial_energy < energy:
                x = x_trial
                break
            step *= 0.5
        positions = unflatten_positions(x, order)
        if volume_mode == "saddle" and enforce_volume:
            positions, _ = enforce_volume_constraint(mesh, positions)
            x = flatten_positions(positions, order)
        _, _, _, g_new_list, _ = energy_grad_vector(
            mesh, positions, order, energy_penalty
        )
        s = [x[i] - x_old[i] for i in range(n)]
        y = [g_new_list[i] - g_list[i] for i in range(n)]
        ys = dot(y, s)
        if ys <= 1e-12:
            continue
        rho = 1.0 / ys
        hy = mat_vec(h, y)
        yhy = dot(y, hy)
        for i in range(n):
            for j in range(n):
                h[i][j] += (
                    (1 + yhy * rho) * rho * s[i] * s[j]
                    - rho * s[i] * hy[j]
                    - rho * hy[i] * s[j]
                )
    return unflatten_positions(x, order)


def vertex_average(mesh: Mesh, positions: Dict[int, Vector]) -> Dict[int, Vector]:
    new_positions: Dict[int, Vector] = {}
    for vid, pos in positions.items():
        if vid in mesh.fixed_ids:
            new_positions[vid] = pos
            continue
        neighbors = mesh.adjacency.get(vid, [])
        if not neighbors:
            new_positions[vid] = pos
            continue
        sx = sy = sz = 0.0
        for nb in neighbors:
            px, py, pz = positions[nb]
            sx += px
            sy += py
            sz += pz
        inv = 1.0 / len(neighbors)
        avg = (sx * inv, sy * inv, sz * inv)
        new_positions[vid] = mesh.project_vertex(vid, avg)
    return new_positions


def minimize(
    mesh: Mesh,
    penalty: float = 50.0,
    max_steps: int = 400,
    grad_tol: float = 1e-6,
    enforce_volume: bool = False,
    volume_mode: str = "penalty",
) -> Tuple[Dict[int, Vector], List[Tuple[int, float, float, float, float]]]:
    positions = mesh.current_positions()
    energy_penalty = penalty if volume_mode == "penalty" else 0.0
    energy, area, volume, grads, grad_norm = mesh.energy(positions, energy_penalty)
    history: List[Tuple[int, float, float, float, float]] = [
        (0, energy, area, volume, grad_norm)
    ]
    for step in range(1, max_steps + 1):
        if grad_norm < grad_tol:
            break
        positions = gradient_steps(
            mesh,
            positions,
            penalty=penalty,
            steps=1,
            enforce_volume=enforce_volume,
            volume_mode=volume_mode,
        )
        energy, area, volume, grads, grad_norm = mesh.energy(positions, energy_penalty)
        history.append((step, energy, area, volume, grad_norm))
    return positions, history
