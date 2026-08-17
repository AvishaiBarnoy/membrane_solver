# Energy and Constraint API Contract

Status: **complete — dispatch inventory and capability boundary**
Owners: `runtime/evaluation_manager.py`, `runtime/energy_manager.py`, and
`runtime/constraint_manager.py`

## Current boundary

Numerical optimization uses dense arrays. `EvaluationManager` dispatches to
`compute_energy_and_gradient_array` when a module provides it, while retaining
legacy dict-based adaptation at explicit compatibility boundaries. Constraints
remain responsible for projection/enforcement and may expose separate joint or
tilt-gradient behavior.

## Protected protocol

| Capability | Required contract |
|---|---|
| Dense energy/shape gradient | return a scalar energy and dense gradient aligned to the active vertex-row map |
| Legacy adapter | only used where dispatch explicitly permits it; minimization-loop callers do not silently fall back |
| Tilt/leaflet fields | preserve field shape, fixed-mask, and absence semantics independently of shape gradients |
| Constraints | projection/enforcement order and joint-gradient behavior stay outside energy-module dispatch |
| Theory lanes | lane selection remains explicit policy, never inferred from API capability |

## Completion boundary

`runtime.module_capabilities` now owns all current `EvaluationManager`
capability checks and the legacy misaligned-module-name fallback. Module
migration remains out of scope: array/dict compatibility paths, constraints,
and numerical module implementations are unchanged.
