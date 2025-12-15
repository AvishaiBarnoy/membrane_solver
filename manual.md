# Membrane Solver Manual

This document explains how to run the membrane solver, how to work with input
meshes, and which physical energies and constraints are currently implemented.
It is intended to stay in sync with the `main` branch: new feature branches
should extend this manual before they are merged.

---

## 1. Installation

- Python: 3.10 or newer.
- Install dependencies from the repository root:

  ```bash
  pip install -r requirements.txt
  ```

- Tests (recommended before/after larger changes):

  ```bash
  pytest -q
  ```

### 1.1 Development tooling (optional)

- Lint:

  ```bash
  pip install ruff
  ruff check .
  ```

- Pre-commit hooks (recommended):

  ```bash
  pip install pre-commit
  pre-commit install
  pre-commit run -a
  ```

---

## 2. Basic usage

Run the main driver from the repository root:

```bash
python main.py -i meshes/cube_good_min_routine.json -o output.json
```

Key command‑line options:

- `-i, --input PATH`
  Input mesh JSON file. If the `.json` suffix is omitted, it will be added.

- `-o, --output PATH`
  Output mesh JSON file. If omitted, the final geometry is **not** saved.

- `--instructions PATH`
  Optional instruction file; each token or whitespace‑separated word is an
  interactive command (e.g. `g100`, `r`, `u`, …). These commands are executed
  before interactive mode starts.

- `--properties` / `-p` / `-i`
  Compute and print basic physical properties (total area, volume, per‑body
  area/volume) and exit without minimization.

- `--volume-mode {lagrange,penalty}`
  Override the global `volume_constraint_mode`.
  - `lagrange` – treat volume as a hard constraint (default). Best paired with
    `--volume-projection false` to avoid redundant geometric projections.
  - `penalty` – add a quadratic volume energy term (soft constraint). Works
    best with `--volume-projection true`.

- `--volume-projection {true,false}`
  Control geometric projection during minimization.
  - `false` is recommended when using the hard `lagrange` constraint (prevents
    double enforcement in the line search).
  - `true` is the historical behaviour and remains the default in penalty mode.

- `--log PATH`
  Log file path (default: `membrane_solver.log`, overwritten each run).

- `-q, --quiet`
  Suppress per‑step console output.

- `--debug`
  Enable verbose DEBUG logging (line‑search diagnostics, constraint details,
  etc.). For normal runs leave this off.

- `--non-interactive`
  Do not enter the interactive prompt after executing the instruction list.

Example: run a scripted minimization, then exit without interactive mode:

```bash
python main.py -i meshes/cube_good_min_routine.json \
               --non-interactive
```

---

## 3. Interactive mode

By default, after any initial instructions are executed the solver enters an
interactive loop:

```text
=== Membrane Solver ===
Input file: meshes/cube_good_min_routine.json
Output file: (not saving)
Energy modules: ['surface']
Constraint modules: ['volume']
Instructions: ['g100', 'r', 'u', 'g100', 'V', 'g20', 'r', 'g50', 'r', 'g50']
>
```

Type commands at the prompt; multiple commands can be written without spaces
(`g10rV5`), or as separate tokens (`g10 r V5`). Use `help` at any time.

Interactive commands:

- `gN`
  Run `N` minimization steps (e.g. `g5`, `g100`). Bare `g` runs one step.

- `gd` / `cg`
  Switch to Gradient Descent or Conjugate Gradient steppers. CG is the default.

- `tX`
  Set step size to `X` (e.g. `t1e-3`).

- `r` / `rN`
  Refine the mesh (triangle refinement + polygonal refinement). Provide a
  number (`r3`) to repeat the refinement pass multiple times. After each pass
  hard constraints are re‑enforced.

- `V` / `VN` / `vertex_average`
  Vertex averaging once (`V` / `vertex_average`) or `N` times (`V5`). After
  averaging, hard constraints are re‑enforced.

- `u`
  Equiangulate the mesh (edge flips to improve triangle quality), followed by
  constraint re‑enforcement.

- `properties` / `props` / `p` / `i`
  Print physical properties (global and per‑body area/volume).

- `visualize` / `s`
  Plot the current geometry in a Matplotlib 3D view.

- `save`
  Save the current geometry to `interactive.temp`.

- `help`, `h`, `?`
  Show a summary of interactive commands and CLI options.

- `quit`, `exit`, `q`
  Leave interactive mode.

> **Tip: Avoiding mesh tangling**
> When running large deformations (e.g. relaxing a square to a circle), avoid
> running many minimization steps (`g50`) in a single block immediately after
> refinement. Instead, interleave minimization with mesh maintenance commands:
> `r` -> `g10` -> `u` (equiangulate) -> `V` (vertex average) -> `g10` ...
> This allows the mesh to "un-kink" and redistribute vertices as it deforms,
> preventing overlapping triangles and degenerate edges.

---

## 4. Mesh JSON structure (overview)

Meshes are JSON files describing vertices, edges, facets, bodies, global
parameters, energy modules and constraint modules. The exact schema is
validated in the test suite; this section focuses on the main concepts.

High‑level layout:

- `vertices`: list of vertex objects with:
  - `index`: integer ID.
  - `position`: `[x, y, z]`.
  - `fixed`: optional boolean; if true the vertex never moves.
  - `options`: optional dict for local parameters and constraints.

- `edges`: list of edges:
  - `index`: integer ID (1‑based).
  - `tail`, `head`: vertex indices.
  - `options`: optional dict.

- `facets`: list of facets:
  - `index`: integer ID.
  - `edge_indices`: list of signed edge indices describing an oriented loop.
  - `fixed`: optional boolean.
  - `options`: dict with keys such as:
    - `"energy"`: list or string of energy module names (e.g. `"surface"`).
    - `"constraints"`: list or string of constraint module names.
    - `"surface_tension"`: per‑facet tension overriding the global default.
    - `"target_area"`: used by facet‑area constraints.

- `bodies`: list of bodies:
  - `index`: integer ID.
  - `facet_indices`: list of facet indices forming a closed region.
  - `target_volume`: optional target volume (used by volume constraints).
  - `options`: dict with keys such as:
    - `"target_volume"`: same as above if `target_volume` is omitted.
    - `"target_area"`: surface‑area constraints.

- `energy_modules`: list of energy module names to load globally.
- `constraint_modules`: list of constraint module names to load globally.
- `instructions`: list of interactive commands to run on startup
  (e.g. `["g100", "r", "u", "g100", "V", "g20", "r", "g50", "r", "g50"]`).

- `global_parameters`: dictionary of defaults and global settings, such as:
  - `"surface_tension"` (float, default `1.0`).
  - `"volume_stiffness"` (float, for penalty mode).
  - `"volume_constraint_mode"`: `"lagrange"` (default) or `"penalty"`.
  - `"volume_projection_during_minimization"` (bool).
  - `"volume_tolerance"` (float).
  - `"step_size"` (float initial step size).
  - `"max_zero_steps"`, `"step_size_floor"` (line‑search/stopping tweaks).
  - `"target_surface_area"`: for global area constraints.
- `"perimeter_constraints"`: see §6.4.

The loader (`geometry.geom_io.parse_geometry`) also:

- Automatically triangulates polygonal facets.
- Builds connectivity maps (vertex↔edges↔facets).
- Populates cached vertex loops for faster geometry calculations.

> **Volume constraint defaults:** For stability the loader enforces paired
> settings. If neither `volume_constraint_mode` nor
> `volume_projection_during_minimization` is present, it defaults to
> `("lagrange", False)`. Supplying only one field automatically picks the
> complementary value (e.g. choosing `penalty` forces projection `True`).
> You can still override both explicitly if needed.

### 4.1 Advanced Input Formats (YAML and Presets)

**YAML Support:**
You can now use `.yaml` or `.yml` files for your meshes. This allows for comments and standard YAML features like anchors (`&`) and aliases (`*`) to reduce repetition.

**Presets (Definitions):**
To avoid repeating complex constraint configurations, you can define a `definitions` block in your input file (JSON or YAML) and reference them using the `preset` key.

Example (`catenoid_good_min.json` style):

```json
{
  "definitions": {
    "top_ring": {
      "fixed": true,
      "constraints": ["pin_to_circle"],
      "pin_to_circle_normal": [0, 0, 1],
      "pin_to_circle_point": [0, 0, 1],
      "pin_to_circle_radius": 1.5
    }
  },
  "edges": [
    [0, 1, {"preset": "top_ring"}],
    [1, 2, {"preset": "top_ring"}]
  ]
}
```

Any options provided in the entity's dictionary will merge with and override the preset values.

---

## 5. Energies

### 5.1 Surface tension (`modules/energy/surface.py`)

This is the default energy for facets. It computes

\[
E_{\text{surface}} = \sum_{\text{facets } f} \gamma_f \, A_f,
\]

where `A_f` is the area and `γ_f` is the surface tension:

- Per‑facet value from `facet.options["surface_tension"]` if present.
- Otherwise `global_parameters["surface_tension"]`.

Every facet automatically gets surface energy unless you explicitly omit the
`surface` module **and** set surface tension to zero.

### 5.2 Volume penalty (`modules/energy/volume.py`)

The volume energy module is active only when:

- `volume_constraint_mode == "penalty"`, **and**
- the `volume` energy module is listed in `energy_modules`.

In that case the body energy is

\[
E_{\text{vol}} = \tfrac{1}{2} k (V - V_0)^2,
\]

where:

- `V` is the current body volume,
- `V0` is the target volume (`target_volume` or `options["target_volume"]`),
- `k` is stiffness (`body.options["volume_stiffness"]` or
  `global_parameters.volume_stiffness`).

In the default `"lagrange"` mode this energy is **disabled**; volume is
handled by the constraint system instead.

### 5.3 Line Tension (`modules/energy/line_tension.py`)

Minimizes the total length of edges flagged with this energy. It computes:

\[
E_{\text{line}} = \sum_{\text{edges } e} \lambda_e \, L_e,
\]

where `L_e` is the edge length and `\lambda_e` is the line tension coefficient.

Configuration:
- Add `"line_tension"` to `energy_modules`.
- Set global tension: `global_parameters["line_tension"] = 1.0`.
- Override per-edge: `edge.options["line_tension"] = 0.5`.
- Flag edges: Ensure edges have `"energy": ["line_tension"]` in their options.

This is commonly used for 2D problems (e.g. square relaxing to a circle) or
for boundary tension in open membranes.

You can also assign line tension from the CLI without editing JSON:

- `--line-tension VALUE`
  Apply a uniform line‑tension modulus to edges.

- `--line-tension-edges ID1,ID2,...`
  Restrict the CLI‑assigned line tension to the listed edge IDs. Without this
  option, all edges are tagged.

These flags update `mesh.energy_modules` and edge options at load time, so
the rest of the pipeline (`EnergyModuleManager`, refinement, constraints) sees
them as if they had been specified in the input file.

---

## 6. Constraints

Constraints are implemented either as:

- Geometric position/gradient projections attached directly to entities
  (e.g. `PinToPlane`), or
- Constraint modules under `modules/constraints`, managed by
  `ConstraintModuleManager` and invoked via `constraint_modules`.

### 6.1 Volume constraint (`modules/constraints/volume.py`)

This is the main mechanism for fixed volume in `"lagrange"` mode.

Activation:

- Include `"volume"` in `constraint_modules`, and
- Set a `target_volume` either on the body or in `body.options`.

Behaviour:

- During minimization, the gradient is projected onto the fixed‑volume
  manifold (Lagrange‑style).
- After discrete mesh operations (refinement, equiangulation, vertex
  averaging), a geometric projection step nudges the mesh back to the exact
  target volume.

### 6.2 Surface‑area constraints

- **Body surface area** (`modules/constraints/body_area.py`)
  - Add `"body_area"` to `constraint_modules`.
  - Set `body.options["target_area"]` for each constrained body.

- **Facet surface area** (`modules/constraints/fix_facet_area.py`)
  - Add `"fix_facet_area"` to `constraint_modules`.
  - Set `facet.options["target_area"]` on each constrained facet.

- **Global surface area** (`modules/constraints/global_area.py`)
  - Add `"global_area"` to `constraint_modules`.
  - Set `global_parameters["target_surface_area"]`.

All of these use small Lagrange‑multiplier style corrections (displacing
vertices along area gradients) to match the specified targets.

### 6.3 Perimeter constraints (`modules/constraints/perimeter.py`)

Global parameter `perimeter_constraints` may be a list of dicts:

```json
"global_parameters": {
  "perimeter_constraints": [
    {
      "edges": [1, 2, 3, 4],
      "target_perimeter": 4.0
    }
  ]
}
```

Add `"perimeter"` to `constraint_modules` to activate. Each constraint defines:

- `edges`: a loop of (signed) edge indices.
- `target_perimeter`: target total length.

The module computes perimeter, its gradient, and applies a small Lagrange step
to match the target.

Example inputs live in `tests/sample_meshes.square_perimeter_input`, and the
regression harness (`tests/test_perimeter_minimization.py`) demonstrates how to
distort the loop, run a short minimization, and verify that the perimeter (and
body-area constraint) are driven back toward their targets even after
refinement and equiangulation. Small residual deviations are expected because
refinement and equiangulation slightly change the discrete geometry; the tests
assert improvement and proximity rather than exact equality.

### 6.4 Geometric constraints: `PinToPlane`

`modules/constraints/pin_to_plane.py` defines a `PinToPlane` helper you can
attach directly to vertices:

```python
from modules.constraints.pin_to_plane import PinToPlane

plane = PinToPlane(plane_normal=[0, 0, 1], plane_point=[0, 0, 0])
vertex.options["constraint"] = plane
```

The minimizer will call `project_position` and `project_gradient` on such
constraints when updating positions/gradients, effectively pinning the vertex
to a plane and keeping forces tangent.

### 6.5 Fixed vertices

Any vertex with `fixed: true` in the JSON is held fixed:

- Its gradient is zeroed in `project_constraints`.
- Position updates and constraint corrections skip it.

This is the primary way to pin boundary curves or special anchor points.

---

## 7. Global parameters and stepper behaviour

Some important tuning parameters in `global_parameters`:

- `step_size`
  Initial step size for the line search (overridden by `tX` in interactive
  mode).

- `max_zero_steps`
  Maximum consecutive failed steps (step size below floor with no energy
  decrease) before early termination.

- `step_size_floor`
  Minimum allowed step size in the line search.

- `volume_constraint_mode`
  - `"lagrange"` (default): hard volume; `volume` energy disabled.
  - `"penalty"`: soft volume energy; use with the `volume` energy module.

- `volume_projection_during_minimization`
  - `False` (recommended with `"lagrange"`): rely on gradient projection and
    occasional hard projections after mesh operations.
  - `True`: force geometric volume projection during minimization as well
    (legacy behaviour; slower).

- `volume_tolerance`
  Allowed relative volume drift before a corrective projection is triggered.

The default stepper is Conjugate Gradient with Armijo backtracking line search.
You can switch to plain Gradient Descent with `gd` in interactive mode.

---

## 8. Worked example: cube → sphere

The mesh `meshes/cube_good_min_routine.json` sets up a cube that relaxes to a
fixed‑volume sphere.

Run:

```bash
python main.py -i meshes/cube_good_min_routine.json -o cube_sphere_out.json
```

The file:

- Selects the `surface` energy module and the `volume` constraint.
- Sets `global_parameters.volume_constraint_mode` to `"lagrange"`.
- Provides an instruction sequence:

  ```json
  "instructions": ["g100", "r", "u", "g100", "V", "g20", "r", "g50", "r", "g50"]
  ```

which roughly corresponds to:

1. Minimize 100 steps.
2. Refine and equiangulate.
3. Minimize again and perform vertex averaging.
4. Repeat refinement + minimization cycles to improve both shape and mesh
   quality.

You can inspect the final geometry with:

```bash
python visualize_geometry.py cube_sphere_out.json
```

For more control over rendering, the visualization CLI supports:

```bash
# Solid, facet-only view
python visualize_geometry.py cube_sphere_out.json --no-edges

# Semi-transparent facets with axes removed, saved to PNG
python visualize_geometry.py cube_sphere_out.json --transparent --no-axes \
                                                  --save outputs/cube_sphere.png

# Line-only meshes (edges only, no facets)
python visualize_geometry.py meshes/simple_line.json --no-facets --scatter
```

Internally, this uses the shared helper
`visualization.plotting.plot_geometry(mesh, ...)`, which is also exercised by
`tests/test_visualize_geometry.py`.

---

## 9. Developer notes (energy modules and inheritance)

For contributors, a few structural rules:

1. Energy modules are loaded once during input parsing. Each module exposes
   a function

   ```python
   compute_energy_and_gradient(mesh, global_params, param_resolver, *, compute_gradient=True)
   ```

   which returns `(energy, gradient_dict)`.

2. `param_resolver` allows per‑entity parameters (e.g. per‑facet
   `surface_tension`) to override `global_parameters`.

3. Default energies:
   - Facets get surface tension energy by default unless surface tension is
     set to zero and/or the module is omitted.
   - Bodies with a `target_volume` use the `volume` constraint in `"lagrange"`
     mode and the volume penalty only in `"penalty"` mode.

4. Inheritance rules during refinement:
   - Child facets inherit all energy and constraints of the parent facet.
   - Split edges inherit constraints of the parent edge.
   - New edges created inside a facet inherit facet‑level constraints.
   - Midpoint vertices inherit constraints (including `fixed`) from their
     parent edge or facet.

5. Common `options` keys:
   - `"refine": true/false` – refine or skip this entity when refining.
   - `"constraints": [...]` – explicit constraint modules for this entity.
   - `"energy": [...]` – explicit energy modules for this entity.
   - Parameter overrides such as `"surface_tension": 5.0`.

When adding new energies or constraints, implement the appropriate
`compute_energy_and_gradient` / `enforce_constraint` functions and update this
manual accordingly before merging into `main`.

6. Volume enforcement modes (global_parameters):
    - `"volume_constraint_mode": "lagrange"` (default) projects gradients
      using body volume gradients; bodies auto-load the hard `volume`
      constraint module.
    - `"volume_constraint_mode": "penalty"` re-activates the quadratic volume
      energy (`modules/energy/volume.py`) so you get Evolver-style ``VOLCONST``
      behaviour without the hard constraint overhead.
    - `volume_projection_during_minimization` controls whether the geometric
      projection runs inside the line search (mostly for legacy penalty mode).

10. Stability & Topology
------------------------

The solver includes safeguards inspired by Surface Evolver to prevent mesh
degeneracy (tangling, overlapping triangles) during energy minimization.

- **Safe Step Heuristic**: The line search automatically rejects steps that would
  cause any triangle to rotate by more than ~30 degrees (flip). To maintain high
  performance, this expensive geometric check is skipped for small steps (displacement
  < 30% of the minimum edge length).
- **Collision Detection**: After every `g` command, the solver checks for vertices
  that have drifted dangerously close to edges they do not belong to. A warning is
  logged (`TOPOLOGY WARNING`) if collisions are detected.
- **Recommendations**: If you see topology warnings:
  1. Reduce step size (`t`).
  2. Interleave `equiangulation` (`u`) and `vertex_averaging` (`V`) more frequently.
  3. Refine (`r`) the mesh to resolve sharp features.

7. Performance Optimization:
   - Core geometry routines (cross products, volume gradients) are heavily optimized.
   - Use `geometry.entities._fast_cross` for small-array cross products instead of `numpy.cross`.
   - Prefer pre-allocating numpy arrays with `np.empty` over list comprehensions in hot loops.
   - See `benchmarks/suite.py` for regression testing.
