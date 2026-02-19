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

- Test categories (mirrors CI jobs):
  - Unit: `pytest -q -m unit`
  - Regression: `pytest -q -m regression`
  - E2E: `pytest -q -m e2e`

- The tilt benchmark runner has smoke coverage to ensure all tilt meshes load and print metrics (`tests/test_tilt_benchmark_runner.py`).
- KH-pure refinement stability tests cover tilt benchmarks under mesh refinement (`tests/test_kh_pure_benchmarks.py`).
- Pre-commit includes a feature-branch guard; set `ALLOW_MAIN_BRANCH=1` to bypass if you must commit on `main`.

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

- `--properties`
  Compute and print basic physical properties (total area, volume, surface
  radius of gyration, per‑body area/volume) and exit without minimization.

- `--radius-of-gyration`
  Compute and print the surface radius of gyration and exit without
  minimization.

- `--viz` / `--viz-save PATH`
  Visualize the input geometry and exit (no minimization). Use `--viz-save` to
  save an image instead of only showing the figure.

- `--compact-output-json`
  Write the output mesh JSON in compact form (single line). Useful for large
  meshes or when diffing output files.

- `--debugger`
  Enter a post‑mortem debugger (prefers `ipdb`, falls back to `pdb`) on uncaught
  exceptions.

- `--volume-mode {lagrange,penalty}`
  Override the global `volume_constraint_mode`.
  - `lagrange` – treat volume as a hard constraint (default). Best paired with
    `global_parameters.volume_projection_during_minimization=false` to avoid
    redundant geometric projections.
  - `penalty` – add a quadratic volume energy term (soft constraint). Works
    best with `global_parameters.volume_projection_during_minimization=true`.

- `--log [PATH]`
  Write logs to a file. If `PATH` is omitted, a log file is written next to the
  input mesh (same basename, `.log` suffix). By default, no log file is written.

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
(`g10rV5`), or as separate tokens (`g10 r V5`), or separated by semicolons
(`g50; V3; g10`). Use `help` at any time.

Command history:

- Use the up/down arrow keys to cycle through previously-entered commands.
- History is persisted across sessions to `~/.membrane_solver_history` when the
  prompt is running in a terminal (TTY).
- Override the file via `MEMBRANE_HISTORY_FILE` or the length via
  `MEMBRANE_HISTORY_LENGTH`.

Interactive commands:

- `gN`
  Run `N` minimization steps (e.g. `g5`, `g100`). Bare `g` runs one step.

- `gd` / `cg` / `bfgs`
  Switch to Gradient Descent, Conjugate Gradient, or BFGS steppers. GD is the
  default.

- `hessian`
  Run a single BFGS-style Hessian step without changing the active stepper.

- `tX`
  Fix step size to `X` (e.g. `t1e-3`). Use `tf` / `t free` to re-enable adaptive
  step sizing.

- `set ...`
  Update global parameters or entity properties interactively.
  Examples:
  - `set surface_tension 1.5`
  - `set global step_size 1e-3`
  - `set vertex 0 fixed true`
  - `set vertex 8 z 1e-3`
  - `set edge 12 fixed true`
  - `set body 0 target_volume 1.0`

- `print ...`
  Inspect entities and derived properties.
  Examples:
  - `print vertex 0`
  - `print edges len > 0.5`
  - `print facets area > 0.1`

- `tilt_stats` / `tstat` (`in`/`out`)
  Print tilt magnitude and divergence summaries, optionally per leaflet.
  Examples:
  - `tstat` (prints `tilt_in` and `tilt_out` when available)
  - `tstat in`
  - `tstat out`

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

- `perturb SCALE` / `kick SCALE`
  Add small random noise to non-fixed vertex positions. Useful for breaking
  symmetry in flat geometries. Default scale is `0.01`.

- `fix [edges|facets|all] [where key=value]`
  Record current geometric properties (length for edges, area for facets) as
  `target_length` or `target_area`. Also automatically enables the
  `edge_length_penalty` energy and `fix_facet_area` constraint for the affected
  entities. Use the `where` clause to filter by options (e.g. `where preset=paper`).

- `properties` / `props` / `p` / `i`
  Print physical properties (global/per‑body area, volume, surface Rg, target volume).

- `python tools/suite.py --profile`
  Profile each benchmark case and save per-case `.pstats` files (plus optional
  text summaries via `--profile-top`) under `benchmarks/outputs/profiles` by default.

- `python tools/profile_tilt.py`
  Profile the inner-loop tilt relaxation hot paths (single-field or leaflet),
  writing `.pstats` and a text summary under `benchmarks/outputs/profiles`.

- `python tools/profile_macro_hotspots.py`
  Profile macro runs step-by-step and optionally capture one command
  (for example `g1`) with `cProfile` to emit `.pstats` and a text summary.

- `python tools/reproduce_theory_parity.py`
  Reproduce theory parity metrics and write a YAML report (default output:
  `benchmarks/outputs/diagnostics/theory_parity_report.yaml`).
  Modes:
  - `--protocol-mode fixed` (default): canonical acceptance protocol
    `g10 r V2 t5e-3 g8 t2e-3 g12`.
  - `--protocol-mode expanded`: convergence-gated exploratory ladder driven by
    `tests/fixtures/theory_parity_expansion_policy.yaml`, with persistent state
    in `benchmarks/outputs/diagnostics/theory_parity_expansion_state.yaml`.
  - Stage 4 in the expansion ladder is explicitly `r; g10`.
  Optional fixed-lane polish:
  - `--fixed-polish-steps N` appends `N` trailing `g1` steps in fixed mode only.
    Default is `0` (no behavior change).

- Theory parity YAML fixtures:
  - `tests/fixtures/theory_parity_baseline.yaml`:
    implementation-regression lane (protects against accidental drift).
  - `tests/fixtures/theory_parity_targets.yaml`:
    TeX-target lane (ratio/relationship validation).
  - `tests/fixtures/theory_parity_expansion_policy.yaml`:
    exploratory expansion policy and Stage 4 safety/rollback thresholds.

- Baseline/target refresh policy:
  - Update `theory_parity_baseline.yaml` only when intended implementation
    behavior changes (or deterministic protocol changes), and document why.
  - Update `theory_parity_targets.yaml` only when theory-alignment expectations
    are intentionally redefined or tightened.
  - Keep fixed-protocol acceptance checks green before promoting expanded mode.

- `python tools/theory_parity_trend.py`
  Generate fixed-lane parity trend diagnostics in YAML:
  `benchmarks/outputs/diagnostics/theory_parity_trend.yaml`.
  The artifact records per-ratio fields:
  - `actual`, `expected`, `abs_tol`, `abs_delta`, `within_tolerance`
  - summary fields:
    `ratio_count`, `within_tolerance_count`, `all_within_tolerance`

- `python tools/theory_parity_guarded_gate.py`
  Apply the fixed-lane guarded gate from trend output and persist streak state:
  `benchmarks/outputs/diagnostics/theory_parity_ci_state.yaml`.
  Gate rule:
  - fail only after 2 consecutive runs where `all_within_tolerance: false`
  - any passing run resets the streak to zero
  - missing prior state is treated as streak zero

- Guarded gate operations (CI runbook):
  - default mode in CI is non-blocking (informational signal).
  - set repository variable `PARITY_GUARDED_GATE=true` to make guarded gate
    blocking in pull requests.
  - unset the variable (or set anything else) to return to non-blocking mode.
  - when a guarded failure occurs, inspect uploaded artifacts first:
    `theory_parity_trend.yaml`, `theory_parity_report.yaml`,
    `theory_parity_ci_state.yaml`.
  - if rollback is needed during incident response, temporarily disable strict
    mode (`PARITY_GUARDED_GATE` not `true`), fix/tune, then re-enable.

- `python tools/tilt_benchmark_runner.py`
  Run `meshes/tilt_benchmarks/*.yaml` and print energy/tilt/divergence summaries
  (optionally writing JSON/CSV and plots).
- Dual-leaflet tilt fields can be visualized via `lv tilt_in`, `lv tilt_out`,
  `lv div_in`, `lv div_out`, or `lv bilayer` (dual-surface overlay colored by
  outer vs inner leaflet), and the tilt benchmark runner supports
  `--color-by tilt_in|tilt_out|tilt_div_in|tilt_div_out` when present.
- Tilt-source decay meshes are available under `meshes/tilt_benchmarks/` (rectangle source pair, annulus with inner-rim source).
- Tilt mass-mode benchmark (`lumped` vs `consistent`) for flat KH reproduction:
  `python benchmarks/benchmark_flat_disk_tilt_mass_mode.py --refine-level 1 --runs 2`.
- Kozlov Milestone-B annulus meshes live under `meshes/caveolin/`:
  - `meshes/caveolin/kozlov_annulus_flat_hard_source.yaml` (hard clamped rim tilt)
  - `meshes/caveolin/kozlov_annulus_flat_soft_source.yaml` (soft rim source via `tilt_rim_source_in`)
  - Decay-length estimate: `python benchmarks/benchmark_kozlov_annulus_decay_length.py --mesh hard` (or `--mesh soft`)
- For the θ_B contact-source form used in the 1-disk analytic derivation (`docs/tex/1_disk_3d.pdf`),
  use `tilt_rim_source_bilayer` with `tilt_rim_source_group` and `tilt_rim_source_strength`
  (equivalent to loading `tilt_rim_source_in` + `tilt_rim_source_out` with equal parameters).
  Alternatively, specify the contact term via `tilt_rim_source_contact_*` (Δε, a, h → γ) instead
  of setting `tilt_rim_source_strength*` directly.
- Single-leaflet disk+outer rim example (shape relaxation, outer leaflet induced via curvature):
  `meshes/caveolin/kozlov_1disk_3d_tensionless_single_leaflet_source.yaml`.
- Disk-profile single-leaflet example (analytic-style target on the disk):
  `meshes/caveolin/kozlov_1disk_3d_tensionless_single_leaflet_profile.yaml`.
- Disk-profile bilayer example (paper-style, symmetric disk forcing):
  `meshes/caveolin/kozlov_1disk_3d_tensionless_bilayer_profile.yaml`.
- Diagnostics helper:
  `python tools/diagnose_1disk_3d_single_leaflet.py --steps 50`.
- Flat one-leaflet TeX benchmark reproduction:
  - Theory-only exact values from `docs/tex/1_disk_flat.tex`:
    `python tools/reproduce_flat_disk_one_leaflet.py --outer-mode disabled`.
  - Free-outer-leaflet validation run:
    `python tools/reproduce_flat_disk_one_leaflet.py --outer-mode free`.
  - Physical `theta_B` scalar optimization mode:
    `python tools/reproduce_flat_disk_one_leaflet.py --outer-mode disabled --smoothness-model splay_twist --theta-mode optimize`.
  - Full local reduced-energy optimization mode (optimize + polish):
    `python tools/reproduce_flat_disk_one_leaflet.py --outer-mode disabled --smoothness-model splay_twist --theta-mode optimize_full`.
  - Side-by-side lane comparison from one command:
    `python tools/reproduce_flat_disk_one_leaflet.py --compare-lanes --refine-level 1 --output /tmp/flat_compare.yaml`.
    This writes both `legacy` and `kh_physical` reports plus a `comparison` summary.
  - Common options:
    `--fixture`, `--refine-level` (default `2`), `--smoothness-model`, `--theta-mode`, `--output`.
    Optional local rim-band refinement controls:
    `--rim-local-refine-steps`, `--rim-local-refine-band-lambda`.
  - Parameterization options:
    `--parameterization legacy|kh_physical` (default `legacy`).
    - `legacy` keeps the historical parity remap used by existing acceptance baselines.
    - `kh_physical` uses direct KH-style dimensionless coefficients (`kappa`, `kappa_t`)
      with explicit physical-to-dimensionless conversion.
  - Physical scaling controls (used when `--parameterization kh_physical`):
    `--kappa-physical`, `--kappa-t-physical`, `--length-scale-nm`,
    `--radius-nm`, `--drive-physical`.
    Example for the TeX physical scaling convention (`R=7 nm`, `L0=15 nm`,
    `kappa=10 kBT`, `kappa_t=10 kBT/nm^2`):
    `python tools/reproduce_flat_disk_one_leaflet.py --parameterization kh_physical --kappa-physical 10 --kappa-t-physical 10 --length-scale-nm 15 --radius-nm 7 --drive-physical 2.857142857142857 --outer-mode disabled --smoothness-model splay_twist --theta-mode optimize`.
  - Notes on parity:
    `legacy` is compared to the scalar reduced model from `docs/tex/1_disk_flat.tex`.
    `kh_physical` is compared to strict KH closed-form coefficients
    (`f=0.5*kappa*(div t)^2 + 0.5*kappa_t*|t|^2`) under the same disk/outer Bessel solution.
  - KH per-theta term audit (mesh vs analytic split):
    `python tools/diagnostics/flat_disk_kh_term_audit.py --refine-level 1 --theta-values 0 6.366e-4 0.004`.
  - KH audit refine sweep:
    `python tools/diagnostics/flat_disk_kh_term_audit.py --refine-levels 1 2 --theta-values 0 6.366e-4`.
  - KH audit local rim-band refinement:
    add `--rim-local-refine-steps` and `--rim-local-refine-band-lambda`.
  - Scan controls (`--theta-mode scan`):
    `--theta-min`, `--theta-max`, `--theta-count`.
  - Optimize controls (`--theta-mode optimize`):
    `--theta-initial`, `--theta-optimize-steps`, `--theta-optimize-every`, `--theta-optimize-delta`, `--theta-optimize-inner-steps`.
  - Optimize-full polish controls (`--theta-mode optimize_full`):
    `--theta-polish-delta`, `--theta-polish-points` (odd, >=3).
  - Optimize presets:
    `--optimize-preset fast_r3` for faster refine-3 iteration;
    `--optimize-preset full_accuracy_r3` for heavier refine-3 optimize/full runs;
    `--optimize-preset kh_wide` for wide KH-lane theta-span exploration.
  - Splay calibration control (benchmark-local, default unchanged):
    `--splay-modulus-scale-in` scales inner `tilt_splay_modulus_in` when
    `--smoothness-model splay_twist` (for refine-3 parity tuning experiments).
  - Inner tilt-mass discretization control:
    `--tilt-mass-mode-in auto|lumped|consistent` (default `auto`).
    `auto` maps to `consistent` for `kh_physical` and `lumped` for `legacy`.
  - Report parity now includes `recommended_mode_for_theory`:
    this selects `optimize` or `optimize_full` by a balanced score over
    `theta_factor` and `energy_factor` (defaults remain unchanged).
  - `--smoothness-model splay_twist` enables inner-leaflet KH-style
    splay/twist splitting (`tilt_splay_twist_in`); by default
    `tilt_twist_modulus_in` is zero unless explicitly set.
  - Baseline acceptance fixtures:
    - `tests/fixtures/flat_disk_one_leaflet_disabled_baseline.yaml`
    - `tests/fixtures/flat_disk_one_leaflet_free_baseline.yaml`
  - Baseline acceptance lane:
    `tests/test_reproduce_flat_disk_one_leaflet_acceptance.py`
  - Behavior/failure-mode tests:
    `tests/test_flat_disk_one_leaflet_benchmark_e2e.py`
    (including empty-scan-bracket and missing-disk-group error paths).
- Kozlov Milestone-C mesh (3D shape coupling) lives under `meshes/caveolin/`:
  - `meshes/caveolin/kozlov_annulus_milestone_c_soft_source.yaml` (bilayer tilt↔curvature coupling; use `break_symmetry` macro or `kick` if it stays flat)
- Bilayer tilt energies use `tilt_in`, `tilt_out`, `tilt_smoothness_in`, `tilt_smoothness_out`,
  and optional `tilt_coupling`. Parameters: `tilt_modulus_in/out`,
  `bending_modulus` (or `bending_modulus_in/out`), `tilt_coupling_modulus`,
  `tilt_coupling_mode`.
- Leaflet-aware bending-tilt coupling is available via `bending_tilt_in` and
  `bending_tilt_out`, using `bending_modulus_in/out` (fallback to `bending_modulus`).
- Example bilayer tilt decay meshes live under `meshes/bilayer_tilt/`, including
  `tilt_bilayer_rect_opposite.yaml`.
- Single-leaflet curvature-induction example: `meshes/bilayer_tilt/tilt_in_annulus_curvature_induction.yaml`
  (outer leaflet responds only via curvature; requires shape relaxation).

- `visualize` / `s`
  Plot the current geometry in a Matplotlib 3D view.
  Examples:
  - `s tilt arrows`
  - `s bilayer` (dual-leaflet overlay when `tilt_in`/`tilt_out` are present)

- `lv`
  Toggle live visualization during minimization; accepts the same scalar modes
  as `s` (including `bilayer`).

- `show_edges [on|off|toggle]`
  Control whether mesh edges are drawn in `s`/`lv` plots. With no argument, it
  toggles the current state.

- `energy`
  Print the per-module energy breakdown. When source modules are present, also
  prints internal energy (no sources) vs external work (sources).
  - `energy total` prints the scalar total only.
  - `energy ref` sets the current state as a reference; subsequent breakdowns
    also print Δ vs that reference.

- `print energy`
  Display the current total energy of the simulation.

- `save`
  Save the current geometry to `interactive.temp`.

- `history`
  Print the commands entered during the current session.

- `help`, `h`, `?`
  Show a summary of interactive commands and CLI options.

- `quit`, `exit`, `q`
  Leave interactive mode.

- `MACRO_NAME`
  If the input defines `macros`, typing a macro name runs its command sequence.

- Tab completion
  When running in a real terminal (TTY), pressing `TAB` completes interactive
  command names (and macro names).

## Expression-based energy/constraints

Entities may specify `expression` (energy) or `constraint_expression` with a
`constraint_target` (hard constraint). Expressions can reference `x`, `y`, `z`,
`x1/x2/x3`, global parameters, and safe math functions (`sin`, `cos`, `sqrt`,
etc.).

Default measures:
- vertices: point
- edges: length
- facets: area
- bodies: volume

Override with `expression_measure` or `constraint_measure`.

> **Tip: Avoiding mesh tangling**
> When running large deformations (e.g. relaxing a square to a circle), avoid
> running many minimization steps (`g50`) in a single block immediately after
> refinement. Instead, interleave minimization with mesh maintenance commands:
> `r` -> `g10` -> `u` (equiangulate) -> `V` (vertex average) -> `g10` ...
> This allows the mesh to "un-kink" and redistribute vertices as it deforms,
> preventing overlapping triangles and degenerate edges.

---

## 4. Error handling & diagnostics

- Edge and facet connectivity strictly use **1-based** IDs. If a command or mesh attempts to access edge `0` (or a missing index), the solver raises `InvalidEdgeIndexError` from `exceptions.py`, making failures easier to trace in logs and in tests (`tests/test_exceptions.py`).
- A fixed edge implies fixed endpoints: when an edge is marked `fixed`, both of its vertices are treated as fixed for minimization and are copied as fixed during refinement.
- Volume penalty calculations are covered by `tests/test_volume_energy.py`, which exercises both energy and gradient paths so future modules can safely rely on the helpers in `modules/energy/volume.py`.
- When `parse_geometry` detects NaN/inf vertex coordinates or edges that reference unknown vertices, it aborts immediately and writes a detailed message to the configured log file (`--log`), helping diagnose malformed inputs.
- Gauss-Bonnet drift monitoring is available for open surfaces with boundary loops. Enable it with `gauss_bonnet_monitor=true` and adjust tolerance scaling via:
  - `gauss_bonnet_eps_angle` (default `1e-4` radians),
  - `gauss_bonnet_c1` for the total invariant tolerance, and
  - `gauss_bonnet_c2` for per-loop tolerances.
  Debug logs report `G`, `K_int_total`, `B_total`, and each loop's `B_j` so drift can be localized after refinement or remeshing.
  Facets can be excluded from the diagnostic by setting
  `gauss_bonnet_exclude: true` in facet options (useful when rigid disks are
  treated as effective holes).
  For `gaussian_curvature`, boundary loops are supported by default: the
  module uses the Gauss–Bonnet sum (interior defects + boundary turning)
  and multiplies by `gaussian_modulus`. To exclude facets from this sum,
  set `gauss_bonnet_exclude: true` in facet options. Enable
  `gaussian_curvature_strict_topology=true` to raise on non-manifold edges,
  invalid boundary loops, or defect mismatches (tune with
  `gaussian_curvature_defect_tol`).
  On a closed torus, local Gaussian curvature can vary in sign across the
  surface while the integrated Gauss-Bonnet invariant remains zero; this is
  expected and should not be interpreted as a topology regression.

## 4.1 Interactive command highlights

- `print energy breakdown` prints per-module energy contributions.
- `print macros` lists defined macros (macros are also printed on load).

---

## 5. Mesh JSON structure (overview)

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

**Defines (Expression Symbols):**
For expression-based energies/constraints, you can define reusable symbols in a top-level `defines` block. These are evaluated as expressions using existing global parameters and other defines.

Example:

```yaml
global_parameters:
  angle: 60.0
defines:
  WALLT: "-cos(angle*pi/180)"
edges:
  1: [0, 1, {expression: "-(WALLT*y)", expression_measure: "length"}]
```

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

### 5.4 Edge Length Penalty (`modules/energy/edge_length_penalty.py`)

This module penalizes deviations from a target length, acting as an elastic
constraint. It computes:

\[
E_{\text{elastic}} = \sum_{\text{edges } e} \tfrac{1}{2} k \, (L_e - L_{e,0})^2,
\]

where:
- `L_e` is the current length.
- `L_{e,0}` is the `target_length` from edge options.
- `k` is the `edge_stiffness` from `global_parameters` (default `100.0`).

This is primarily used via the `fix edges` command to simulate inextensible
objects like paper.

### 5.5 Bending Energy (`modules/energy/bending.py`)

This module implements the squared mean curvature integral (Willmore energy):

\[
E_{\text{bending}} = \kappa \sum_{i} \frac{\|\vec{K}_i\|^2}{4 A_i},
\]

where:
- `\kappa` is the `bending_modulus` from `global_parameters`.
- `\vec{K}_i` is the integrated curvature vector at vertex `i`, calculated using **Cotangent Weights**.
- `A_i` is the dual vertex area, calculated using the **Mixed Voronoi** formulation.

The module ignores boundary vertices, meaning a flat planar patch correctly yields exactly zero bending energy.

#### 5.4.1 Helfrich model (spontaneous curvature)

The default bending model is `helfrich` with `spontaneous_curvature=0`, so it
reduces to the Willmore form when no spontaneous curvature is set. Use
`global_parameters.bending_energy_model="willmore"` to force the pure Willmore
form. The preferred curvature (for Helfrich) is taken from
`global_parameters.spontaneous_curvature` (alias: `intrinsic_curvature`).

#### 5.4.2 Gradient modes

Select the gradient implementation via `global_parameters.bending_gradient_mode`:

- `analytic` (default): backpropagates through the discrete cotan-Laplacian + mixed-area
  discretization and is validated against finite differences.
- `approx`: fast Laplacian-based approximation (stable for large runs, but less accurate).
- `finite_difference`: central-difference gradient of `compute_total_energy`
  (slow; intended for tests and debugging).

This energy is commonly used for curvature regularization or bending-dominated
relaxations (e.g. cube-to-sphere with surface tension disabled and area/volume
constraints enabled).

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

### 6.4 Rim-slope matching (`modules/constraints/rim_slope_match_out.py`)

Use this module to enforce the tensionless rim boundary condition (γ=0) as a
hard constraint rather than a penalty energy. It ties the outer-leaflet radial
tilt to the local outer slope at the rim, and optionally matches the inner
leaflet to the disk tilt just inside the rim.

Activation:

- Add `"rim_slope_match_out"` to `constraint_modules`.
- Set `rim_slope_match_group`, `rim_slope_match_outer_group`, and (optionally)
  `rim_slope_match_disk_group` in `global_parameters`.

Notes:
- The constraint uses a small-slope approximation and only differentiates
  along the plane normal.
- It operates per-vertex when the disk and rim rings share vertex counts;
  otherwise it uses a weighted mean disk tilt.
- The penalty energy module with the same name remains available as a fallback.

### 6.5 Geometric constraints: `PinToPlane`

`modules/constraints/pin_to_plane.py` is a geometric constraint module. Attach
it via the `constraints` list on vertices or edges:

```yaml
vertices:
  0: [0, 0, 1, {constraints: ["pin_to_plane"]}]
```

Alias support: `constraints: ["pin_surface_group_to_shape"]` maps to the same
module for non-breaking migration.

The module enforces the projection geometrically during minimization and after
mesh operations. Because it does not supply constraint gradients, it does not
participate in KKT projection and will emit a warning when gradients are
assembled.

Optional modes:
- `pin_to_plane_mode: "slide"` keeps the normal fixed but fits the plane point
  to the group centroid (lets the plane translate along its normal).
- `pin_to_plane_mode: "fit"` fits both normal and point from the tagged group.
- Use `pin_to_plane_group` to define which tagged vertices share the same plane.
- Alias keys `pin_surface_group_to_shape_mode/group/normal/point` are also
  accepted and mapped to the corresponding `pin_to_plane_*` keys.

### 6.6 Tilt-vector rim continuity (`modules/constraints/tilt_vector_match_rim.py`)

Use this module to enforce *full in-plane* tilt continuity across a disk/annulus
rim, per leaflet. It is intended for non-axisymmetric cases (e.g. multiple
disks on curved surfaces) where matching only the radial component is not
enough.

Tag vertices on two rings:
- the disk-side ring just inside the rim: `tilt_vector_match_role: "disk"`
- the rim ring at r=R: `tilt_vector_match_role: "rim"`
and set the same `tilt_vector_match_group` string on both rings to identify the
disk instance.

The module pairs vertices by polar angle in a local disk frame (center + fitted
plane normal) and constrains the two in-plane components (u,v) of `tilt_in` and
`tilt_out` to match between each paired vertex.

### 6.7 Geometric constraints: `PinToCircle`

`modules/constraints/pin_to_circle.py` pins tagged vertices/edges to a circle.
By default it uses a fixed plane, center, and radius. Set
`pin_to_circle_mode: "fit"` to keep the rim circular while letting the circle
translate/rotate with the mesh (use `pin_to_circle_group` to separate multiple
fitted rims).

```yaml
global_parameters:
  pin_to_circle_mode: fit
  pin_to_circle_radius: 1.0
vertices:
  - [1, 0, 0, {constraints: ["pin_to_circle"], pin_to_circle_group: "rim"}]
```

### 6.6 Fixed vertices

Any vertex with `fixed: true` in the JSON is held fixed:

- Its gradient is zeroed before stepping.
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

The default stepper is Gradient Descent with Armijo backtracking line search.
You can switch to Conjugate Gradient with `cg` in interactive mode.

### 7.1 Tilt relaxation parameters

These settings control how shape updates and tilt updates are scheduled:

- `tilt_solve_mode`
  - `"off"`: disable tilt relaxation.
  - `"nested"`: run tilt-only relaxation blocks inside each shape step.
  - `"coupled"`: interleave shape and tilt updates in a coupled loop.

For `nested` and `coupled`, the following iteration controls apply:

- `tilt_inner_steps`
  Number of tilt-only updates per nested relaxation block.

- `tilt_coupled_steps`
  Number of tilt updates per coupled outer step.

- `tilt_step_size`
  Initial line-search step size used by the tilt optimizer.

- `tilt_tol`
  Relative tolerance used to stop inner tilt iterations early when converged.

Tilt optimizer selection:

- `tilt_solver`
  - `"gd"` (default): gradient descent tilt relaxation.
  - `"cg"`: conjugate gradient tilt relaxation (opt-in).

- `tilt_cg_max_iters`
  Maximum CG iterations per tilt relaxation call. Defaults to the active
  `tilt_inner_steps`/`tilt_coupled_steps` count when unset.

- `tilt_cg_preconditioner`
  - `"jacobi"` (default for CG): diagonal preconditioner based on tilt modulus
    and smoothness weights.
  - `"none"`: disable preconditioning.

## 7.2 Macros

Input files can define macros (Evolver-style command sequences). A macro is a
named list of command lines; in interactive mode, typing the macro name runs
those commands in order. Macros may also be referenced from `instructions`.

Example:

```yaml
macros:
  gogo: "g 5; r; g 5; r; g 5"
instructions:
  - gogo
```

## 7.3 Explicit IDs (Optional)

The input format supports an optional “explicit ID” mapping form for
`vertices`, `edges`, `faces`, and `bodies`. This is useful when defining bodies
and facets by hand, since it avoids counting list positions.

Example:

```yaml
vertices:
  10: [0, 0, 0]
  20: [1, 0, 0]
  30: [0, 1, 0]
edges:
  1: [10, 20]
  2: [20, 30]
  3: [30, 10]
faces:
  100: [1, 2, 3]
bodies:
  7:
    faces: [100]
    target_volume: 0.0
```

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
python -m visualization.cli cube_sphere_out.json
```

For more control over rendering, the visualization CLI supports:

```bash
# Solid, facet-only view
python -m visualization.cli cube_sphere_out.json --no-edges

# Semi-transparent facets with axes removed, saved to PNG
python -m visualization.cli cube_sphere_out.json --transparent --no-axes \
                                                 --save outputs/cube_sphere.png

# Line-only meshes (edges only, no facets)
python -m visualization.cli meshes/simple_line.json --no-facets --scatter
```

Internally, this uses the shared helper
`visualization.plotting.plot_geometry(mesh, ...)`, which is also exercised by
`tests/test_visualize_geometry.py`.

### 8.2 Paper Folding (Spontaneous Curvature)

To simulate a piece of paper rolling into a cylinder, we use spontaneous
curvature combined with inextensibility constraints (fixed local area and fixed
edge lengths).

Benchmark: `benchmarks/inputs/bench_spontaneous_folding.json`

Sequence of operations:
1.  **Refine**: Create a dense enough mesh to resolve the curvature.
2.  **Snapshot**: Run `snapshot all` (alias: `fix all`) to lock the current (flat) area and lengths as the target state.
3.  **Perturb**: Run `perturb 0.05` to break the flat symmetry.
4.  **Bending**: Ensure `bending_energy_model` is `"helfrich"` and `spontaneous_curvature` is non-zero.
5.  **Minimize**: Use `bfgs` for efficient minimization of the stiff bending energy.

```text
r2
snapshot all
perturb 0.05
bfgs
g500
```

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
- **Body Orientation Checks**: Closed bodies are validated to ensure consistent
  facet orientation and positive signed volume. Open bodies (e.g., a droplet on a
  hard plane) are exempt from the outwardness check because their volume depends
  on the closure convention.
- **Recommendations**: If you see topology warnings:
  1. Reduce step size (`t`).
  2. Interleave `equiangulation` (`u`) and `vertex_averaging` (`V`) more frequently.
  3. Refine (`r`) the mesh to resolve sharp features.

7. Performance Optimization:
   - Core geometry routines (cross products, volume gradients) are heavily optimized.
   - Use `geometry.entities._fast_cross` for small-array cross products instead of `numpy.cross`.
   - Prefer pre-allocating numpy arrays with `np.empty` over list comprehensions in hot loops.
   - Tilt relaxation caches curvature/area weights while geometry is frozen to avoid recomputation.
   - See `tools/suite.py` for regression testing.

8. Optional compiled kernels (Fortran / f2py):
   - Some hot-loop kernels can be accelerated with Fortran, compiled into Python
     extension modules via NumPy f2py.
   - Kernels are **optional** and are not built on import. Build them explicitly
     (see below), otherwise the solver uses the pure Python/NumPy implementation.
   - Prerequisites:
     - A Fortran compiler (typically `gfortran`).
     - NumPy (for `numpy.f2py`).
     - On macOS, you may need to install a compiler toolchain (e.g. via Homebrew or Xcode CLI tools).
   - Build helper (recommended):
     - `python -m membrane_solver.build_ext`
     - Or at install time:
       - `MEMBRANE_SOLVER_BUILD_EXT=1 pip install -e .`
   - Example manual build (surface energy kernel):
     - `cd fortran_kernels && python -m numpy.f2py -c -m surface_energy surface_energy.f90`
     - This should produce `fortran_kernels/surface_energy.*.so` (platform-specific name).
   - Example manual build (bending kernels, optional):
     - `cd fortran_kernels && python -m numpy.f2py -c -m bending_kernels bending_kernels.f90`
     - This should produce `fortran_kernels/bending_kernels.*.so` (platform-specific name).
   - Example manual build (tilt kernels, optional):
     - `cd fortran_kernels && python -m numpy.f2py -c -m tilt_kernels tilt_kernels.f90`
     - This should produce `fortran_kernels/tilt_kernels.*.so` (platform-specific name).
   - Runtime behaviour:
     - If `fortran_kernels.surface_energy` is importable, the `surface` energy module
       will use it automatically for pure-triangle meshes; otherwise it falls back to NumPy.
   - Set `MEMBRANE_DISABLE_FORTRAN_SURFACE=1` to force the NumPy fallback.
   - Set `MEMBRANE_DISABLE_FORTRAN_BENDING=1` to disable compiled bending kernels.
   - Set `MEMBRANE_DISABLE_FORTRAN_TILT=1` to disable compiled tilt kernels (divergence/curvature).
