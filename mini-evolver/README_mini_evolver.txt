Mini Evolver Engine
===================

Run:
  python3 evolver_cli.py <input>

Interactive usage (default):
  - Loads the file, runs its read block commands (if any), then opens a prompt.
  - Enter commands like "g 10; r; g 10" or a macro name (e.g. "gogo").
  - Use "q" to quit.

Input formats:
  - .fe (Surface Evolver data file subset)
  - .json / .yaml (geometry input; see inputs/cube.json and inputs/cube.yaml)

Examples:
  python3 evolver_cli.py fe/cube.fe
  python3 evolver_cli.py fe/mound.fe
  python3 evolver_cli.py inputs/cube.json

Implemented commands/macros:
  - g N / gN   Gradient descent for N steps
  - r          Refine: split each triangle into four
  - hessian    Quasi-Newton (BFGS) minimization pass
  - gogo       Runs the gogo macro if present in the input
  - refine     Treated like r when found inside macros
  - s          Interactive: enable live plot updates

Common options:
  --penalty <float>      Volume penalty weight
  --steps <int>          Max steps for standalone minimize (non-gogo)
  --enforce-volume       Nudge each step along volume gradient
  --hessian-steps <int>  BFGS steps per hessian command
  --no-gogo              Ignore gogo macro even if present
  --no-numpy             Disable numpy acceleration
  --non-interactive      Run once and exit (interactive is default)
  --live-plot            Show live plot updates during minimization
  --volume-mode <mode>   Volume constraint: penalty or saddle

JSON/YAML schema notes:
  - vertices: {"id": [x, y, z], ...}
  - faces: [[v1, v2, v3, ...], ...]
  - bodies: [{"id": 1, "faces": [1,2,3], "target_volume": 1, "density": 1.0}, ...]
    If bodies is omitted, a single body using all faces is assumed.
  - fixed_vertices: [ids]
  - vertex_constraints: {"id": constraint_id, ...}
  - constraints: {"id": {"axis": 3, "value": 0.0, "energy": {"e1": "...", "e2": "...", "e3": "..."}}, ...}
  - params: {"angle": 90.0, ...}
  - defines: {"WALLT": "(-cos(angle*pi/180))", ...}
  - macros: {"gogo": "g 5; r; ...", ...}
  - no_refine_edges: [[v1, v2], ...] or [edge_id, ...]
  - no_refine_faces: [face_id, ...]
  - surface_tension: <float> (default 1.0)
  - square_curvature_modulus: <float> (default 0.0)
  - quantities:
      - name: "bend"
        type: "energy"
        method: "sq_mean_curvature|area"
        modulus: 1.0
        scope: "global|body|facet"
        targets: [1, 2]        # body or facet ids
        params: {"h0": 0.0}    # spontaneous curvature
  --plot                 Show matplotlib 3D plot
  --save-plot <path>     Save plot to an image file
