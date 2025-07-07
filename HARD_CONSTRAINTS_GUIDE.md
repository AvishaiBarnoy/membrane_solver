# Hard Constraints with Lagrange Multipliers for Volume Conservation

This guide explains how to use hard constraints with Lagrange multipliers for volume conservation instead of the soft constraint approach that relies on `volume_stiffness`.

## Overview

### Soft Constraints (Previous Approach)
- **Method**: Penalty method with energy term `E = 1/2 * k * (V - V₀)²`
- **Parameter**: `volume_stiffness` (spring constant)
- **Issues**: 
  - Volume conservation is approximate
  - Requires tuning of stiffness parameter
  - Can cause numerical stiffness issues
  - Higher stiffness → better volume conservation but harder to solve

### Hard Constraints (New Approach)
- **Method**: Lagrange multipliers with exact constraint enforcement
- **Parameter**: `volume_multiplier` (Lagrange multiplier λ)
- **Benefits**:
  - Exact volume conservation
  - No need to tune stiffness parameters
  - Better numerical stability
  - Physically more accurate

## Implementation Details

### 1. Constraint Enforcement
The hard constraint approach uses two mechanisms:

**A. Direct Constraint Enforcement (`modules/constraints/volume.py`)**
- Applies vertex displacements after each optimization step
- Uses Lagrange multiplier approach: `displacement = -λ * ∇V`
- Ensures exact volume conservation

**B. Energy-Based Lagrange Multipliers (`modules/constraints/fixed_volume.py`)**
- Adds Lagrange multiplier term to energy: `E = λ * (V - V₀)`
- Gradient contribution: `∇E = λ * ∇V`
- Integrates constraint into optimization naturally

### 2. Configuration Changes

**Old Configuration (Soft Constraints):**
```json
{
    "bodies": {
        "faces": [[0, 1, 2, 3, 4, 5]],
        "target_volume": [1.0],
        "energy": ["volume"]
    },
    "global_parameters": {
        "volume_stiffness": 1000.0
    },
    "energy_modules": ["surface", "volume"],
    "constraint_modules": []
}
```

**New Configuration (Hard Constraints):**
```json
{
    "bodies": {
        "faces": [[0, 1, 2, 3, 4, 5]],
        "target_volume": [1.0]
    },
    "global_parameters": {
        "volume_multiplier": 0.0
    },
    "energy_modules": ["surface"],
    "constraint_modules": ["volume"]
}
```

### 3. Key Parameter Changes

| Parameter | Old (Soft) | New (Hard) | Description |
|-----------|------------|------------|-------------|
| `volume_stiffness` | Required (e.g., 1000.0) | Not used | Spring constant for penalty method |
| `volume_multiplier` | Not used | Optional (0.0) | Lagrange multiplier λ |
| Energy modules | `["volume"]` | Remove volume | No penalty energy needed |
| Constraint modules | `[]` | `["volume"]` | Enable hard constraint |

### 4. Lagrange Multiplier Updates

The Lagrange multiplier λ can be:
- **Static**: Set to a fixed value (usually 0.0 initially)
- **Dynamic**: Updated during optimization to maintain constraints
- **Body-specific**: Different λ for each body if needed

## Usage Examples

### Example 1: Converting Existing Mesh
To convert an existing mesh from soft to hard constraints:

1. **Remove volume from energy modules**:
   ```json
   "energy_modules": ["surface"]  // Remove "volume"
   ```

2. **Add volume to constraint modules**:
   ```json
   "constraint_modules": ["volume"]
   ```

3. **Update parameters**:
   ```json
   "global_parameters": {
       "volume_multiplier": 0.0  // Add this
       // Remove or ignore "volume_stiffness"
   }
   ```

### Example 2: Using the Example Configuration
Run the provided example:
```bash
python main.py -i meshes/cube_hard_constraints.json -o output_hard.json
```

### Example 3: Body-Specific Multipliers
For different Lagrange multipliers per body:
```json
"bodies": {
    "faces": [[0, 1, 2, 3, 4, 5]],
    "target_volume": [1.0],
    "options": {"volume_multiplier": 5.0}  // Body-specific λ
}
```

## Technical Details

### Volume Gradient Computation
Both approaches require computing `∇V` (volume gradient w.r.t. vertex positions):
- Computed by `body.compute_volume_gradient(mesh)`
- Returns dictionary `{vertex_id: gradient_vector}`
- Used for both constraint enforcement and energy gradients

### Constraint Enforcement Process
1. **Optimization step**: Standard gradient descent/conjugate gradient
2. **Constraint enforcement**: After each successful step
   - Compute volume error: `ΔV = V_current - V_target`
   - Compute volume gradient: `∇V`
   - Calculate Lagrange multiplier: `λ = ΔV / ||∇V||²`
   - Apply vertex displacements: `x_new = x_old - λ * ∇V`

### Numerical Considerations
- **Tolerance**: Volume constraints are enforced to tolerance `1e-12`
- **Iterations**: Maximum 3 iterations per constraint enforcement
- **Stability**: Hard constraints are generally more stable than high stiffness

## Comparison of Approaches

| Aspect | Soft Constraints | Hard Constraints |
|--------|------------------|------------------|
| Volume conservation | Approximate | Exact |
| Parameter tuning | Required (stiffness) | Minimal (λ usually 0) |
| Numerical stability | Can be poor | Generally better |
| Physical accuracy | Approximate | Exact |
| Computational cost | Per-step energy | Per-step constraint |
| Implementation | Energy penalty | Lagrange multipliers |

## Troubleshooting

### Common Issues
1. **Constraint not enforced**: Check that `"volume"` is in `constraint_modules`
2. **Volume still changing**: Verify constraint tolerance and max iterations
3. **Optimization instability**: Try smaller step sizes or different steppers

### Debugging
Enable debug logging to see constraint enforcement:
```python
logger.debug("Applying volume constraint on body {body.index}: ΔV={delta_v:.3e}, λ={lam:.3e}")
```

### Performance Tips
- Hard constraints are typically faster than very stiff penalty methods
- Use conjugate gradient (`"cg"`) for better convergence
- Start with smaller step sizes when switching approaches

## Migration Guide

To migrate from soft to hard constraints:

1. **Backup your current configuration**
2. **Modify JSON configuration** as shown above
3. **Test with simple geometry** (e.g., `cube_hard_constraints.json`)
4. **Verify volume conservation** in output
5. **Adjust optimization parameters** if needed

This approach provides exact volume conservation without the numerical issues associated with high stiffness penalty methods.