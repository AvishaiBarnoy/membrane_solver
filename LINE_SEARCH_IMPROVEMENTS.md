# Line Search Improvements and Optimizations

This document describes the improvements and optimizations made to the line search algorithms for step size selection in the optimization process.

## Overview

The original implementation used a basic Armijo backtracking line search. The enhanced implementation provides several significant improvements:

1. **Strong Wolfe Line Search**: A more sophisticated algorithm that satisfies both Armijo and curvature conditions
2. **Interpolation Methods**: Quadratic and cubic interpolation for efficient step size reduction
3. **More-Thuente Algorithm**: A robust two-phase (bracketing + zoom) approach
4. **Adaptive Parameters**: Dynamic adjustment of line search parameters based on problem characteristics
5. **Enhanced Optimization Steppers**: Improved gradient descent and conjugate gradient algorithms

## Key Improvements

### 1. Strong Wolfe Conditions

The enhanced line search satisfies both:
- **Armijo condition**: `φ(α) ≤ φ(0) + c₁·α·φ'(0)`
- **Curvature condition**: `|φ'(α)| ≤ c₂·|φ'(0)|`

This provides better convergence guarantees compared to Armijo-only conditions.

### 2. More-Thuente Algorithm

Implementation includes:
- **Bracketing Phase**: Efficiently finds an interval containing acceptable step sizes
- **Zoom Phase**: Uses interpolation to narrow down to optimal step size
- **Robust Safeguards**: Handles edge cases and numerical issues

### 3. Interpolation-Based Step Size Reduction

Instead of geometric reduction (α ← β·α), the enhanced version uses:
- **Cubic interpolation**: When sufficient information is available
- **Quadratic interpolation**: As a fallback
- **Adaptive reduction factors**: Based on iteration progress

### 4. Adaptive Parameter Selection

- **Dynamic c₂ adjustment**: Based on problem curvature estimates
- **Algorithm selection**: Automatically chooses best line search based on problem characteristics
- **Intelligent fallbacks**: Gracefully degrades to simpler methods when needed

## Available Line Search Algorithms

### 1. `backtracking_line_search` (Original)
- Basic Armijo backtracking
- Kept for backward compatibility
- Fixed geometric step reduction

### 2. `strong_wolfe_line_search` (New)
- Implements Strong Wolfe conditions
- Two-phase More-Thuente algorithm
- Interpolation-based step selection
- Adaptive parameters

### 3. `adaptive_line_search` (New)
- Automatically selects best algorithm
- Considers problem characteristics
- Provides optimal performance across different scenarios

## Enhanced Optimization Steppers

### Enhanced Gradient Descent

```python
from runtime.steppers.gradient_descent import EnhancedGradientDescent

stepper = EnhancedGradientDescent(
    line_search="adaptive",  # "adaptive", "strong_wolfe", "backtrack", "simple"
    max_iter=15,
    c1=1e-4,
    c2=0.9
)
```

### Enhanced Conjugate Gradient

```python
from runtime.steppers.conjugate_gradient import EnhancedConjugateGradient

stepper = EnhancedConjugateGradient(
    line_search="adaptive",
    beta_formula="polak_ribiere",  # "polak_ribiere", "fletcher_reeves", "hestenes_stiefel"
    adaptive_restart=True,
    c1=1e-4,
    c2=0.1  # Lower for CG
)
```

## Usage Examples

### Basic Usage (Drop-in Replacement)

The enhanced algorithms can be used as drop-in replacements:

```python
# Original
success, new_step = backtracking_line_search(
    mesh, direction, gradient, step_size, energy_fn
)

# Enhanced with Strong Wolfe
success, new_step = strong_wolfe_line_search(
    mesh, direction, gradient, step_size, energy_fn, gradient_fn
)

# Adaptive (recommended)
success, new_step = adaptive_line_search(
    mesh, direction, gradient, step_size, energy_fn, gradient_fn
)
```

### Advanced Configuration

```python
# Configure Strong Wolfe line search
success, new_step = strong_wolfe_line_search(
    mesh, direction, gradient, step_size, energy_fn, gradient_fn,
    c1=1e-4,          # Armijo parameter
    c2=0.9,           # Curvature parameter
    max_iter=20,      # Maximum iterations
    interpolation=True,    # Use interpolation
    adaptive_c2=True       # Adaptive curvature parameter
)
```

## Performance Characteristics

### Expected Improvements

1. **Fewer Function Evaluations**: Strong Wolfe + interpolation typically requires 2-5 function evaluations vs 5-10 for basic backtracking
2. **Better Step Sizes**: More accurate step size selection leads to fewer optimization iterations
3. **Improved Convergence**: Strong Wolfe conditions provide better convergence guarantees
4. **Robustness**: Better handling of ill-conditioned problems and edge cases

### When to Use Each Algorithm

- **`adaptive_line_search`**: Default choice - automatically selects best method
- **`strong_wolfe_line_search`**: When gradient information is available and high accuracy is needed
- **`backtracking_line_search`**: For backward compatibility or when memory/computation is extremely limited

## Algorithm Selection Logic

The `adaptive_line_search` selects algorithms based on:

1. **Gradient availability**: If `gradient_fn` is provided → Strong Wolfe
2. **Problem conditioning**: Based on gradient and direction norms
3. **Convergence history**: Falls back to simpler methods if sophisticated ones fail

## Integration with Existing Code

The improvements are designed to be minimally invasive:

1. **Backward Compatibility**: Original `backtracking_line_search` unchanged
2. **Gradual Adoption**: Can be integrated incrementally
3. **Automatic Benefits**: Using `EnhancedGradientDescent` or `EnhancedConjugateGradient` automatically provides improvements

## Technical Notes

### Numerical Stability

- Robust safeguards against division by zero
- Proper handling of very small/large step sizes
- Graceful degradation when interpolation fails

### Memory Efficiency

- Minimal additional memory overhead
- Efficient position saving/restoration
- No persistent state beyond what's necessary

### Constraint Handling

- Proper integration with vertex constraints
- Position projection after each trial step
- Maintains feasibility throughout line search

## Testing and Validation

The enhanced line search has been designed to:
- Pass all existing tests for backward compatibility
- Provide improved performance on standard optimization problems
- Handle edge cases robustly

To test the improvements:

```bash
python -m pytest tests/test_conjugate_gradient.py -v
```

## Future Enhancements

Potential areas for further improvement:
1. **Limited-memory BFGS line search**: For quasi-Newton methods
2. **Trust region methods**: Alternative to line search for some problems
3. **Parallel line search**: Multiple trial points evaluated simultaneously
4. **Problem-specific tuning**: Adaptive parameters based on energy function characteristics