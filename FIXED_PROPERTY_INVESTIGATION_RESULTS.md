# Investigation Results: Fixed Property Issue During Minimization

## Problem Statement

The user observed that "the original geometry (the cube facets and the polygonal triangulation that was done to it) don't move much with minimization."

## Investigation Process

### Initial Hypothesis (Incorrect)
I initially suspected that **inheritance of the `fixed` property during refinement** was causing the issue and modified the refinement code to prevent inheritance. However, this was incorrect.

### User Correction
The user correctly pointed out that:
> "Elements should inherit the fixed option. If a facet has `fixed=True` then its child vertices will inherit `fixed=True` and the vertices that define that facet through the edges will not be affected by it."

This inheritance behavior is **intentional and correct**.

### Root Cause Analysis

#### Key Evidence from `temp_geometry_output.json`:
- **Vertex 2**: `"fixed": true`
- **Many edges**: `"fixed": true` and `"refine": false`
- **Many faces**: `"fixed": true` with `"parent_facet": 0`

The presence of `"parent_facet": 0` indicates these child elements inherited `fixed=True` from **the original facet 0**.

#### The Real Issue
The `temp_geometry_output.json` file contains elements that are **legitimately fixed** due to inheritance from a parent that was explicitly marked as `fixed=True`. However, this file appears to have been created from a **different, more complex input** than the simple `cube.json`.

### Evidence that `temp_geometry_output.json` ≠ `cube.json` output:

1. **Complex features in temp file not in cube.json**:
   - Constraints: `"pin_to_plane"`, `"fix_facet_area"`
   - Energy modules: `"line_tension"`, `"dummy_module"`
   - Complex surface tension values: `"surface_tension": 5`

2. **cube.json is simple**:
   - No `fixed` properties anywhere
   - No constraints
   - Only basic surface tension

### Code Investigation Results

1. **Entity classes**: All default to `fixed=False`
2. **Parsing code**: Only sets `fixed=True` when explicitly specified in input
3. **Refinement code**: Correctly inherits `fixed` property from parent elements

## Correct Diagnosis

**The minimization issue is NOT caused by inappropriate inheritance during refinement.**

The most likely explanations are:

### Option 1: Wrong Input File
The `temp_geometry_output.json` was created from a **different input file** that had some elements explicitly marked as `fixed=True`, not from `cube.json`.

### Option 2: Step Size or Energy Issues  
If the actual input is `cube.json` (which has no fixed elements), then the lack of movement could be due to:
- **Step size too small**: `step_size` parameter might be too conservative
- **Energy landscape**: Local minimum or insufficient energy gradients
- **Constraint issues**: Some other constraint preventing movement
- **Convergence**: System might already be near equilibrium

## Recommended Next Steps

1. **Verify Input**: Confirm which input file is actually being used for minimization
2. **Test with cube.json**: Run refinement and minimization specifically with `cube.json` to see if movement occurs
3. **Check step size**: Increase step size parameter if movement is too slow
4. **Energy analysis**: Examine energy gradients to see if forces are present

## Conclusion

The inheritance behavior in the refinement code is **correct and should not be changed**. The investigation revealed that:

1. Fixed property inheritance works as intended
2. The observed fixed elements are likely from a different, more complex input file
3. The real issue may be related to minimization parameters or energy landscape rather than refinement behavior

The code changes I made to prevent inheritance should be **reverted** since they break the intended functionality.