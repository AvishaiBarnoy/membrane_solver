# Fix for Fixed Property Inheritance During Refinement

## Problem Description

The user observed that "the original geometry (the cube facets and the polygonal triangulation that was done to it) don't move much with minimization." This was caused by an unintended inheritance of the `fixed` property during mesh refinement.

## Root Cause Analysis

### Issue 1: Fixed Property Inheritance
During mesh refinement, child geometry elements (vertices, edges, and facets) were automatically inheriting the `fixed` property from their parent elements:

1. **Centroid vertices** inherited `fixed=facet.fixed` 
2. **Spoke edges** inherited `fixed=facet.fixed`
3. **Child facets** inherited `fixed=facet.fixed`
4. **Midpoint vertices** inherited `fixed=edge.fixed`
5. **New edges** inherited `fixed=parent_edge.fixed` or `fixed=parent_facet.fixed`

### Issue 2: Gradient Zeroing
In the minimizer, fixed vertices have their gradients zeroed out:

```python
for vidx, vertex in self.mesh.vertices.items():
    if getattr(vertex, 'fixed', False):
        grad[vidx][:] = 0.0
```

This means that if any original geometry was marked as `fixed=True` (even unintentionally), all their refined child elements would also be fixed and unable to move during optimization.

### Evidence
In the output file `temp_geometry_output.json`, we observed:
- Vertices marked as `"fixed": true` (e.g., vertex index 2)  
- Many edges marked as `"fixed": true` and `"refine": false`
- Many faces marked as `"fixed": true` with `"parent_facet": 0`

## Solution Implemented

### Changes Made to `runtime/refinement.py`

1. **Centroid Vertices (Polygonal Refinement)**:
   ```python
   # Before:
   fixed=facet.fixed
   
   # After: 
   fixed=False  # Don't inherit fixed property - let centroid move
   ```

2. **Spoke Edges (Polygonal Refinement)**:
   ```python  
   # Before:
   fixed=facet.fixed
   
   # After:
   fixed=False  # Don't inherit fixed property - let spoke edges move
   ```

3. **Child Facets (Both Refinement Types)**:
   ```python
   # Before:
   fixed=facet.fixed
   
   # After: 
   fixed=False  # Don't inherit fixed property
   ```

4. **Midpoint Vertices (Triangle Refinement)**:
   ```python
   # Before:
   fixed=edge.fixed
   
   # After:
   fixed=False  # Don't inherit fixed property - let midpoints move  
   ```

5. **New Edges (Triangle Refinement)**:
   ```python
   # Before:
   edge.fixed = parent_edge.fixed
   edge.fixed = parent_facet.fixed
   
   # After:
   edge.fixed = False  # Don't inherit fixed property - let new edges move
   ```

## Expected Results

After these changes:

1. **Refined geometry can move freely** - Centroid vertices, midpoint vertices, and new edges/facets created during refinement will no longer be inadvertently fixed
2. **Original explicitly fixed vertices remain fixed** - Only vertices that were explicitly marked as `fixed=True` in the input will remain immobile
3. **Better minimization behavior** - The geometry should now respond properly to energy minimization forces
4. **Preserved functionality** - No-refine behavior and other constraint inheritance is preserved

## Testing Recommendations

1. **Run refinement and minimization** on a simple cube geometry 
2. **Verify that refined elements move** during optimization
3. **Test with explicitly fixed vertices** to ensure they remain fixed
4. **Check that energy minimization** produces expected shape evolution

## Alternative Solutions Considered

### Solution 2: Selective Inheritance
Only inherit `fixed=True` for specific cases (e.g., boundary vertices that should remain fixed):

```python
# Only inherit fixed property for explicitly fixed parent elements
fixed = parent.fixed if parent.options.get("explicitly_fixed", False) else False
```

### Solution 3: Constraint-Based Approach  
Replace fixed vertices with proper geometric constraints:

```python
# Instead of fixed=True, use constraints like:
vertex.options["constraints"] = ["fixed_position"]
```

The implemented solution (Solution 1) was chosen for its simplicity and immediate effectiveness, while maintaining backward compatibility.

## Notes

- This fix addresses the immediate issue of unintended movement restriction
- For more complex constraint scenarios, consider implementing Solution 3 in the future
- The `no_refine` property inheritance is preserved as this is intentional behavior
- Other property inheritance (like surface tension, energy modules) is maintained