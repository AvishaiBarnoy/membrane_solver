# Surface Tension and Energy Module Assignment Fix

## Questions Answered

### 1. **Does the new vertex inherit surface tension in polygonal refinement?**

**Answer: No, and this is correct behavior.**

In polygonal refinement, the new centroid vertex does NOT inherit surface tension because:
- **Surface tension is a facet property, not a vertex property**
- The centroid vertex only inherits constraints from the parent facet
- This is the correct behavior since surface tension affects facet energy, not vertex energy

```python
# Centroid vertex creation (correct)
centroid_options = {}
if "constraints" in facet.options:
    centroid_options["constraints"] = facet.options["constraints"]
centroid_vertex = Vertex(
    index=centroid_idx,
    position=np.asarray(centroid_pos, dtype=float),
    fixed=facet.fixed,
    options=centroid_options,  # Only constraints, no surface_tension
)
```

### 2. **Do all facets have energy modules and surface_tension?**

**Answer: Yes, after the fixes implemented.**

## Issues Found and Fixed

### Issue 1: Missing surface_tension during initial loading
**Problem**: Facets loaded from JSON got default `energy: ["surface"]` but no `surface_tension` value.

**Fix**: Added automatic `surface_tension` assignment during mesh loading:
```python
# In geometry/geom_io.py
# Ensure all facets have surface_tension set
if "surface_tension" not in options:
    mesh.facets[i].options["surface_tension"] = mesh.global_parameters.get("surface_tension", 1.0)
```

### Issue 2: Inconsistent child facet properties in polygonal refinement
**Problem**: Child facets created during polygonal refinement had inconsistent property inheritance.

**Fix**: Improved child facet creation to ensure consistent properties:
```python
# In runtime/refinement.py
child_options = facet.options.copy()  # Use copy() to avoid modifying parent
child_options["surface_tension"] = facet.options.get("surface_tension", mesh.global_parameters.get("surface_tension", 1.0))
child_options["parent_facet"] = facet.index
child_options["constraints"] = facet.options.get("constraints", [])
# Ensure child facets have energy module set
if "energy" not in child_options:
    child_options["energy"] = ["surface"]
```

## Test Results

### Cube.json Test (Polygonal → Triangular)
- ✅ **24 triangular facets** created from 6 polygonal faces
- ✅ **All facets have `energy: ["surface"]`**
- ✅ **All facets have `surface_tension: 1.0`** (from global parameters)
- ✅ **Proper parent-child relationship tracking**

### Simple Triangle Test
- ✅ **Single triangle gets default properties**
- ✅ **Custom surface_tension values are preserved**
- ✅ **Global surface_tension fallback works**

## Key Improvements

1. **Consistent Property Assignment**: All facets now get `energy` and `surface_tension` properties
2. **Global Parameter Fallback**: Missing properties use global parameter values
3. **Property Inheritance**: Child facets properly inherit from parent facets
4. **Custom Values Preserved**: User-specified values in JSON are not overwritten

## Impact

- **Energy calculations** will now work consistently across all facets
- **Surface tension effects** are properly applied to all surfaces
- **Mesh loading** is more robust and handles edge cases
- **Polygonal refinement** maintains property consistency

The system now ensures that every facet in the mesh has the necessary properties for energy calculations, regardless of how it was created (loaded from file or generated during refinement).