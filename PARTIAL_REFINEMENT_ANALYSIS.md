# Partial Refinement Cases - Detailed Analysis

## Overview

Partial refinement occurs when only some edges of a triangle can be refined, typically due to `no_refine` constraints. This creates complex subdivision scenarios that require careful handling to maintain mesh quality and consistency.

## The Challenge

In the standard Evolver refinement scheme, all three edges of a triangle are refined simultaneously, creating a 1-to-4 subdivision:

```
Original Triangle:
    v2
    /\
   /  \
  /    \
 /______\
v0      v1

Standard 1-to-4 Refinement:
    v2
    /\
   /  \
  m20  m12
 /____\
/      \
v0_m01_v1

Results in 4 triangles:
1. (v0, m01, m20)
2. (m01, v1, m12) 
3. (m12, v2, m20)
4. (m01, m12, m20) - center triangle
```

However, when some edges have `no_refine=True`, we can't create midpoints for those edges, leading to partial refinement scenarios.

## Partial Refinement Cases

### Case 1: One Edge Refinable (1-to-2 Subdivision)

When only one edge can be refined, we split the triangle into two triangles using the midpoint of the refinable edge.

#### Subcase 1a: Edge v0-v1 is refinable
```
Original:           After Refinement:
    v2                  v2
    /\                  /|\
   /  \                / | \
  /    \              /  |  \
 /______\            /   |   \
v0      v1          v0__m01__v1

Results in 2 triangles:
1. (v0, m01, v2) - uses split edge + diagonal
2. (m01, v1, v2) - uses split edge + original edges
```

#### Subcase 1b: Edge v1-v2 is refinable
```
Original:           After Refinement:
    v2                  v2
    /\                  /\
   /  \                /  \
  /    \              /____\
 /______\            /      m12
v0      v1          v0_______v1

Results in 2 triangles:
1. (v1, m12, v0) - diagonal + split edge + original edge
2. (m12, v2, v0) - split edge + original edges
```

#### Subcase 1c: Edge v2-v0 is refinable
```
Original:           After Refinement:
    v2                  v2
    /\                  /\
   /  \                /  \
  /    \              /    \
 /______\            m20____\
v0      v1             \    v1
                        \___/
                         v0

Results in 2 triangles:
1. (v2, m20, v1) - split edge + diagonal + original edge
2. (m20, v0, v1) - split edge + original edges
```

### Case 2: Two Edges Refinable (1-to-3 Subdivision)

When two edges can be refined, we create a more complex subdivision pattern. This is the most challenging case.

#### Subcase 2a: Edges v0-v1 and v1-v2 are refinable
```
Original:           After Refinement:
    v2                  v2
    /\                  /\
   /  \                /  \
  /    \              /____\
 /______\            /      m12
v0      v1          v0__m01__v1

Results in 3 triangles:
1. (v0, m01, v2) - uses one split edge + diagonal to opposite vertex
2. (m01, v1, m12) - uses both split edges + new connecting edge
3. (v2, m12, v0) - uses one split edge + original edge + diagonal
```

#### Subcase 2b: Edges v1-v2 and v2-v0 are refinable
```
Similar pattern with different edge combinations
```

#### Subcase 2c: Edges v2-v0 and v0-v1 are refinable
```
Similar pattern with different edge combinations
```

### Case 3: No Edges Refinable (1-to-1 Copy)

When no edges can be refined (all marked `no_refine` or belong to `no_refine` facets), the triangle is simply copied to the new mesh with identical topology.

## Current Implementation Status

### ✅ Fully Implemented Cases

1. **All edges refinable (1-to-4)**: Complete standard refinement
2. **One edge refinable (1-to-2)**: All three subcases implemented
3. **No edges refinable (1-to-1)**: Simple copy operation

### ⚠️ Partially Implemented Cases

**Two edges refinable (1-to-3)**: Currently falls back to copying the original facet unchanged. This is a conservative approach that maintains mesh validity but doesn't perform optimal refinement.

## Implementation Details

### Edge Refinability Determination

```python
# Collect edges that belong to no_refine facets
edges_in_no_refine_facets = set()
for facet in mesh.facets.values():
    if facet.options.get("no_refine", False):
        for ei in facet.edge_indices:
            edges_in_no_refine_facets.add(abs(ei))

# Determine which edges can be refined
for facet in mesh.facets.values():
    if not facet.options.get("no_refine", False):
        for ei in facet.edge_indices:
            edge_idx = abs(ei)
            edge = mesh.get_edge(edge_idx)
            if (not edge.options.get("no_refine", False) and 
                edge_idx not in edges_in_no_refine_facets):
                edges_to_refine.add(edge_idx)
```

### Subdivision Logic

```python
# Check refinability of each edge
refinable_edges = [abs(ei) in edges_to_refine for ei in oriented]

if all(refinable_edges):
    # Standard 1-to-4 refinement
elif sum(refinable_edges) == 1:
    # 1-to-2 refinement (implemented)
elif sum(refinable_edges) == 2:
    # 1-to-3 refinement (fallback to copy)
else:
    # No refinement (copy)
```

## Geometric Considerations

### Edge Orientation Preservation

When creating new edges in partial refinement:
- Split edges inherit properties from their parent edge
- Diagonal edges (connecting midpoints to opposite vertices) inherit properties from the parent facet
- All new edges respect the `no_refine` inheritance rules

### Normal Direction Consistency

After creating child triangles:
```python
for child_facet in child_facets:
    child_normal = child_facet.normal(new_mesh)
    if np.dot(child_normal, parent_normal) < 0:
        child_facet.edge_indices = [-idx for idx in reversed(child_facet.edge_indices)]
```

This ensures all child facets maintain the same orientation as their parent.

## Future Enhancements

### Complete 1-to-3 Refinement Implementation

The two-edge refinable case could be fully implemented with these patterns:

1. **Identify the non-refinable edge**
2. **Create midpoints for the two refinable edges**
3. **Connect the midpoints to create the internal subdivision**
4. **Ensure proper orientation and property inheritance**

Example for edges v0-v1 and v1-v2 refinable:
```python
# Create triangles:
# 1. (v0, m01, m20) where m20 = v2 (no midpoint)
# 2. (m01, v1, m12) 
# 3. (m01, m12, v2) where we connect m01 to v2 directly
```

### Hanging Node Handling

Partial refinement can create "hanging nodes" where refined edges meet unrefined edges. The current implementation handles this by:
- Maintaining edge connectivity through proper indexing
- Ensuring child edges inherit appropriate constraints
- Preserving mesh topology integrity

## Performance Implications

- **1-to-4 refinement**: Optimal performance, standard case
- **1-to-2 refinement**: Good performance, simple subdivision
- **1-to-3 refinement**: Currently conservative (copy), could be optimized
- **1-to-1 copy**: Minimal overhead

## Mesh Quality Considerations

Partial refinement can affect mesh quality:
- **Aspect ratios**: May become less uniform
- **Edge lengths**: Create scale differences between refined and unrefined regions
- **Connectivity**: Maintain topological correctness but may create irregular patterns

The current implementation prioritizes correctness and stability over aggressive refinement in complex cases.