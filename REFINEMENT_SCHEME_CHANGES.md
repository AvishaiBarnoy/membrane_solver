# Refinement Scheme Changes

## Overview

The refinement scheme has been updated to match the Evolver's behavior as described in the user's requirements. The key changes ensure that:

1. The `no_refine` flag on a facet prevents edges created within that facet from being refined
2. Child edges preserve the normal direction of the parent facet
3. The refinement follows the standard 1-to-4 subdivision pattern for triangles

## Key Changes Made

### 1. Modified Edge Refinement Logic

**Previous behavior**: The `no_refine` flag on a facet prevented the entire facet from being subdivided.

**New behavior**: The `no_refine` flag on a facet prevents its edges from being refined, even if those edges belong to other facets that could be refined.

**Implementation**: 
- Added logic to collect all edges that belong to facets marked with `no_refine=True`
- Edges are only refined if they don't belong to any `no_refine` facet and aren't marked `no_refine` themselves

### 2. Improved Edge Property Inheritance

**Changes made**:
- Updated `get_or_create_edge()` function to accept parent edge and parent facet parameters
- Edges created from splitting parent edges inherit the parent edge's properties
- Edges created within a facet (connecting midpoints) inherit the facet's properties
- If a parent facet has `no_refine=True`, new edges created within it are marked with `no_refine=True`

### 3. Enhanced Normal Direction Preservation

**Implementation**:
- All child facets are checked against their parent facet's normal direction
- If a child facet's normal is not aligned with the parent (dot product < 0), the edge indices are reversed
- This ensures consistent orientation across refinement levels

### 4. Fixed Facet Indexing and Options Preservation

**Issues fixed**:
- Corrected facet constructor calls to use proper parameter names
- Fixed facet indexing to avoid conflicts between copied and newly created facets
- Ensured that facet options (including `no_refine`) are properly preserved when copying facets

### 5. Partial Refinement Support

**Added support for**:
- Cases where only some edges of a triangle can be refined
- 1-to-2 subdivision when only one edge is refinable
- Fallback behavior for complex partial refinement cases

## Evolver Compliance

The updated refinement scheme now follows the Evolver's approach:

> "Refinement" of a triangulation refers to creating a new triangulation by subdividing each triangle of the original triangulation. The scheme used in the Evolver is to create new vertices at the midpoints of all edges and use these to subdivide each facet into four new facets each similar to the original.

> Certain attributes of new elements are inherited from the old elements in which they were created. Fixedness, constraints, and boundaries are always inherited. Torus wrapping on edges is inherited by some but not all new edges. Surface tension and displayability are inherited by the new facets. 'Extra' attributes are inherited by the same type of element.

The key insight is that `no_refine` on a facet means "don't refine edges created in this facet", not "don't refine this facet at all".

## Test Results

All existing tests continue to pass, including:
- `test_no_refine_skips_triangle_refinement`: Now correctly handles the case where one triangle has `no_refine=True`
- `test_polygonal_facets_triangulated_even_with_no_refine`: Continues to work as expected
- All other refinement tests pass, ensuring backward compatibility

## Files Modified

- `runtime/refinement.py`: Main refinement logic updated
- All changes are backward compatible with existing functionality