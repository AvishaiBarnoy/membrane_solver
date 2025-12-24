# Surface Evolver vs. Membrane Solver: Architectural Comparison

This document outlines the differences in design, physics, and performance between Ken Brakke's **Surface Evolver (SE)** and this **Membrane Solver**.

## 1. Data Architecture

| Feature | Surface Evolver | Membrane Solver |
| :--- | :--- | :--- |
| **Paradigm** | **Array of Structures (AoS)** | **Hybrid Structure of Arrays (SoA)** |
| **Storage** | Linked C structures (pointers). | Python entities (AoS) for topology; NumPy arrays (SoA) for optimization. |
| **Iteration** | C loops following pointers (fast but hard to parallelize). | Vectorized NumPy kernels (BLAS/LAPACK) via Scatter-Gather. |
| **Memory** | Manual management in C. | Managed by Python/NumPy. |

**Key Insight**: SE is optimized for low-latency pointer hopping in C. Our system is optimized for high-throughput batch operations. For very large meshes, our "Gather" overhead is offset by the efficiency of vectorized math.

## 2. Surface Tension (Area Minimization)

Both systems use the **Variation of Area** ($\nabla A$) to compute surface tension forces.
- **Evolver**: Computes $\nabla A$ facet-by-facet, accumulating into vertex force pointers.
- **Membrane Solver**: Computes $\nabla A$ for all triangles simultaneously in a single NumPy pass, using `np.add.at` to scatter results.

The resulting forces are **mathematically identical**.

## 3. Mean Curvature Implementation

### Surface Evolver (`sqcurve`)
- **Method**: **Variation of Area**.
- **Logic**: Since the area gradient $\nabla A$ is proportional to the mean curvature vector $\vec{H}$ ($\nabla A = -2H\vec{n}$), SE uses the magnitude of the area force to derive $H^2$.
- **Normalization**: SE usually integrates $(1/R_1 + 1/R_2)^2$.
- **Advantage**: Reuses existing "Area Force" code; works on any N-gon.

### Membrane Solver (`bending`)
- **Method**: **Discrete Laplace-Beltrami (Cotangent Weights)**.
- **Logic**: Directly computes the discrete shape operator using the angles of the triangles. Mean curvature is defined as $\Delta_{LB} \mathbf{x} = -2\vec{H}$.
- **Normalization**: Integrates $H^2$ (Willmore Energy). Note: $4\times$ smaller than SE's default.
- **Advantage**: Modern standard in DDG; superior convergence on non-uniform or distorted meshes.

## 4. Optimization & Steppers

| SE Command | Solver Equivalent | Notes |
| :--- | :--- | :--- |
| `g` (gradient) | `GradientDescent` | Both use line search. |
| `conj_grad` | `ConjugateGradient` | Both support Polak-Ribière. |
| `hessian` | N/A | The Solver does not yet implement a full Hessian. |

## 5. Performance Scaling
SE is faster for small meshes (< 1,000 vertices) due to the overhead of launching NumPy kernels and Python object creation.
The Membrane Solver becomes increasingly efficient as the vertex count grows, as the $O(N)$ vectorized math dominates the $O(1)$ Python overhead.
