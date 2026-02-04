# Performance Optimization Ideas

This document outlines strategies to improve the simulation speed of `membrane_solver`, combining architectural refactoring and low-level kernel optimizations.

## 1. Unified Leaflet Energy Modules
**Concept:** Merge `tilt_in`/`bending_tilt_in` and `tilt_out`/`bending_tilt_out` into unified leaflet-specific modules (e.g., `leaflet_energy_in.py`).

- **Shared Geometry:** Compute triangle positions, areas, normals, and area gradients once per leaflet and reuse them for both the tilt penalty and the bending-tilt coupling terms.
- **Reduced Overhead:** Halves the memory bandwidth required for gathering vertex data and scattering gradients.
- **Improved Orchestration:** Reduces function call depth in the high-frequency tilt relaxation loops.

## 2. Fortran Offloading for Tilt Relaxation
**Concept:** Move the iterative relaxation loops (Nested/Coupled) from `runtime/minimizer.py` into compiled Fortran kernels.

- **Loop Efficiency:** Eliminates Python interpreter overhead during the hundreds of inner-steps required for tilt convergence.
- **Boundary Minimization:** Dramatically reduces the number of times the code must cross the Python-Fortran boundary by passing the entire mesh state once and receiving the fully relaxed tilt field.

## 3. Native Fortran Bending and Tilt Gradients
**Concept:** Replace remaining NumPy-based energy and gradient calculations—particularly in `bending.py` and `bending_tilt_leaflet.py`—with Fortran implementations.

- **Hotspot Optimization:** Targeting `grad_cotan` and `np.add.at` operations which profiling identifies as primary bottlenecks.
- **SIMD Utilization:** Allows for low-level vectorization of complex geometric derivatives that are difficult for NumPy to optimize efficiently.

## 4. Strict "Scatter-Gather" SoA Architecture
**Concept:** Enforce a project-wide Structure of Arrays (SoA) pattern, removing all legacy dictionary-based fallbacks.

- **Zero-Conversion Pipeline:** Ensures the `EnergyModuleManager` and `Minimizer` always operate on dense arrays, bypassing the cost of dictionary creation and sparse updates.
- **Parallel Evaluation:** A clean SoA architecture enables safe parallelization of energy modules using OpenMP or multi-processing, as each module operates on independent or read-only views of the data.
