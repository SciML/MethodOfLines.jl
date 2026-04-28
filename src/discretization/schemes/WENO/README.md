# Non-Uniform WENO Mathematical & Codegen Prototype

This directory contains the isolated mathematical and architectural prototypes for calculating **dynamic essentially non-oscillatory (WENO) weights**, smoothness indicators, and symbolic flux rules on non-uniform, clustered grids.

This prototype was built as a proof-of-concept for the GSoC 2026 proposal:
**Comprehensive Non-Uniform Grid Support for MethodOfLines.jl**.

## 1. Architectural Philosophy (Two-Stage Engine)

Following technical discussions with the SciML core team regarding `MethodOfLines.jl`'s rule-based multidimensional machinery, this prototype has been evolved into a strictly isolated, two-stage architecture:

1. **Numerical Reference Implementation (`weno_weights.jl`):** Handles the heavy mathematical derivation, resolving negative weight instabilities, and proving theoretical convergence limits.
2. **The Symbolic Codegen Engine (`weno_symbolic.jl`):** Translates the numerical mathematics into pure Abstract Syntax Tree (AST) rules. It generates multidimensional-ready equations via `Symbolics.jl` to ensure zero-allocation runtime performance.

## 2. Stage 1: The Numerical Core (Mathematics)

### 2.1 Dynamic Smoothness Indicators ($\beta_k$)
Standard WENO implementations often rely on hardcoded fractional constants derived from uniform grid spacing assumptions. On highly clustered grids, this geometric assumption collapses. This engine dynamically derives the exact, scaled smoothness indicators based on local sub-stencil geometry ($h_1$, $h_2$).

### 2.2 Shi-Hu-Shu Negative Weight Regularization
In highly clustered non-uniform grids, ideal Lagrange weights ($d_k$) can mathematically become negative. This prototype implements the **Shi-Hu-Shu (2002) Weight Splitting** technique. By splitting weights into positive and negative components with proper mass scaling, the engine maintains high-order accuracy while strictly obeying the Partition of Unity ($\sum \omega_k = 1.0$).

## 3. Stage 2: The Symbolic Codegen Engine (Architecture)

To seamlessly integrate with `PDEBase.jl`'s finite difference rules generation without relying on direct Cartesian loops, the `weno_symbolic.jl` engine provides:

* **Zero-Allocation Runtime:** By generating pure AST expressions and utilizing `build_function`, the resulting machine code executes the entire non-linear flux formulation with strictly **0 bytes of memory allocation**.
* **Type Stability & GPU Readiness:** Replaces standard `Float64` arithmetic with type-stable integer and rational scaling (`1//1000000`), guaranteeing native promotion for `Float32` GPU arrays without performance degradation.
* **Auto-Differentiability (AD):** The symbolic AST evaluates cleanly with `ForwardDiff.Dual` numbers, ensuring full compatibility with implicit/stiff ODE solvers.

## 4. Engineering & Validation Suites

The prototype is guarded by a comprehensive, multi-layered `Test` suite:

* **Mathematical Invariants (`test_weno.jl`):** Verifies the Partition of Unity and numerical splitting bounds.
* **MMS Convergence (`weno_convergence_test.jl`):** Mathematically proves that the splitting technique maintains the theoretical **$O(\Delta x^3)$ convergence order** on highly stretched grids.
* **Rigorous Validation Suite (`test_weno_symbolic.jl`):** A strict profiling suite that enforces zero-allocations, validates `ForwardDiff` Jacobian compatibility, and tests for graceful `NaN` propagation and `ArgumentError` boundary fallbacks.

## 5. How to Test the Prototype

You can interact with the prototype directly from the terminal.

**To run the Symbolic Codegen Profiling & AD Verification (Zero-Allocation & AD):**
```bash
julia test_weno_symbolic.jl
```

**To run the Strict Numerical Unit Tests:**
```bash
julia test_weno.jl
```

**To run the $O(\Delta x^3)$ Convergence Verification (MMS):**
```bash
julia weno_convergence_test.jl
```
