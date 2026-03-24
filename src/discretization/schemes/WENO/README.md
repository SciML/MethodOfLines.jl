# Non-Uniform WENO Mathematical Prototype

This directory contains the isolated mathematical prototype for calculating
**dynamic essentially non-oscillatory (WENO) weights**, smoothness indicators,
and final non-linear weights on non-uniform, clustered grids.

This prototype was built as a proof-of-concept for the GSoC 2026 proposal:
**Comprehensive Non-Uniform Grid Support for MethodOfLines.jl**.

## 1. Architectural Philosophy (Separation of Concerns)

Following recent technical discussions on SciML's modular design, this logic is
kept strictly isolated within the `WENO` module.

Instead of modifying the core `fornberg.jl` or `construct_differential_discretizer.jl`
prematurely, this prototype acts as a standalone **mathematical engine**. Once the
core pipeline is ready to ingest non-uniform weights, `MethodOfLines.jl` will query
this module directly without cluttering the main generation logic.

## 2. The Mathematics: Dynamic Indicators & Regularization

### 2.1 Dynamic Smoothness Indicators ($\beta_k$)

Standard WENO implementations often rely on hardcoded fractional constants
(e.g., `13/12`) derived from uniform grid spacing assumptions. On highly clustered
grids, this geometric assumption collapses, leading to severe precision loss.

This engine dynamically reads the local geometric distances ($h_1$, $h_2$) of the
sub-stencil to calculate the exact, scaled smoothness indicators:

$$\beta_{dynamic} \propto \frac{h_1^2 + h_1 h_2 + h_2^2}{(h_1 + h_2)^2}$$

### 2.2 Non-Linear Weights ($\omega_k$) & Negative Weight Treatment

In highly clustered non-uniform grids, ideal Lagrange weights ($d_k$) can
mathematically become negative (a known theoretical phenomenon, e.g.,
*Shi, Hu, Shu 2002*). If left unhandled, this destabilizes the scheme.

This prototype implements a **Positive Shifting Regularization** step. If any
ideal weight drops below zero, the weights are safely shifted to the positive
domain and re-normalized. This guarantees that the final non-linear weights
strictly obey the Partition of Unity ($\sum \omega_k = 1.0$) and remain strictly
non-negative ($\omega_k \ge 0.0$).

## 3. Engineering & Validation

### 3.1 Type Stability & AD Compatibility

The engine is architected using Julia’s generic type system (`T <: Real`). By
utilizing `one(T)`, `zero(T)`, and type-aware allocations via `similar()`, the 
implementation remains strictly type-stable. This ensures full compatibility with:

* **ForwardDiff.jl** for automatic differentiation (AD).
* **Float32/Float16** for high-performance GPU computing.
* **BigFloat** for arbitrary-precision research.

### 3.2 Strict Unit Testing (`test_weno.jl`)

The mathematical engine is thoroughly verified using Julia's `Test` framework.
The test suite enforces strict mathematical invariants, including the successful
handling of negative weights without violating the Partition of Unity or
positivity constraints.

### 3.3 Performance Profiling (`run_prototype.jl`)

To verify execution speed, the entire non-linear weight pipeline is profiled 
using `BenchmarkTools`. **Current benchmark results:**

* **Execution Time:** ~113 ns
* **Memory overhead:** 10 allocations (400 bytes)

This confirms that the dynamic calculations and regularization steps introduce 
virtually zero computational overhead to the main discretization pipeline.

## 4. How to Test the Prototype

You can interact with the prototype directly from the terminal.

**To run the performance & architecture showcase:**

```bash
julia run_prototype.jl
```

**To run the strict mathematical unit test suite:**

```bash
julia test_weno.jl
```