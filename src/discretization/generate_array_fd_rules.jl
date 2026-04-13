"""
Array-level finite difference rule generation for `ArrayDiscretization`.

For PDEs on uniform grids, stencil weights are identical at every interior
point in each dimension.  This module pre-computes them once and builds a
single ArrayOp expression using N symbolic index variables `_i1, _i2, ...`
(one per spatial dimension).  For non-uniform grids, per-point weights are
stored in Const-wrapped matrices indexed by symbolic grid indices.  The
ArrayOp is a genuine symbolic array operation that, when ModelingToolkit
supports native array compilation, will compile to a single vectorized loop.
Until then, MTK's `flatten_equations` scalarizes it into individual
per-point equations, preserving correctness.

Supported ArrayOp patterns:
- Centred (even-order) derivatives on uniform and non-uniform grids
- Upwind (odd-order) derivatives with UpwindScheme on uniform and non-uniform grids
- Staggered grid (odd-order) derivatives on uniform grids
- WENO (Jiang-Shu) first-order derivatives on uniform grids
- Mixed cross-derivatives on uniform and non-uniform grids
- Nonlinear Laplacian `Dx(expr * Dx(u))` on uniform and non-uniform grids
- Spherical Laplacian `r^{-2} * Dr(r^2 * Dr(u))` on uniform and non-uniform grids

Generic user-defined `FunctionalScheme` falls back to per-point computation
via `discretize_equation_at_point` from the scalar path, which supports ALL
scheme types.

This file used to be a single ~2900-LOC monolith; it is now split into the
thematic files under `array_fd/`.  Include order below matters only for
struct-dispatched functions (types must be defined before functions
dispatching on them).  Everything else is resolved at runtime.
"""

# Stencil data structures: centred / upwind / nonlinlap / WENO / staggered.
include("array_fd/stencil_info.jl")

# Central context bag (ArrayOpContext, StencilCaches), Const alias, and the
# `_tap_expr` tap-building helpers used by every rule builder downstream.
include("array_fd/context.jl")

# All `precompute_*` caches, including full-interior mode and its data
# structures (FullInteriorStencilInfo et al.).
include("array_fd/precompute.jl")

# PDE term-pattern detection (nonlinlap, spherical Laplacian).
include("array_fd/pattern_detection.jl")

# Validation: symbolic equation comparison + multi-point template sampling.
include("array_fd/validation.jl")

# Orchestrator: `generate_array_interior_eqs`.
include("array_fd/interior_driver.jl")

# Low-level substitution helpers: `_substitute_terms`, `_wrap_periodic_idx`,
# and the derivative-detection wrappers around `PDEBase.differential_order`.
include("array_fd/substitution_helpers.jl")

# Core ArrayOp builder: `_build_interior_arrayop`.
include("array_fd/arrayop_builder.jl")

# Per-scheme rule builders.  Each file defines one or two helpers plus the
# corresponding `_build_*_rules` entry point called from `_build_interior_arrayop`.
include("array_fd/rules_upwind.jl")
include("array_fd/rules_mixed.jl")
include("array_fd/rules_nonlinlap.jl")
include("array_fd/rules_spherical.jl")
include("array_fd/rules_weno.jl")
