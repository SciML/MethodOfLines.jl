# [Array Discretization Coverage](@id array_coverage)

[`ArrayDiscretization`](@ref) emits each region of the discretized system as a single
symbolic array equation over slices, instead of one scalar equation per grid point. Where
a pattern has no slice representation, the affected equation falls back to the pointwise
form, which produces the same numbers — so this page is a map of *where the array form
currently applies*, not a list of things that fail.

Use [`StrictArrayDiscretization`](@ref) to turn any such fallback into an error, which is
the quickest way to find out whether a given system is fully in array form.

## What is covered

| Region | Status | Equations |
|---|---|---|
| Interior, centered even-order derivatives | ✅ array | 1 per PDE |
| Interior, upwind odd-order derivatives | ✅ array | 1 per PDE |
| Boundary faces (Dirichlet / Neumann / Robin) | ✅ array | 1 per face |
| Periodic / self-interface boundaries | ✅ array | 1 per face + O(1) seam points |
| Corner and edge regions | ✅ array | 1 per box |
| Near-boundary interior points ("frame") | pointwise | see below |

Periodic directions need no special interior handling: the scalar path never takes a
boundary-stencil branch there (both ends report as interfaces), so the interior stencil
already applies across the whole axis. The few points whose taps cross the seam are
emitted individually, which is `O(1)` in grid size, not `O(n)`.

Both uniform and non-uniform grids are covered, in any number of dimensions, except that
periodic support requires a uniform grid in the periodic direction (the scalar path
rejects non-uniform interface boundaries outright). The
resulting system has `3^N` equations for an `N`-dimensional problem, independent of grid
resolution: one per combination of {below the interior, inside it, above it} per axis.

The frame — interior points close enough to a boundary that their stencil differs from
the translation-invariant one — is pointwise under every strategy. This is structural
rather than a gap: those points genuinely use different stencils. At `approx_order = 2`
with Dirichlet conditions the frame is empty; at higher orders it costs a small,
resolution-independent number of equations per boundary.

## What still needs an array form

Each entry below falls back to pointwise discretization for the whole equation. They are
listed roughly in increasing order of difficulty.

### Mixed derivatives — `Dx(Dy(u))`

`generate_mixed_rules` applies a tensor-product stencil at each point. The stencil is
translation invariant across the interior, so this is the closest to the existing
interior work: it needs a two-dimensional analogue of the shifted-slice sum, offsetting
along two axes at once rather than one.

### Nonlinear Laplacian — `Dx(a(u) * Dx(u))`

`cartesian_nonlinear_laplacian` evaluates the coefficient at half-offset points and forms
a half-offset centered difference. The stencil is translation invariant, so the obstacle
is not the difference itself but building the interleaved half-offset coefficient
expression over slices.

### Spherical Laplacian — `r^-2 * Dr(r^2 * Dr(u))`

Same shape as the nonlinear Laplacian plus the `r`-dependent weighting, which is a grid
value and already expressible as an array. Likely follows immediately from whatever
approach works for the nonlinear Laplacian.

### Interface boundaries between two domains

Self-periodic boundaries (`u(t, 0) ~ u(t, 1)`, the same variable and independent variable
at both ends) are supported. Genuine two-domain interfaces — where the two sides are
different variables — are not: their stencil taps land in another variable's array, so
the slice arithmetic does not carry over. These fall back cleanly.

### Boundary values inside interior equations

An interior equation referring to `u(t, 1)` mixes a full-interior slice with a single
boundary value. Broadcasting a scalar against a slice already works, so the missing piece
is recognising the boundary value and substituting the corresponding element rather than
falling back on the whole equation.

### WENO and other functional advection schemes

WENO's nonlinear smoothness indicators make the coefficients solution-dependent at every
point. The stencil taps are still uniform, so a slice form is plausible — every
smoothness indicator is itself a broadcast over slices — but it is a substantially larger
expression than the linear schemes and worth benchmarking before assuming it pays off.

### Integrals

`Integral(x in ClosedInterval(...))(u)` couples every point in the integration direction,
so the result is a reduction rather than a stencil. Whole-domain integrals reduce to a
scalar (a weighted sum over a slice, which is expressible); cumulative integrals produce
a running sum along the axis, which needs a different primitive than shifted slices.

### Staggered grids

`StaggeredGrid` selects stencils by the alignment of each variable, and the array path
declines the whole system today. The alignment is fixed per variable, so the selection is
constant across the interior — this is likely more mechanical than conceptual.

### Variables of differing dimensionality

Systems mixing, say, `u(t, x, y)` with `v(t, x)` need the lower-dimensional variable
broadcast against the higher-dimensional slice. Shape handling is the whole problem here.

## Non-goals

Some pointwise output is inherent and is not a gap to close:

  - The frame, as described above.
  - Single-point regions. A 1D boundary is one point, and a corner in any dimension is one
    point; a one-element slice equation is a more convoluted spelling of the scalar one,
    so those stay scalar deliberately.

## Checking a system

```julia
disc = MOLFiniteDifference([x => dx], t;
    discretization_strategy = StrictArrayDiscretization())
sys, tspan = symbolic_discretize(pdesys, disc)
```

This raises `MethodOfLines.ArrayDiscretizationError` naming the equation and the reason if
any part of the system cannot be represented in array form. With `ArrayDiscretization` the
same information is available at debug level:

```julia
ENV["JULIA_DEBUG"] = "MethodOfLines"
```
