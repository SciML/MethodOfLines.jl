# Discretization strategies
# -------------------------
abstract type AbstractDiscretizationStrategy end

"""
    ScalarizedDiscretization()

The default discretization strategy: discretize the `PDESystem` into one scalar equation
per interior grid point.

Pass as `discretization_strategy` to [`MOLFiniteDifference`](@ref).
"""
struct ScalarizedDiscretization <: AbstractDiscretizationStrategy end

"""
    ArrayDiscretization()

Discretize the interior of each PDE into a single symbolic array equation over slices of
the discretized (array) variables, e.g. for the 1D heat equation with second order
approximation:

```julia
D(u[2:n-1]) - (u[1:n-2] .- 2u[2:n-1] .+ u[3:n]) ./ dx^2 ~ 0
```

This keeps the number of symbolic equations independent of the grid resolution, which
scales much better to large systems during symbolic processing, and gives compilers that
consume array equations the structure needed to generate looped code.

Boundary, extrapolation and corner equations are generated pointwise as in
[`ScalarizedDiscretization`](@ref), as are interior points close enough to a boundary
that their stencil differs from the translation-invariant interior stencil. Boundary
values appearing in an interior equation (e.g. `u(t, 1)`) are substituted for the
corresponding array element or face slice on every array box, including size-1 wrap
boxes. Nonlinear Laplacians `Dx(a(u) * Dx(u))` and staggered grids are supported in
slice form. Equations containing patterns with no slice representation (WENO or functional
advection schemes, spherical Laplacians, integrals, mixed derivatives,
interfaces joining two different variables, derivatives of boundary
values, time-literal references such as `u(0, x)`, boundary values on edge-aligned
grids, stationary systems) automatically fall back to pointwise scalar equations,
matching `ScalarizedDiscretization` for those equations. Where the array form is used,
numerics match the scalar path whenever the scalar path can express the same
boundary-value substitutions; the array path also substitutes periodic-face and
free-standing-corner references that scalar `boundaryvalfuncs` currently leave symbolic.

Pass as `discretization_strategy` to [`MOLFiniteDifference`](@ref). Use
[`StrictArrayDiscretization`](@ref) to make the fallback an error instead.
"""
struct ArrayDiscretization <: AbstractDiscretizationStrategy end

"""
    StrictArrayDiscretization()

Like [`ArrayDiscretization`](@ref), but raises an error instead of silently falling back
to pointwise discretization when an equation contains a pattern with no slice
representation.

Useful for testing and for work that depends on getting the array form: with
`ArrayDiscretization` an unsupported pattern still discretizes correctly, just
pointwise, which is easy to miss. This makes that visible.

The error covers whole-equation fallback only. Boundary, corner and extrapolation
equations, and interior points near a boundary whose stencil differs from the
translation-invariant one, are pointwise under either strategy — they are irregular by
construction, not unsupported — so they are not errors here.

Pass as `discretization_strategy` to [`MOLFiniteDifference`](@ref).
"""
struct StrictArrayDiscretization <: AbstractDiscretizationStrategy end

# The two array strategies produce the same discretization and differ only in whether an
# unrepresentable pattern falls back or raises, so the machinery dispatches on both.
const AnyArrayDiscretization = Union{ArrayDiscretization, StrictArrayDiscretization}

isstrict(::AbstractDiscretizationStrategy) = false
isstrict(::StrictArrayDiscretization) = true
