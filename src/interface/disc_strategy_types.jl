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
that their stencil differs from the translation-invariant interior stencil. Equations
containing patterns with no slice representation (WENO or functional advection schemes,
nonlinear or spherical Laplacians, integrals, mixed derivatives, interface/periodic
boundary conditions, staggered grids, boundary values appearing in interior equations)
automatically fall back to pointwise scalar equations, so results are always identical
to `ScalarizedDiscretization`.

Pass as `discretization_strategy` to [`MOLFiniteDifference`](@ref).
"""
struct ArrayDiscretization <: AbstractDiscretizationStrategy end
