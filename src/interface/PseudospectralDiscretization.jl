abstract type AbstractSpectralScheme end

"""
    ChebyshevCollocation(n)

Chebyshev–Lobatto pseudospectral collocation with `n` grid points on the domain
`[a, b]`, including both endpoints. Spatial derivatives of order `d` are
discretized with the differentiation matrix of the degree `n - 1` polynomial
interpolant through the grid values (equivalently, the maximal-stencil Fornberg
weights on the Lobatto nodes).

Use this scheme for directions with non-periodic (truncating) boundary
conditions such as Dirichlet or Neumann conditions.
"""
struct ChebyshevCollocation <: AbstractSpectralScheme
    n::Int
    function ChebyshevCollocation(n)
        n >= 2 || throw(ArgumentError("ChebyshevCollocation requires at least 2 collocation points, got $n"))
        return new(n)
    end
end

"""
    FourierCollocation(n)

Fourier pseudospectral collocation with `n` distinct equispaced grid points on
the periodic domain `[a, b)`. The grid stores `n + 1` points, with the upper
endpoint identified with the lower endpoint by the periodic boundary
condition `u(t, a) ~ u(t, b)`.

Spatial derivatives of order `d` are discretized with the differentiation
matrix of the trigonometric interpolant, computed as the `d`th power of the
first-derivative matrix.

This scheme requires a periodic boundary condition in the associated
direction; conversely, periodic directions must use `FourierCollocation`.
"""
struct FourierCollocation <: AbstractSpectralScheme
    n::Int
    function FourierCollocation(n)
        n >= 4 || throw(ArgumentError("FourierCollocation requires at least 4 collocation points, got $n"))
        return new(n)
    end
end

"""
    PseudospectralDiscretization(dxs, time = nothing; kwargs...)

A pseudospectral (collocation) discretization algorithm.

Each spatial independent variable is discretized on a collocation grid, and
every spatial derivative `Differential(x)^d` is replaced by the dense
differentiation matrix of the spectral interpolant in that direction, applied in
physical (grid) space; no transform to spectral coefficients is taken. Boundary
conditions replace the equation at the boundary nodes, exactly as in
[`MOLFiniteDifference`](@ref), with derivatives in a condition taken from the
boundary row of the differentiation matrix. Nested derivative terms such as
`Differential(x)(a(u) * Differential(x)(u))` are evaluated directly by
collocation, so no PDE-system transformation is needed.

The interior of each PDE is emitted as one symbolic array equation over slices of
the discretized variables, with each derivative an opaque operator holding its
differentiation matrix, so the number of symbolic equations and the size of the
generated code are independent of the resolution in one and two spatial
dimensions. See the [pseudospectral](@ref pseudospectral) documentation page for
the boundary conditions each grid type accepts and the scaling.

# Arguments

- `dxs`: A vector of pairs mapping each independent variable to a collocation
  specification:
  - `x => n::Integer` or `x => ChebyshevCollocation(n)` discretizes `x` on `n`
    Chebyshev–Lobatto points. Use for non-periodic directions.
  - `x => FourierCollocation(n)` discretizes `x` on `n` distinct equispaced
    points and requires a periodic boundary condition `u(t, a) ~ u(t, b)` in
    that direction. Higher-order periodic matching conditions such as
    `Differential(x)(u(t, a)) ~ Differential(x)(u(t, b))` are redundant - they
    are satisfied identically by the trigonometric interpolant - and are
    skipped.
  - `x => grid::AbstractVector` uses a custom set of `n` collocation nodes
    (polynomial-interpolation derivatives via Fornberg weights).
- `time`: The continuous variable, usually time. If `time = nothing`,
  discretization yields a `NonlinearProblem`. Defaults to `nothing`.

# Keywords

- `kwargs`: Additional keyword arguments passed to the generated problem.

# Example

```julia
using ModelingToolkit, MethodOfLines

@parameters t x
@variables u(..)
Dt = Differential(t)
Dxx = Differential(x)^2

# Chebyshev collocation on 32 points
discretization = PseudospectralDiscretization([x => 32], t)

# Fourier collocation on 64 distinct points, for a periodic domain
discretization = PseudospectralDiscretization([x => FourierCollocation(64)], t)
```

# Limitations

- Multi-domain (interface) boundary conditions are not supported; only
  same-variable periodic conditions may be used with `FourierCollocation`.
- `Integral` terms are not supported.
- Unlike `MOLFiniteDifference`, there is no upwinding or special scheme
  selection: all derivative orders in a direction share the same spectral
  differentiation matrix.
- Derivatives are dense matrix products, `O(n^2)` per application in one
  dimension, and the Jacobian is dense.
"""
struct PseudospectralDiscretization <: AbstractEquationSystemDiscretization
    dxs::Any
    time::Any
    kwargs::Any
end

function PseudospectralDiscretization(dxs, time = nothing; kwargs...)
    @assert (time isa Num) | (time isa Nothing) "time must be a Num, or Nothing - got $(typeof(time)). See docs for PseudospectralDiscretization."

    dxs = Dict{Any, Any}(dxs)
    for (x, spec) in dxs
        spec isa Integer && (dxs[x] = ChebyshevCollocation(spec))
        dxs[x] isa Union{AbstractSpectralScheme, AbstractVector} ||
            throw(ArgumentError("Invalid collocation specification $(dxs[x]) for $x. Pass an Integer, ChebyshevCollocation(n), FourierCollocation(n), or a grid vector."))
    end
    return PseudospectralDiscretization(dxs, time, kwargs)
end

PDEBase.get_time(disc::PseudospectralDiscretization) = disc.time

const MOLDiscretization = Union{MOLFiniteDifference, PseudospectralDiscretization}
