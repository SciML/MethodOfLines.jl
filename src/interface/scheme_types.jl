abstract type AbstractScheme end

"""
    UpwindScheme([approx_order])

Upwind finite difference scheme for advection terms.
"""
struct UpwindScheme <: AbstractScheme
    order::Any
    function UpwindScheme(approx_order = 1)
        return new(approx_order)
    end
end

extent(scheme::UpwindScheme, dorder) = 0 # dorder + scheme.order - 1

extent(scheme::AbstractScheme, dorder, dx) = extent(scheme, dorder)

# Functional Schemes

"""
# FunctionalScheme
```
F = FunctionalScheme{interior_points, boundary_points}(interior, lower, upper, is_nonuniform, parameters; name)
```
A user definable scheme that takes a set of functions as input. The functions define the derivative at the interior, lower boundary, and upper boundary.

`lower` and `upper` should be vectors of functions. In general, `upper` and `lower` must be at least `floor(interior_points/2)` long. Where you have no good approximation for a derivative at the boundary, you can use `nothing` as a placeholder. MethodOfLines will then attempt to use an extrapolation here where necessary. Be warned that this can lead to instability.

The boundary functions define the derivative at their index in the function vector, numbering from the boundary. For example, if `boundary_points = 3`, the first function in the vector will define the derivative at the boundary, the second at the boundary plus one step, and the third at the boundary plus two steps.

The functions making up the scheme take the following inputs:

Functions must be of the form `f(u, p, t, deriv_iv, d_iv)`.

Where the function would branch on a value of u, p, t, or d_iv, use `ifelse` instead of standard conditionals.

For the interior, `u` takes a vector of dependent variable values in the direction of the derivative
of length `interior_points`. `interior_points` must be odd, as this function defines the derivative at the center of the input points.

For the lower and upper boundaries, `u` takes a vector of dependent variable values of length `boundary_points`. This will be the `boundary_points` number of points closest to the lower and upper boundary respectively.
`p` will take all parameter values in the order specified in the PDESystem, with the scheme's parameters prepended to the list.

`deriv_iv` takes a vector of independent variable values of the same support as for `u`, for the independent variable in the direction of the derivative.

If `is_nonuniform` is false, `d_iv` will take a scalar value of the stepsize between the points used to call the function in `u` and `deriv_iv`.

If `is_nonuniform` is true, the scheme must be able to accept `d_iv` as a vector of stepsizes between the points used to call the function in `u` and `deriv_iv`, therefore of length `length(u)-1`. A method should also be defined for the case where `d_iv` is a scalar, in which case the stepsizes are assumed to be uniform.
"""
struct FunctionalScheme{F, V1, V2, V3} <: AbstractScheme
    """
    The function that defines the scheme on the interior.
    """
    interior::F
    """
    The vector of functions that defines the scheme near the lower boundary.
    """
    lower::V1
    """
    The vector functions that defines the scheme near the upper boundary.
    """
    upper::V2
    """
    The number of interior points that the interior scheme takes as input
    """
    interior_points::Int
    """
    The number of boundary points that the lower and upper schemes take as input.
    """
    boundary_points::Int
    """
    Whether this scheme takes grid steps as input for the nonuniform case.
    """
    is_nonuniform::Bool
    """
    parameters for the scheme, should be a vector of numbers.
    """
    ps::V3
    """
    The name of the scheme.
    """
    name::String
end

function FunctionalScheme{ips, bps}(
        interior, lower, upper, is_nonuniform = false,
        ps = []; name = "FunctionalScheme"
    ) where {ips, bps}
    @assert ips % 2 == 1 "interior_points must be odd."
    return FunctionalScheme{
        typeof(interior), typeof(lower),
        typeof(upper), typeof(ps),
    }(
        interior, lower, upper,
        ips, bps, is_nonuniform, ps, name
    )
end

function extent(scheme::FunctionalScheme, dorder)
    @assert dorder == 1 "Only first order spatial derivatives are implemented for functional schemes."
    lower = length(findall(isnothing, scheme.lower))
    upper = length(findall(isnothing, scheme.upper))
    @assert lower == upper "Scheme must have symmetric extent; same number of placeholders in lower and upper boundary functions."
    return lower
end

"""
    HermiteScheme(; deriv_var = nothing)

Hermite finite differences: every spatial derivative of a variable `u` is approximated from
the values of `u` and of its first derivative at the stencil points (Fornberg 2021). Pass it
as `advection_scheme` to [`MOLFiniteDifference`](@ref).

On the points of MOL's centred stencil for a given `approx_order`, a Hermite stencil uses
twice as many values: with `m` points its order is `2m - d` for a derivative of order `d`,
about twice the `m - d` (or `m - d + 1`) of the ordinary stencil on the same points. The first derivative is carried as an
extra unknown, the derivative partner `uˍx = Dx(u)`, with its own evolution equation.

# Automatic partners

With `HermiteScheme()` the PDE system is transformed before discretization: for each
dependent variable `u` and spatial variable `x`, a partner `uˍx` is added with the
evolution equation `Dt(uˍx) ~ Dx(R)` from `u`'s equation `Dt(u) ~ R`, the initial condition
`Dx` of `u`'s, and boundary conditions derived from `u`'s:

  - Dirichlet data: at a boundary of `x`, `Dx(uˍx)` from `u`'s equation on the boundary
    (for a wave equation as a system, also from the other equations, so data on `u`
    alone are enough); along the boundary, the derivative of the data.
  - Neumann or Robin data: a condition on `uˍx` that replaces the one on `u`.
  - Periodic conditions carry over to the partners.

An equation `Dt(Dt(u)) ~ R` is first rewritten as `Dt(u) ~ uˍt`, `Dt(uˍt) ~ R`; it needs
the initial condition `Dt(u(t0, x)) ~ g` and Dirichlet, Neumann or periodic conditions on `u`.

In more than one dimension, the mixed derivative `u_xy` is approximated from the
partners as `δy uˍx + δx uˍy - δx δy u`, of twice the order of `δx δy u`.

# Explicit partners

`HermiteScheme(deriv_var = u => v)` (one spatial dimension) or
`HermiteScheme(deriv_var = [(u, x) => vx, (u, y) => vy])` (several) uses the given variables
as partners. The system is not transformed: the equations, initial and boundary conditions
of the partners are part of the `PDESystem`.

# Discretization

Away from the boundaries the derivatives use Hermite stencils on MOL's points. Near a
boundary the stencils are graded, so that the semi-discrete system is stable; the
boundaries then set the global order. Variables without a partner use MOL's stencils.
No upwinding is applied, also not to advection terms. `discretize` returns a `DAEProblem`: the
boundary conditions of the partners are algebraic equations.

# Limitations

These throw an `ArgumentError`:

  - periodic conditions or interfaces on a non-uniform grid;
  - interfaces between two variables;
  - time derivatives of third or higher order, and `Dt(Dt(u))` with Robin conditions or
    conditions containing time derivatives (write these as a system);
  - an automatic `HermiteScheme()` with `should_transform = false`.

A boundary without a condition (a free outflow of an advection equation) has a lower order
than the interior, which limits the global order for large `approx_order`.

# Example

```julia
using MethodOfLines, ModelingToolkit, DomainSets, OrdinaryDiffEq

@parameters t x
@variables u(..)
Dt, Dx = Differential(t), Differential(x)
domains = [t ∈ Interval(0.0, 0.25), x ∈ Interval(0.0, 1.0)]
disc = MOLFiniteDifference([x => 0.05], t; advection_scheme = HermiteScheme())

# heat equation, exact solution exp(-π²t) sin(πx)
eq = Dt(u(t, x)) ~ Dx(Dx(u(t, x)))
bcs = [u(0, x) ~ sin(π * x), u(t, 0) ~ 0, u(t, 1) ~ 0]
@named heat = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

prob = discretize(heat, disc)        # a DAEProblem with unknowns u and uˍx
sol = solve(prob; abstol = 1e-10, reltol = 1e-10)
maximum(abs, sol[u(t, x)][end, :] .- exp(-π^2 * 0.25) .* sin.(π .* sol[x]))

# wave equation with fixed ends, exact solution cos(πt) sin(πx)
eq = Dt(Dt(u(t, x))) ~ Dx(Dx(u(t, x)))
bcs = [u(0, x) ~ sin(π * x), Dt(u(0, x)) ~ 0, u(t, 0) ~ 0, u(t, 1) ~ 0]
@named wave = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

sol = solve(discretize(wave, disc); abstol = 1e-10, reltol = 1e-10)
maximum(abs, sol[u(t, x)][end, :] .- cos(π * 0.25) .* sin.(π .* sol[x]))
```

# Reference

B. Fornberg, An algorithm for calculating Hermite-based finite difference weights,
IMA Journal of Numerical Analysis 41 (2021), 801–813.
"""
struct HermiteScheme{D} <: AbstractScheme
    """Map from a variable (or `(u, x)` pair) to its derivative partner `v = Dx(u)`."""
    deriv_var::D
    """Whether the partners are introduced by transforming the system (`HermiteScheme()`)."""
    automatic::Bool
    """
    `(u, x)` pairs whose PDE has only even-order derivatives in `x`, filled by the automatic
    transformation; their stencils near `x` boundaries are wider (see
    `_hermite_boundary_stencil`).
    """
    diffusive::Set{Any}
end

HermiteScheme(deriv_var, automatic::Bool) = HermiteScheme(deriv_var, automatic, Set{Any}())

function HermiteScheme(; deriv_var = nothing)
    return if deriv_var === nothing
        HermiteScheme(Dict{Any, Any}(), true)
    elseif deriv_var isa Pair
        HermiteScheme(Dict(deriv_var), false)
    else
        HermiteScheme(Dict(deriv_var...), false)
    end
end

# Hermite uses the centered operator's one-sided boundary taps directly, so it
# does not remove additional points from the equation iteration domain.
extent(::HermiteScheme, dorder) = 0

function lower_extent(scheme::FunctionalScheme, dorder)
    @assert dorder == 1 "Only first order spatial derivatives are implemented for functional schemes."
    return length(findall(isnothing, scheme.lower))
end

function upper_extent(scheme::FunctionalScheme, dorder)
    @assert dorder == 1 "Only first order spatial derivatives are implemented for functional schemes."
    return length(findall(isnothing, scheme.upper))
end
