"""
    MOLFiniteDifference(dxs, time=nothing;
                        approx_order = 2, advection_scheme = UpwindScheme(),
                        grid_align = CenterAlignedGrid(), kwargs...)

A discretization algorithm.

# Arguments

- `dxs`: A vector of pairs of parameters to the grid step in this dimension, i.e.
  `[x => 0.2, y => 0.1]`. For a non-uniform rectilinear grid, replace any or all of the
  step sizes with the grid to use with that variable. It must be an `AbstractVector`,
  but not a `StepRangeLen`.
- `time`: The continuous variable, usually time. If `time = nothing`, discretization
  yields a `NonlinearProblem`. Defaults to `nothing`.

# Keywords

- `approx_order`: The order of the derivative approximation.
- `advection_scheme`: The scheme used to discretize first-order spatial derivatives
  and associated coefficients. Defaults to `UpwindScheme()`. `WENOScheme()` is more
  stable and accurate at the cost of complexity.
- `grid_align`: The grid alignment value. Use [`center_align`](@ref),
  [`edge_align`](@ref), or `StaggeredGrid()` as appropriate for the discretization.
- `kwargs`: Additional keyword arguments passed to the generated problem.

# Fields

- `dxs`: A dictionary mapping each discretized independent variable to an integer
  grid size, a spacing, or an explicit grid.
- `time`: The independent variable left undiscretized, or `nothing` for a fully
  discretized system.
- `approx_order`: The requested finite-difference approximation order.
- `advection_scheme`: The scheme used for first-order spatial derivatives.
- `grid_align`: The grid alignment marker.
- `should_transform`: Whether supported symbolic transformations are applied before
  discretization.
- `useIR`: Whether ModelingToolkit's intermediate representation is used.
- `callbacks`: Symbolic discretization callbacks.
- `kwargs`: Additional keyword arguments forwarded to the generated problem.

# Example

```julia
using ModelingToolkit
using MethodOfLines

@parameters t x
discretization = MOLFiniteDifference([x => 0.1], t)
```

"""
struct MOLFiniteDifference{G} <: AbstractEquationSystemDiscretization
    dxs::Any
    time::Any
    approx_order::Int
    advection_scheme::Any
    grid_align::G
    should_transform::Bool
    useIR::Bool
    callbacks::Any
    kwargs::Any
end

# Constructors. If no order is specified, both upwind and centered differences will be 2nd order
function MOLFiniteDifference(
        dxs, time = nothing; approx_order = 2,
        advection_scheme = UpwindScheme(), grid_align = CenterAlignedGrid(),
        upwind_order = nothing, should_transform = true,
        useIR = true, callbacks = [], kwargs...
    )
    :discretization_strategy in keys(kwargs) && throw(
        ArgumentError(
            "`discretization_strategy` was removed in v1; MethodOfLines always uses array-form discretization with automatic pointwise fallback."
        )
    )
    :use_ODAE in keys(kwargs) && throw(
        ArgumentError(
            "`use_ODAE` was removed in v1; use `symbolic_discretize`, `mtkcompile`, and `ODEProblem` when an explicit ODE problem is required."
        )
    )
    if upwind_order !== nothing
        @warn "`upwind_order` no longer does anything, and will be removed in a future release. See the docs for the current interface."
    end
    if approx_order % 2 != 0
        @warn "Discretization approx_order must be even, rounding up to $(approx_order + 1)"
    end
    @assert approx_order >= 1 "approx_order must be at least 1"

    @assert (time isa Num) | (time isa Nothing) "time must be a Num, or Nothing - got $(typeof(time)). See docs for MOLFiniteDifference."

    if (
            grid_align == StaggeredGrid() &&
                !(:edge_aligned_var in keys(kwargs))
        )
        @warn "when using StaggeredGrid(), you must set 'edge_aligned_var' keyword arg"
    end

    dxs = dxs isa Dict ? dxs : Dict(dxs)

    return MOLFiniteDifference{typeof(grid_align)}(
        dxs, time, approx_order, advection_scheme, grid_align, should_transform,
        useIR, callbacks, kwargs
    )
end

PDEBase.get_time(disc::MOLFiniteDifference) = disc.time
