"""
    MOLFiniteDifference(dxs, time=nothing;
                        approx_order = 2, advection_scheme = UpwindScheme(),
                        grid_align = CenterAlignedGrid(), kwargs...)

A discretization algorithm.

## Arguments

- `dxs`: A vector of pairs of parameters to the grid step in this dimension, i.e. `[x=>0.2, y=>0.1]`.
    For a non-uniform rectilinear grid, replace any or all of the step sizes with the grid you'd like to
    use with that variable, must be an `AbstractVector` but not a `StepRangeLen`.
- `time`: Your choice of continuous variable, usually time. If `time = nothing`, then discretization
    yields a `NonlinearProblem`. Defaults to `nothing`.

## Keyword Arguments

- `approx_order`: The order of the derivative approximation.
- `advection_scheme`: The scheme to be used to discretize advection terms, i.e. first order spatial derivatives and associated coefficients. Defaults to `UpwindScheme()`. WENOScheme() is also available, and is more stable and accurate at the cost of complexity.
- `grid_align`: The grid alignment types. See [`CenterAlignedGrid`](@ref) and [`EdgeAlignedGrid`](@ref).
- `use_ODAE`: If `true`, the discretization will use the `ODAEproblem` constructor.
    Defaults to `false`.
- `kwargs`: Any other keyword arguments you want to pass to the `ODEProblem`.

"""
struct MOLFiniteDifference{G, D} <: AbstractEquationSystemDiscretization
    dxs::Any
    time::Any
    approx_order::Int
    advection_scheme::Any
    grid_align::G
    should_transform::Bool
    use_ODAE::Bool
    disc_strategy::D
    useIR::Bool
    callbacks::Any
    kwargs::Any
end

# Constructors. If no order is specified, both upwind and centered differences will be 2nd order
function MOLFiniteDifference(
        dxs, time = nothing; approx_order = 2,
        advection_scheme = UpwindScheme(), grid_align = CenterAlignedGrid(),
        discretization_strategy = ScalarizedDiscretization(),
        upwind_order = nothing, should_transform = true,
        use_ODAE = false, useIR = true, callbacks = [], kwargs...
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

    return MOLFiniteDifference{typeof(grid_align), typeof(discretization_strategy)}(
        dxs, time, approx_order, advection_scheme, grid_align, should_transform,
        use_ODAE, discretization_strategy, useIR, callbacks, kwargs
    )
end

PDEBase.get_time(disc::MOLFiniteDifference) = disc.time
