"""
    MOLFiniteDifference(dxs, time=nothing;
                        approx_order = 2, upwind_order = 1,
                        grid_align = CenterAlignedGrid(), kwargs...)

A discretization algorithm.

## Arguments

  - `dxs`: A vector of pairs of parameters to the grid step in this dimension, i.e. `[x=>0.2, y=>0.1]`.
    For a non uniform rectilinear grid, replace any or all of the step sizes with the grid you'd like to
    use with that variable, must be an ``AbstractVector`` but not a ``StepRangeLen``.
  - `time`: Your choice of continuous variable, usually time. If `time = nothing`, then discretization
    yeilds a `NonlinearProblem`.

## Keyword Arguments

  - `approx_order`: The order of the derivative approximation.
  - `upwind_order`: The order of the upwind scheme. Currently unstable at any value other than 1.
  - `grid_align`: The grid alignment types. See [`CenterAlignedGrid()`](@ref) and [`EdgeAlignedGrid()`](@ref).
  - `kwargs`: Any other keyword arguments you want to pass to the discretization.

"""
struct MOLFiniteDifference{G} <: DiffEqBase.AbstractDiscretization
    dxs
    time
    approx_order::Int
    upwind_order::Int
    grid_align::G
    kwargs
end

# Constructors. If no order is specified, both upwind and centered differences will be 2nd order
function MOLFiniteDifference(dxs, time=nothing; approx_order = 2, upwind_order = 1, grid_align=CenterAlignedGrid(), kwargs...)

    if approx_order % 2 != 0
        @warn "Discretization approx_order must be even, rounding up to $(approx_order+1)"
    end
    @assert approx_order >= 1 "approx_order must be at least 1"
    @assert upwind_order >= 1 "upwind_order must be at least 1"

    @assert (time isa Num) | (time isa Nothing) "time must be a Num, or Nothing - got $(typeof(time)). See docs for MOLFiniteDifference."

    return MOLFiniteDifference{typeof(grid_align)}(dxs, time, approx_order, upwind_order, grid_align, kwargs)
end
