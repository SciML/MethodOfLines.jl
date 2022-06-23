struct MOLFiniteDifference{G} <: DiffEqBase.AbstractDiscretization
    dxs::Any
    time::Any
    approx_order::Int
    upwind_order::Int
    grid_align::G
    kwargs::Any
end

# Constructors. If no order is specified, both upwind and centered differences will be 2nd order
function MOLFiniteDifference(
    dxs,
    time = nothing;
    approx_order = 2,
    upwind_order = 1,
    grid_align = CenterAlignedGrid(),
    kwargs...,
)

    if approx_order % 2 != 0
        @warn "Discretization approx_order must be even, rounding up to $(approx_order+1)"
    end
    @assert approx_order >= 1 "approx_order must be at least 1"
    @assert upwind_order >= 1 "upwind_order must be at least 1"

    @assert (time isa Num) | (time isa Nothing) "time must be a Num, or Nothing - got $(typeof(time)). See docs for MOLFiniteDifference."

    return MOLFiniteDifference{typeof(grid_align)}(
        dxs,
        time,
        approx_order,
        upwind_order,
        grid_align,
        kwargs,
    )
end
