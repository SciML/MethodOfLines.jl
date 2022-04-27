struct MOLFiniteDifference{G} <: DiffEqBase.AbstractDiscretization
    dxs
    time
    approx_order::Int
    upwind_order::Int
    grid_align::G
end

# Constructors. If no order is specified, both upwind and centered differences will be 2nd order
function MOLFiniteDifference(dxs, time=nothing; approx_order = 2, upwind_order = 1, grid_align=CenterAlignedGrid())

    if approx_order % 2 != 0
        @warn "Discretization approx_order must be even, rounding up to $(approx_order+1)"
    end
    @assert approx_order >= 1 "approx_order must be at least 1"
    @assert upwind_order >= 1 "upwind_order must be at least 1"

    @assert (t isa Num | t isa Nothing) "time must be a Num, or Nothing. See docs for MOLFiniteDifference."

    return MOLFiniteDifference{typeof(grid_align)}(dxs, time, approx_order, upwind_order, grid_align)
end
