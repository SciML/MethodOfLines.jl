struct MOLFiniteDifference{G} <: DiffEqBase.AbstractDiscretization
    dxs
    time
    approx_order::Int
    grid_align::G
end

# Constructors. If no order is specified, both upwind and centered differences will be 2nd order
function MOLFiniteDifference(dxs, time=nothing; approx_order = 2, grid_align=CenterAlignedGrid())
    
    if approx_order % 2 != 0
        warn("Discretization approx_order must be even, rounding up to $(approx_order+1)")
    end
    return MOLFiniteDifference{typeof(grid_align)}(dxs, time, approx_order, grid_align)
end