struct MOLFiniteDifference{G} <: DiffEqBase.AbstractDiscretization
    dxs
    time
    overlap_map
    approx_order::Int
    upwind_order::Int
    grid_align::G
end

# Constructors. If no order is specified, both upwind and centered differences will be 2nd order
function MOLFiniteDifference(dxs, time=nothing; approx_order = 2, upwind_order = 1, grid_align=CenterAlignedGrid(), overlap_map = [])

    if approx_order % 2 != 0
        @warn "Discretization approx_order must be even, rounding up to $(approx_order+1)"
    end
    @assert approx_order >= 1 "approx_order must be at least 1"
    @assert upwind_order >= 1 "upwind_order must be at least 1"

    @assert (time isa Num) | (time isa Nothing) "time must be a Num, or Nothing - got $(typeof(time)). See docs for MOLFiniteDifference."

    _overlap_map = Dict(mapreduce(vcat, keys(overlap_map)) do x
        [y => x for y in overlap_map[x]]
    end)

    @assert collect(keys(_overlap_map)) == unique(collect(keys(_overlap_map))) "Variables cannot be subsets of more than one domain, check overlap_map."

    return MOLFiniteDifference{typeof(grid_align)}(dxs, time, overlap_map, approx_order, upwind_order, grid_align)
end
