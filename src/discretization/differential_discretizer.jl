# Use DiffEqOperators to generate weights and calculate derivative orders
struct DifferentialDiscretizer{T, D1, S}
    approx_order::Int
    advection_scheme::S
    map::D1
    halfoffsetmap::Tuple{Dict,Dict}
    windmap
    interpmap::Dict{Num,DerivativeOperator}
    orders::Dict{Num,Vector{Int}}
    boundary::Dict{Num,DerivativeOperator}
end

function DifferentialDiscretizer(pdesys, s, discretization, orders)
    pdeeqs = pdesys.eqs isa Vector ? pdesys.eqs : [pdesys.eqs]
    bcs = pdesys.bcs isa Vector ? pdesys.bcs : [pdesys.bcs]
    approx_order = discretization.approx_order
    advection_scheme = discretization.advection_scheme
    upwind_order = advection_scheme isa UpwindScheme ? advection_scheme.order : 1

    differentialmap = Array{Pair{Num,DerivativeOperator},1}()
    nonlinlap_inner = []
    nonlinlap_outer = []
    windpos = []
    windneg = []
    interp = []
    boundary = []
    # TODO: Make sure that only nessecary orders are calculated, this is the lowest hanging performance fruit.
    for x in s.x̄
        orders_ = orders[x]
        _orders = Set(vcat(orders_, [1, 2]))

        if s.grid[x] isa StepRangeLen # Uniform grid case
            dx = s.dxs[x]

            nonlinlap_outer = push!(nonlinlap_outer, Differential(x) => CompleteHalfCenteredDifference(1, approx_order, dx))
        elseif s.grid[x] isa AbstractVector # The nonuniform grid case
            dx = s.grid[x]

            nonlinlap_outer = push!(nonlinlap_outer, Differential(x) => CompleteHalfCenteredDifference(1, approx_order, [(discx[i+1] + discx[i]) / 2 for i in 1:length(discx)-1]))
        else
            error("s.grid contains nonvectors")
        end
        rs = [(Differential(x)^d) => CompleteCenteredDifference(d, approx_order, dx) for d in _orders]
        differentialmap = vcat(differentialmap, rs)

        if advection_scheme isa UpwindScheme
            upwind_orders = orders_[isodd.(orders_)]
        else
            upwind_orders = setdiff(orders_[isodd.(orders_)], [1])
        end
        windpos = vcat(windpos, [(Differential(x)^d) => CompleteUpwindDifference(d, upwind_order, dx, 0) for d in upwind_orders])
        windneg = vcat(windneg, [(Differential(x)^d) => CompleteUpwindDifference(d, upwind_order, dx, d + upwind_order - 1) for d in upwind_orders])
        # only calculate all orders if they are needed for the edge aligned grid
        # TODO: Formalize orders in a type, only do BC_orders[x]
        if get_grid_type(s) <: EdgeAlignedGrid
            half_orders = orders_
        else
            half_orders = (1,)
        end
        nonlinlap_inner = vcat(nonlinlap_inner, [Differential(x)^d => CompleteHalfCenteredDifference(d, approx_order, dx) for d in _orders])
        # A 0th order derivative off the grid is an interpolation
        push!(interp, x => CompleteHalfCenteredDifference(0, max(4, approx_order), dx))
        push!(boundary, x => BoundaryInterpolatorExtrapolator(max(6, approx_order), dx))
    end

    return DifferentialDiscretizer{eltype(orders),typeof(Dict(differentialmap)),typeof(advection_scheme)}(approx_order, advection_scheme, Dict(differentialmap), (Dict(nonlinlap_inner), Dict(nonlinlap_outer)), (Dict(windpos), Dict(windneg)), Dict(interp), Dict(orders), Dict(boundary))
end
