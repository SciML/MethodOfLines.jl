# Use DiffEqOperators to generate weights and calculate derivative orders
struct DifferentialDiscretizer{T, D1, S}
    approx_order::Int
    advection_scheme::S
    map::D1
    halfoffsetmap::Tuple{Dict, Dict}
    windmap::Any
    interpmap::Dict{Num, DerivativeOperator}
    orders::Dict{Num, Vector{Int}}
    boundary::Dict{Num, DerivativeOperator}
end

function DifferentialDiscretizer(pdesys, s, discretization, orders)
    pdeeqs = pdesys.eqs isa Vector ? pdesys.eqs : [pdesys.eqs]
    bcs = pdesys.bcs isa Vector ? pdesys.bcs : [pdesys.bcs]
    approx_order = discretization.approx_order
    advection_scheme = discretization.advection_scheme
    upwind_order = advection_scheme isa UpwindScheme ? advection_scheme.order : 0

    differentialmap = Array{Pair{Num, DerivativeOperator}, 1}()
    nonlinlap_inner = []
    nonlinlap_outer = []
    windpos = []
    windneg = []
    interp = []
    boundary = []
    for x in s.x̄
        orders_ = orders[x]
        _orders = Set(vcat(orders_, [1, 2]))

        if s.grid[x] isa StepRangeLen # Uniform grid case
            # TODO: Only generate weights for derivatives that are actually used and avoid redundant calculations
            rs = [(Differential(x)^d) => CompleteCenteredDifference(d, approx_order,
                                                                    s.dxs[x])
                  for d in _orders]

            if advection_scheme isa UpwindScheme
                windpos = vcat(windpos,
                               [(Differential(x)^d) => CompleteUpwindDifference(d,
                                                                                upwind_order,
                                                                                s.dxs[x], 0)
                                for d in orders_[isodd.(orders_)]])
                windneg = vcat(windneg,
                               [(Differential(x)^d) => CompleteUpwindDifference(d,
                                                                                upwind_order,
                                                                                s.dxs[x],
                                                                                d +
                                                                                upwind_order -
                                                                                1)
                                for d in orders_[isodd.(orders_)]])
            end

            nonlinlap_inner = vcat(nonlinlap_inner,
                                   [Differential(x)^d => CompleteHalfCenteredDifference(d,
                                                                                        approx_order,
                                                                                        s.dxs[x])
                                    for d in _orders])
            nonlinlap_outer = push!(nonlinlap_outer,
                                    Differential(x) => CompleteHalfCenteredDifference(1,
                                                                                      approx_order,
                                                                                      s.dxs[x]))
            differentialmap = vcat(differentialmap, rs)
            # A 0th order derivative off the grid is an interpolation
            push!(interp,
                  x => CompleteHalfCenteredDifference(0, max(4, approx_order), s.dxs[x]))
            push!(boundary,
                  x => BoundaryInterpolatorExtrapolator(max(6, approx_order), s.dxs[x]))

        elseif s.grid[x] isa AbstractVector # The nonuniform grid case
            rs = [(Differential(x)^d) => CompleteCenteredDifference(d, approx_order,
                                                                    s.grid[x])
                  for d in _orders]
            if advection_scheme isa UpwindScheme
                windpos = vcat(windpos,
                               [(Differential(x)^d) => CompleteUpwindDifference(d,
                                                                                upwind_order,
                                                                                s.grid[x],
                                                                                0)
                                for d in orders_[isodd.(orders_)]])
                windneg = vcat(windneg,
                               [(Differential(x)^d) => CompleteUpwindDifference(d,
                                                                                upwind_order,
                                                                                s.grid[x],
                                                                                d +
                                                                                upwind_order -
                                                                                1)
                                for d in orders_[isodd.(orders_)]])
            end

            discx = s.grid[x]
            nonlinlap_inner = vcat(nonlinlap_inner,
                                   [Differential(x)^d => CompleteHalfCenteredDifference(d,
                                                                                        approx_order,
                                                                                        s.grid[x])
                                    for d in _orders])
            nonlinlap_outer = push!(nonlinlap_outer,
                                    Differential(x) => CompleteHalfCenteredDifference(1,
                                                                                      approx_order,
                                                                                      [(discx[i + 1] +
                                                                                        discx[i]) /
                                                                                       2
                                                                                       for i in 1:(length(discx) - 1)]))
            differentialmap = vcat(differentialmap, rs)
            # A 0th order derivative off the grid is an interpolation
            push!(interp,
                  x => CompleteHalfCenteredDifference(0, max(4, approx_order), s.grid[x]))
            push!(boundary,
                  x => BoundaryInterpolatorExtrapolator(max(6, approx_order), s.grid[x]))
        else
            @assert false "s.grid contains nonvectors"
        end
    end

    return DifferentialDiscretizer{eltype(orders), typeof(Dict(differentialmap)),
                                   typeof(advection_scheme)}(approx_order, advection_scheme,
                                                             Dict(differentialmap),
                                                             (Dict(nonlinlap_inner),
                                                              Dict(nonlinlap_outer)),
                                                             (Dict(windpos), Dict(windneg)),
                                                             Dict(interp), Dict(orders),
                                                             Dict(boundary))
end
