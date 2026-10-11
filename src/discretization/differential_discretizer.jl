# Use DiffEqOperators to generate weights and calculate derivative orders
struct HermiteWeightData
    d_weights::Union{Nothing, Vector{Float64}}
    e_weights::Union{Nothing, Vector{Float64}}
    stencil_length::Int
end

struct DifferentialDiscretizer{T, D1, S, H} <: AbstractDifferentialDiscretizer
    approx_order::Int
    advection_scheme::S
    map::D1
    halfoffsetmap::Tuple{Dict, Dict}
    windmap::Any
    interpmap::Dict{Num, DerivativeOperator}
    orders::Dict{Num, Vector{Int}}
    boundary::Dict{Num, DerivativeOperator}
    callbacks::Any
    icformulas::Any
    hermitemap::H  # HFD weights by derivative operator
end

function PDEBase.construct_differential_discretizer(
        pdesys, s::DiscreteSpace, discretization::MOLFiniteDifference, orders
    )
    pdeeqs = get_eqs(pdesys)
    pdeeqs = pdeeqs isa Vector ? pdeeqs : [pdeeqs]
    bcs = get_bcs(pdesys)
    bcs = bcs isa Vector ? bcs : [bcs]
    approx_order = discretization.approx_order
    advection_scheme = discretization.advection_scheme
    callbacks = discretization.callbacks
    upwind_order = advection_scheme isa UpwindScheme ? advection_scheme.order : 1

    differentialmap = Array{Pair{Num, DerivativeOperator}, 1}()
    nonlinlap_inner = []
    nonlinlap_outer = []
    windpos = []
    windneg = []
    interp = []
    boundary = []
    # TODO: Make sure that only necessary orders are calculated, this is the lowest hanging performance fruit.
    for x in s.x̄
        orders_ = orders[x]
        _orders = Set(vcat(orders_, [1, 2]))

        if s.grid[x] isa StepRangeLen # Uniform grid case
            dx = s.dxs[x]

            nonlinlap_outer = push!(
                nonlinlap_outer,
                Differential(x) => CompleteHalfCenteredDifference(1, approx_order, dx)
            )
        elseif s.grid[x] isa AbstractVector # The nonuniform grid case
            dx = s.grid[x]

            nonlinlap_outer = push!(
                nonlinlap_outer,
                Differential(x) => CompleteHalfCenteredDifference(
                    1, approx_order, [(dx[i + 1] + dx[i]) / 2 for i in 1:(length(dx) - 1)]
                )
            )
        else
            error("s.grid contains nonvectors")
        end
        rs = [
            (Differential(x)^d) => CompleteCenteredDifference(d, approx_order, dx)
                for d in _orders
        ]
        differentialmap = vcat(differentialmap, rs)

        if advection_scheme isa UpwindScheme
            upwind_orders = orders_[isodd.(orders_)]
        else
            upwind_orders = setdiff(orders_[isodd.(orders_)], [1])
        end
        windpos = vcat(
            windpos,
            [
                (Differential(x)^d) => CompleteUpwindDifference(d, upwind_order, dx, 0)
                    for d in upwind_orders
            ]
        )
        windneg = vcat(
            windneg,
            [
                (Differential(x)^d) => CompleteUpwindDifference(
                    d, upwind_order, dx, d + upwind_order - 1
                ) for d in upwind_orders
            ]
        )
        # only calculate all orders if they are needed for the edge aligned grid
        # TODO: Formalize orders in a type, only do BC_orders[x]
        if get_grid_type(s) <: EdgeAlignedGrid
            half_orders = orders_
        else
            half_orders = (1,)
        end
        nonlinlap_inner = vcat(
            nonlinlap_inner,
            [
                Differential(x)^d => CompleteHalfCenteredDifference(d, approx_order, dx)
                    for d in _orders
            ]
        )
        # A 0th order derivative off the grid is an interpolation
        push!(interp, x => CompleteHalfCenteredDifference(0, max(4, approx_order), dx))
        push!(boundary, x => BoundaryInterpolatorExtrapolator(max(6, approx_order), dx))
    end

    # Compute Hermite FD weights if HermiteScheme is active
    hermitemap = Dict{Differential, HermiteWeightData}()
    if advection_scheme isa HermiteScheme
        for x in s.x̄
            orders_ = orders[x]
            _orders = Set(vcat(orders_, [1, 2]))
            dx = s.dxs[x]
            for d in _orders
                if s.grid[x] isa StepRangeLen
                    # Uniform grid: single set of weights
                    stencil_length = d + approx_order - 1 + (d + approx_order) % 2
                    half = div(stencil_length, 2)
                    stencil_nodes = [Float64(k * dx) for k in -half:half]
                    d_w, e_w = calculate_weights(d, 0.0, stencil_nodes; dfdx = true)
                    hermitemap[Differential(x)^d] =
                        HermiteWeightData(d_w, e_w, stencil_length)
                else
                    # Nonuniform stencils depend on the evaluation point and are
                    # generated from the selected taps below. Do not precompute
                    # per-point matrices here; that would duplicate work and
                    # allocate O(N) weight vectors for every derivative order.
                    hermitemap[Differential(x)^d] =
                        HermiteWeightData(nothing, nothing, 0)
                end
            end
        end
    end

    return DifferentialDiscretizer{
        eltype(orders), typeof(Dict(differentialmap)), typeof(advection_scheme),
        typeof(hermitemap),
    }(
        approx_order, advection_scheme, Dict(differentialmap),
        (Dict(nonlinlap_inner), Dict(nonlinlap_outer)), (Dict(windpos), Dict(windneg)),
        Dict(interp), Dict(orders), Dict(boundary), callbacks,
        array_ic_formulas(pdesys, s), hermitemap
    )
end

"""
Canonical dependent variable => order-0 IC formula after `symbolic_linear_solve`.
Time-literals evaluate this field; the discrete unknown is `U(t)`, not `U(t0)`.
"""
function array_ic_formulas(pdesys, s)
    s.time === nothing && return Dict()
    bcs = get_bcs(pdesys)
    bcs = bcs isa Vector ? bcs : [bcs]
    v = s.vars
    bcorders = Dict(x => d_orders(x, bcs) for x in all_ivs(v))
    boundarymap = PDEBase.parse_bcs(bcs, v, bcorders)
    formulas = Dict()
    for uop in operation.(v.ū)
        for ic in get(get(boundarymap, uop, Dict()), s.time, [])
            isupper(ic) && continue
            ic.order == 0 || continue
            formulas[safe_unwrap(depvar(ic.u, s))] = symbolic_linear_solve(ic.eq, ic.u)
        end
    end
    return formulas
end
