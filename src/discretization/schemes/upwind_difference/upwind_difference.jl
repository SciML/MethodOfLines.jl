@inline function _upwind_difference(
        D::DerivativeOperator{T, N, Wind, DX}, II, s, bs,
        ispositive, u, jx
    ) where {T, N, Wind, DX <: Number}
    j, x = jx
    I1 = unitindex(ndims(u, s), j)
    haslower, hasupper = haslowerupper(bs, x)
    if !ispositive
        if (II[j] > (length(s, x) - D.boundary_point_count)) & !hasupper
            weights = D.high_boundary_coefs[length(s, x) - II[j] + 1]
            offset = length(s, x) - II[j]
            Itap = [II + (i + offset) * I1 for i in (-D.boundary_stencil_length + 1):0]
        else
            weights = D.stencil_coefs
            Itap = [bwrap(II + i * I1, bs, s, jx) for i in 0:(D.stencil_length - 1)]
        end
    else
        if (II[j] <= D.offside) & !haslower
            weights = D.low_boundary_coefs[II[j]]
            offset = 1 - II[j]
            Itap = [II + (i + offset) * I1 for i in 0:(D.boundary_stencil_length - 1)]
        else
            weights = D.stencil_coefs
            Itap = [bwrap(II + i * I1, bs, s, jx) for i in (-D.stencil_length + 1):0]
        end
    end
    return weights, Itap
end

@inline function _upwind_difference(
        D::DerivativeOperator{T, N, Wind, DX}, II, s, bs,
        ispositive, u, jx
    ) where {T, N, Wind, DX <: AbstractVector}
    j, x = jx
    @assert length(bs) == 0 "Interface boundary conditions are not yet supported for nonuniform dx dimensions, such as $x, please post an issue to https://github.com/SciML/MethodOfLines.jl if you need this functionality."
    I1 = unitindex(ndims(u, s), j)
    if !ispositive
        @assert D.offside == 0

        if (II[j] > (length(s, x) - D.boundary_point_count))
            weights = D.high_boundary_coefs[length(s, x) - II[j] + 1]
            offset = length(s, x) - II[j]
            Itap = [II + (i + offset) * I1 for i in (-D.boundary_stencil_length + 1):0]
        else
            weights = D.stencil_coefs[II[j]]
            Itap = [II + i * I1 for i in 0:(D.stencil_length - 1)]
        end
    else
        if (II[j] <= D.offside)
            weights = D.low_boundary_coefs[II[j]]
            offset = 1 - II[j]
            Itap = [II + (i + offset) * I1 for i in 0:(D.boundary_stencil_length - 1)]
        else
            weights = D.stencil_coefs[II[j] - D.offside]
            Itap = [II + i * I1 for i in (-D.stencil_length + 1):0]
        end
    end
    return weights, Itap
end

"""
# upwind_difference
Generate a finite difference expression in `u` using the upwind difference at point `II::CartesianIndex`
in the direction of `x`
"""
function upwind_difference(
        d::Int, II::CartesianIndex, s::DiscreteSpace,
        bs, derivweights, jx, u, ufunc, ispositive
    )
    j, x = jx
    # return if this is an ODE
    ndims(u, s) == 0 && return 0
    D = !ispositive ? derivweights.windmap[1][Differential(x)^d] :
        derivweights.windmap[2][Differential(x)^d]
    #@show D.stencil_coefs, D.stencil_length, D.boundary_stencil_length, D.boundary_point_count
    # unit index in direction of the derivative
    weights, Itap = _upwind_difference(D, II, s, bs, ispositive, u, jx)
    return sym_dot(weights, ufunc(u, Itap, x))  # unit correction applied at rule generation level
end

function upwind_difference(
        expr, d::Int, II::CartesianIndex, s::DiscreteSpace, bs,
        depvars, derivweights, (j, x), u, central_ufunc, indexmap
    )
    # TODO: Allow derivatives in expr
    expr = substitute(
        expr, valmaps(s, u, depvars, Idx(II, s, depvar(u, s), indexmap), indexmap)
    )
    return IfElse.ifelse(
        expr > 0,
        expr *
            upwind_difference(d, II, s, bs, derivweights, (j, x), u, central_ufunc, true),
        expr *
            upwind_difference(d, II, s, bs, derivweights, (j, x), u, central_ufunc, false)
    )
end

@inline function generate_winding_rules(
        II::CartesianIndex, s::DiscreteSpace, depvars,
        derivweights::DifferentialDiscretizer, bcmap, indexmap, terms; skip = []
    )
    wind_ufunc(v, I, x) = s.discvars[v][I]
    # for all independent variables and dependant variables
    rules = safe_vcat(#Catch multiplication
        reduce(
            safe_vcat,
            [
                reduce(
                        safe_vcat,
                        [
                            [
                                @rule *(
                                    ~~a,
                                    $(Differential(x)^d)(u),
                                    ~~b
                                ) => unit_correct(
                                    upwind_difference(
                                        *(~a..., ~b...), d, Idx(II, s, u, indexmap), s,
                                        filter_interfaces(bcmap[operation(u)][x]), depvars,
                                        derivweights, (x2i(s, u, x), x), u, wind_ufunc, indexmap
                                    ),
                                    x, d, derivweights.unit_map
                                )
                                for d in (
                                    let orders = derivweights.orders[x]
                                        setdiff(orders[isodd.(orders)], skip)
                                end
                                )
                            ] for x in ivs(u, s)
                        ],
                        init = []
                    ) for u in depvars
            ],
            init = []
        ),

        #Catch division and multiplication, see issue #1
        reduce(
            safe_vcat,
            [
                reduce(
                        safe_vcat,
                        [
                            [
                                @rule /(
                                    *(~~a, $(Differential(x)^d)(u), ~~b),
                                    ~c
                                ) => unit_correct(
                                    upwind_difference(
                                        *(~a..., ~b...) / ~c, d, Idx(II, s, u, indexmap), s,
                                        filter_interfaces(bcmap[operation(u)][x]), depvars,
                                        derivweights, (x2i(s, u, x), x), u, wind_ufunc, indexmap
                                    ),
                                    x, d, derivweights.unit_map
                                )
                                for d in (
                                    let orders = derivweights.orders[x]
                                        setdiff(orders[isodd.(orders)], skip)
                                end
                                )
                            ] for x in ivs(u, s)
                        ],
                        init = []
                    ) for u in depvars
            ],
            init = []
        )
    )

    wind_rules = []

    # wind_exprs = []
    for t in terms
        for r in rules
            if r(t) !== nothing
                push!(wind_rules, t => r(t))
            end
        end
    end

    return safe_vcat(
        wind_rules,
        vec(
            mapreduce(safe_vcat, depvars, init = []) do u
                mapreduce(safe_vcat, ivs(u, s), init = []) do x
                    j = x2i(s, u, x)
                    let orders = setdiff(derivweights.orders[x], skip)
                        oddorders = orders[isodd.(orders)]
                        # for all odd orders
                        if length(oddorders) > 0
                            map(oddorders) do d
                                (Differential(x)^d)(u) => unit_correct(
                                    upwind_difference(
                                        d, Idx(II, s, u, indexmap), s,
                                        filter_interfaces(bcmap[operation(u)][x]),
                                        derivweights, (j, x), u, wind_ufunc, true
                                    ),
                                    x, d, derivweights.unit_map
                                )
                            end
                        else
                            []
                        end
                    end
                end
            end
        )
    )
end
