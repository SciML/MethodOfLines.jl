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
Zero-allocation first-order upwind stencil for non-uniform grids using the Fornberg
half-node formula. Builds the symbolic expression directly into the AST without
allocating any weight or tap-point vector.

  v > 0  (left-biased):  (u_i - u_{i-1}) / (x_i - x_{i-1})
  v < 0  (right-biased): (u_{i+1} - u_i) / (x_{i+1} - x_i)

Boundary fallback: at i == 1 the right-biased stencil is used; at i == n the
left-biased stencil is used. In practice the PDE solver never evaluates interior
upwind rules at the boundary nodes (those are handled by BC equations), so these
branches are defensive only.
"""
@inline function _fornberg_upwind(
        II::CartesianIndex, s::DiscreteSpace,
        (j, x), u, ufunc, ispositive
    )
    I1 = unitindex(ndims(u, s), j)
    i = II[j]
    xgrid = s.grid[x]
    n = length(xgrid)

    if ispositive
        if i > 1
            Δx = xgrid[i] - xgrid[i - 1]
            return (ufunc(u, II, x) - ufunc(u, II - I1, x)) / Δx
        else
            Δx = xgrid[i + 1] - xgrid[i]
            return (ufunc(u, II + I1, x) - ufunc(u, II, x)) / Δx
        end
    else
        if i < n
            Δx = xgrid[i + 1] - xgrid[i]
            return (ufunc(u, II + I1, x) - ufunc(u, II, x)) / Δx
        else
            Δx = xgrid[i] - xgrid[i - 1]
            return (ufunc(u, II, x) - ufunc(u, II - I1, x)) / Δx
        end
    end
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

    if d == 1 && !(s.grid[x] isa StepRangeLen)
        return _fornberg_upwind(II, s, jx, u, ufunc, ispositive)
    end

    D = !ispositive ? derivweights.windmap[1][Differential(x)^d] :
        derivweights.windmap[2][Differential(x)^d]
    weights, Itap = _upwind_difference(D, II, s, bs, ispositive, u, jx)
    return sym_dot(weights, ufunc(u, Itap, x))
end

function upwind_difference(
        expr, d::Int, II::CartesianIndex, s::DiscreteSpace, bs,
        depvars, derivweights, (j, x), u, central_ufunc, indexmap
    )
    # TODO: Allow derivatives in expr
    expr = substitute(
        expr, Dict(valmaps(s, u, depvars, Idx(II, s, depvar(u, s), indexmap), indexmap))
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
                                ) => upwind_difference(
                                    *(~a..., ~b...), d, Idx(II, s, u, indexmap), s,
                                    filter_interfaces(bcmap[operation(u)][x]), depvars,
                                    derivweights, (x2i(s, u, x), x), u, wind_ufunc, indexmap
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
                                ) => upwind_difference(
                                    *(~a..., ~b...) / ~c, d, Idx(II, s, u, indexmap), s,
                                    filter_interfaces(bcmap[operation(u)][x]), depvars,
                                    derivweights, (x2i(s, u, x), x), u, wind_ufunc, indexmap
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
                                (Differential(x)^d)(u) => upwind_difference(
                                    d, Idx(II, s, u, indexmap), s,
                                    filter_interfaces(bcmap[operation(u)][x]),
                                    derivweights, (j, x), u, wind_ufunc, true
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
