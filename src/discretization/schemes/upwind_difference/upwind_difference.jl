@inline function _upwind_difference(D::DerivativeOperator{T,N,Wind,DX}, II, s, b, ispositive, u, jx) where {T,N,Wind,DX<:Number}
    j, x = jx
    I1 = unitindex(ndims(u, s), j)
    if !ispositive
        if (II[j] > (length(s, x) - D.boundary_point_count)) & (b isa Val{false})
            weights = D.high_boundary_coefs[length(s, x)-II[j]+1]
            offset = length(s, x) - II[j]
            Itap = [II + (i + offset) * I1 for i in (-D.boundary_stencil_length+1):0]
        else
            weights = D.stencil_coefs
            Itap = [wrapperiodic(II + i * I1, s, b, u, jx) for i in 0:D.stencil_length-1]
        end
    else
        if (II[j] <= D.offside) & (b isa Val{false})
            weights = D.low_boundary_coefs[II[j]]
            offset = 1 - II[j]
            Itap = [II + (i + offset) * I1 for i in 0:(D.boundary_stencil_length-1)]
        else
            weights = D.stencil_coefs
            Itap = [wrapperiodic(II + i * I1, s, b, u, jx) for i in -D.stencil_length+1:0]
        end
    end
    return weights, Itap
end

@inline function _upwind_difference(D::DerivativeOperator{T,N,Wind,DX}, II, s, b, ispositive, u, jx) where {T,N,Wind,DX<:AbstractVector}
    j, x = jx
    @assert b isa Val{false} "Periodic boundary conditions are not yet supported for nonuniform dx dimensions, such as $x, please post an issue to https://github.com/SciML/MethodOfLines.jl if you need this functionality."
    I1 = unitindex(ndims(u, s), j)
    if !ispositive
        @assert D.offside == 0

        if (II[j] > (length(s, x) - D.boundary_point_count))
            weights = D.high_boundary_coefs[length(s, x)-II[j]+1]
            offset = length(s, x) - II[j]
            Itap = [II + (i + offset) * I1 for i in (-D.boundary_stencil_length+1):0]
        else
            weights = D.stencil_coefs[II[j]]
            Itap = [II + i * I1 for i in 0:D.stencil_length-1]
        end
    else
        if (II[j] <= D.offside)
            weights = D.low_boundary_coefs[II[j]]
            offset = 1 - II[j]
            Itap = [II + (i + offset) * I1 for i in 0:(D.boundary_stencil_length-1)]
        else
            weights = D.stencil_coefs[II[j]-D.offside]
            Itap = [II + i * I1 for i in -D.stencil_length+1:0]
        end
    end
    return weights, Itap
end

"""
# upwind_difference
Generate a finite difference expression in `u` using the upwind difference at point `II::CartesianIndex`
in the direction of `x`
"""
function upwind_difference(d::Int, II::CartesianIndex, s::DiscreteSpace, b, derivweights, jx, u, ufunc, ispositive)
    j, x = jx
    ndims(u, s) == 0 && return Num(0)
    D = !ispositive ? derivweights.windmap[1][Differential(x)^d] : derivweights.windmap[2][Differential(x)^d]
    #@show D.stencil_coefs, D.stencil_length, D.boundary_stencil_length, D.boundary_point_count
    # unit index in direction of the derivative
    weights, Itap = _upwind_difference(D, II, s, b, ispositive, u, jx)
    return dot(weights, ufunc(u, Itap, x))
end

@inline function upwind_difference(expr, d::Int, II::CartesianIndex, s::DiscreteSpace, b, depvars, derivweights, (j, x), u, central_ufunc, indexmap)
    # TODO: Allow derivatives in expr
    expr = substitute(expr, valmaps(s, u, depvars, Idx(II, s, depvar(u, s), indexmap), indexmap))
    IfElse.ifelse(expr > 0,
        expr * upwind_difference(d, II, s, b, derivweights, (j, x), u, central_ufunc, true),
        expr * upwind_difference(d, II, s, b, derivweights, (j, x), u, central_ufunc, false))
end

@inline function generate_winding_rules(II::CartesianIndex, s::DiscreteSpace, depvars, derivweights::DifferentialDiscretizer, pmap, indexmap, terms)
    wind_ufunc(v, I, x) = s.discvars[v][I]
    # for all independent variables and dependant variables
    rules = vcat(#Catch multiplication
        reduce(vcat, [reduce(vcat, [[@rule *(~~a, $(Differential(x)^d)(u), ~~b) => upwind_difference(*(~a..., ~b...), d, Idx(II, s, u, indexmap), s, pmap.map[operation(u)][x], depvars, derivweights, (x2i(s, u, x), x), u, wind_ufunc, indexmap) for d in (
            let orders = derivweights.orders[x]
                orders[isodd.(orders)]
            end
        )] for x in params(u, s)]) for u in depvars]),

        #Catch division and multiplication, see issue #1
        reduce(vcat, [reduce(vcat, [[@rule /(*(~~a, $(Differential(x)^d)(u), ~~b), ~c) => upwind_difference(*(~a..., ~b...) / ~c, d, Idx(II, s, u, indexmap), s, pmap.map[operation(u)][x], depvars, derivweights, (x2i(s, u, x), x), u, wind_ufunc, indexmap) for d in (
            let orders = derivweights.orders[x]
                orders[isodd.(orders)]
            end
        )] for x in params(u, s)]) for u in depvars])
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

    return vcat(wind_rules, vec(mapreduce(vcat, depvars) do u
        mapreduce(vcat, params(u, s)) do x
            j = x2i(s, u, x)
            let orders = derivweights.orders[x]
                oddorders = orders[isodd.(orders)]
                # for all odd orders
                if length(oddorders) > 0
                    map(oddorders) do d
                        (Differential(x)^d)(u) => upwind_difference(d, Idx(II, s, u, indexmap), s, pmap.map[operation(u)][x], derivweights, (j, x), u, wind_ufunc, true)
                    end
                else
                    []
                end
            end
        end
    end))
end
