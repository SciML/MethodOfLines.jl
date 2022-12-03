function euler_integral(II, s, jx, u, ufunc) #where {T,N,Wind,DX<:Number}
    j, x = jx
    ndims(u, s) != 1 && return Num(0)
    # unit index in direction of the derivative
    I1 = unitindex(ndims(u, s), j)
    # dx for multiplication
    dx = s.dxs[x]

    @info "EULER INTEGRAL DEBUG"
    @info II[j]
    @info dx
    Itap = [(II - I1 * i) for i = 0:(II[j]-1)] # Sum from current position till CartesianIndex 2

    weights = [dx * (i != (II[j] - 1)) for i = 0:(II[j]-1)]
    @info weights
    @info Itap
    @info ufunc(u, Itap, x)

    return dot(weights, ufunc(u, Itap, x))
end

@inline function generate_integration_rules(II::CartesianIndex, s::DiscreteSpace, depvars, indexmap, terms)
    ufunc(u, I, x) = s.discvars[u][I]
    #x_min = s.grid[x][1]

    # rules = Vector{Any}([])
    # for u in depvars
    #     urules = Vector{Any}([])
    #     for x in params(u, s)
    #         xx = Num(x)
    #         domain = DomainSets.ClosedInterval(0.0, xx)
    #         vcat(urules, [Integral(x in domain)(u) => euler_integral(Idx(II, s, u, indexmap), s, (x2i(s, u, x), x), u, ufunc)])
    #     end
    #     rules = vcat(rules, urules)
    # end

    rules = reduce(vcat, [reduce(vcat, [Integral(Num(x) in DomainSets.ClosedInterval(0.0, Num(x)))(u) => euler_integral(Idx(II, s, u, indexmap), s, (x2i(s, u, x), x), u, ufunc)
                                        for x in params(u, s)])
                          for u in depvars])
    return rules
    # return reduce(vcat, [reduce(vcat, [[(Differential(x)^d)(u) => central_difference(derivweights.map[Differential(x)^d], Idx(II, s, u, indexmap), s, pmap.map[operation(u)][x], (x2i(s, u, x), x), u, central_ufunc) for d in (
    #     let orders = derivweights.orders[x]
    #         orders[iseven.(orders)]
    #     end
    # )] for x in params(u, s)]) for u in depvars])
end
