function euler_integral(II, s, b, jx, u, ufunc) #where {T,N,Wind,DX<:Number}
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

@inline function generate_integration_rules(II::CartesianIndex, s::DiscreteSpace, depvars, pmap, indexmap, terms)
    ufunc(u, I, x) = s.discvars[u][I]
    #x_min = s.grid[x][1]
    @info "Generating"

    rules = Vector{Any}([])
    for u in depvars
        urules = Vector{Any}([])
        for x in params(u, s)
            @info "INTEGRATION RULES" II[x2i(s, u, x)]
            result_expr = euler_integral(Idx(II, s, u, indexmap), s, pmap.map[operation(u)][x], (x2i(s, u, x), x), u, ufunc)
            int_expr = Integral(x in DomainSets.ClosedInterval(0.0, x))(u)
            @info "RESULTS" result_expr typeof(result_expr) typeof(Integral(x in DomainSets.ClosedInterval(0.0, x))(u))
            rule = Integral(x in DomainSets.ClosedInterval(0.0, x))(u) => result_expr
            vcat(urules, [Integral(x in DomainSets.ClosedInterval(0.0, x))(u) => euler_integral(Idx(II, s, u, indexmap), s, pmap.map[operation(u)][x], (x2i(s, u, x), x), u, ufunc)])
        end
        rules = vcat(rules, urules)
    end

    rules = reduce(vcat, [reduce(vcat, [[Integral(x in DomainSets.ClosedInterval(0.0, x))(u) => euler_integral(Idx(II, s, u, indexmap), s, pmap.map[operation(u)][x], (x2i(s, u, x), x), u, ufunc)
                                         for x in params(u, s)]])
                          for u in depvars])
    rule_pairs = []
    for t in terms
        for r in rules
            if r(t) !== nothing
                push!(rule_pairs, t => r(t))
            end
        end
    end
    return rule_pairs
    # return reduce(vcat, [reduce(vcat, [[(Differential(x)^d)(u) => central_difference(derivweights.map[Differential(x)^d], Idx(II, s, u, indexmap), s, pmap.map[operation(u)][x], (x2i(s, u, x), x), u, central_ufunc) for d in (
    #     let orders = derivweights.orders[x]
    #         orders[iseven.(orders)]
    #     end
    # )] for x in params(u, s)]) for u in depvars])
end
