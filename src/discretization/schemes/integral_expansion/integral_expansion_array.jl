function euler_integral(interior, s, jx, u, udisc)
    j, x = jx
    dx = s.dxs[x]

    interior = get_interior(u, s, interior)
    is = get_is(u, s)

    oppairs = map(interior[j]) do i
        integral_op_pair(dx, udisc, j, is, interior, i)
    end

    return Construct_ArrayMaker(interior, oppairs)
end

# An integral across the whole domain (xmin .. xmax)
function whole_domain_integral(II, s, jx, u, udisc)
    j, x = jx
    dx = s.dxs[x]

    interior = get_interior(u, s, interior)
    is = get_is(u, s)

    return IntegralArrayOp(dx, udisc, j, is, interior, true)
end

@inline function generate_euler_integration_rules(interior, s::DiscreteSpace, depvars, indexmap, terms)
    eulerrules = reduce(safe_vcat, [[Integral(x in DomainSets.ClosedInterval(s.vars.intervals[x][1], Num(x)))(u) =>
                                         euler_integral(interior, s, (x2i(s, u, x), x), u, s.discvars[u])
                                     for x in params(u, s)]
                                    for u in depvars], init = [])
    return eulerrules
end

@inline function generate_whole_domain_integration_rules(interior, s::DiscreteSpace, depvars, indexmap, terms, bvar = nothing)
    wholedomainrules = reduce(safe_vcat,
                              [[Integral(x in DomainSets.ClosedInterval(s.vars.intervals[x][1], s.vars.intervals[x][2]))(u) =>
                                    whole_domain_integral(interior, s, (x2i(s, u, x), x), u, s.discvars[u])
                                for x in filter(x -> (!haskey(indexmap, x) | isequal(x, bvar)), params(u, s))]
                               for u in depvars],
                              init = [])

    return wholedomainrules
end
