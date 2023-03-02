function euler_integral(interior, s, jx, u, udisc)
    j, x = jx
    dx = s.dxs[x]
    interior = get_interior(u, s, interior)
    ranges = get_ranges(u, s)
    is = get_is(u, s)

    oppairs = map(interior[j]) do i
        integral_op_pair(dx, udisc, j, is, ranges, interior, i)
    end

    return NullBG_ArrayMaker(ranges, oppairs)[interior...]
end

# An integral across the whole domain (xmin .. xmax)
function whole_domain_integral(interior, s, jx, u, udisc)
    j, x = jx
    dx = s.dxs[x]

    interior = get_interior(u, s, interior)
    ranges = get_ranges(u, s)
    is = get_is(u, s)

    lenx = length(s, x)

    return IntegralArrayMaker(dx, udisc, lenx, j, is, ranges, interior, true)[interior...]
end

@inline function generate_euler_integration_rules(interior, s::DiscreteSpace, depvars, indexmap, terms)
    eulerrules = reduce(safe_vcat, [[Integral(x in DomainSets.ClosedInterval(s.vars.intervals[x][1], Num(x)))(u) =>
                                         euler_integral(interior, s, (x2i(s, u, x), x), u, s.discvars[u])
                                     for x in ivs(u, s)]
                                    for u in depvars], init = [])
    return eulerrules
end

@inline function generate_whole_domain_integration_rules(interior, s::DiscreteSpace, depvars, indexmap, terms, bvar = nothing)
    wholedomainrules = reduce(safe_vcat,
                              [[Integral(x in DomainSets.ClosedInterval(s.vars.intervals[x][1], s.vars.intervals[x][2]))(u) =>
                                    whole_domain_integral(interior, s, (x2i(s, u, x), x), u, s.discvars[u])
                                for x in filter(x -> (!haskey(indexmap, x) | isequal(x, bvar)), ivs(u, s))]
                               for u in depvars],
                              init = [])

    return wholedomainrules
end
