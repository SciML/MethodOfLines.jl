# use the trapezoid rule
function _euler_integral(interior, s, jx, u, udisc, dx::Number) #where {T,N,Wind,DX<:Number}
    j, x = jx

    interior = get_interior(u, s, interior)
    is = get_is(u, s)

    taps(i) = -1:0 .+ i
    weights(i) = [dx / 2, dx / 2]

    oppairs = map(first(interior):last(interior)) do i
        integral_op_pair(weights, taps, udisc, j, is, interior, i)
    end

    return Construct_ArrayMaker(interior, oppairs)
end

# Nonuniform dx
function _euler_integral(interior, s, jx, u, udisc, dx::AbstractVector) #where {T,N,Wind,DX<:Number}
    j, x = jx

    interior = get_interior(u, s, interior)
    is = get_is(u, s)


    taps(i) = -1:0 .+ i
    weights(i) = [dx[i-1] / 2, dx[i] / 2]

    oppairs = map(first(interior):last(interior)) do i
        integral_op_pair(weights, taps, udisc, j, is, interior, i)
    end

    return Construct_ArrayMaker(interior, oppairs)
end

function euler_integral(interior, s, jx, u, udisc)
    j, x = jx
    dx = s.dxs[x]
    return _euler_integral(interior, s, jx, u, udisc, dx)
end

# An integral across the whole domain (xmin .. xmax)
function whole_domain_integral(II, s, jx, u, udisc)
    j, x = jx
    dx = s.dxs[x]
    return _wd_integral(interior, s, jx, u, udisc, dx)
end

function _wd_integral(interior, s, jx, u, udisc, dx::Number) #where {T,N,Wind,DX<:Number}
    j, x = jx

    interior = get_interior(u, s, interior)
    is = get_is(u, s)
    lenx = length(s, x)

    taps(i) = -1:0 .+ i
    weights(i) = [dx / 2, dx / 2]

    return IntegralArrayOp(weights, taps, lenx, udisc, j, is, interior)
end

function _wd_integral(interior, s, jx, u, udisc, dx::AbstractVector) #where {T,N,Wind,DX<:Number}
    j, x = jx

    interior = get_interior(u, s, interior)
    is = get_is(u, s)
    lenx = length(s, x)

    taps(i) = -1:0 .+ i
    weights(i) = [dx[i-1]/2, dx[i]/2]

    return IntegralArrayOp(weights, taps, lenx, udisc, j, is, interior, true)
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
