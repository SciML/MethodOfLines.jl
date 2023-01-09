# use the trapezoid rule
function _euler_integral(II, s, jx, u, ufunc, dx::Number) #where {T,N,Wind,DX<:Number}
    j, x = jx
    if II[j] == 1
        return Num(0)
    end
    # unit index in direction of the derivative
    I1 = unitindex(ndims(u, s), j)
    # dx for multiplication
    Itap = [II - I1, II]
    weights = [dx / 2, dx / 2]

    return dot(weights, ufunc(u, Itap, x)) + _euler_integral(II - I1, s, jx, u, ufunc, dx)
end

# Nonuniform dx
function _euler_integral(II, s, jx, u, ufunc, dx::AbstractVector) #where {T,N,Wind,DX<:Number}
    j, x = jx
    if II[j] == 1
        return Num(0)
    end
    # unit index in direction of the derivative
    I1 = unitindex(ndims(u, s), j)
    # dx for multiplication
    Itap = [II - I1, II]
    weights = fill(dx[II[j]-1] / 2, 2)

    return dot(weights, ufunc(u, Itap, x)) + _euler_integral(II - I1, s, jx, u, ufunc, dx)
end

function euler_integral(II, s, jx, u, ufunc)
    j, x = jx
    dx = s.dxs[x]
    return _euler_integral(II, s, jx, u, ufunc, dx)
end

# An integral across the whole domain (xmin .. xmax)
function whole_domain_integral(II, s, jx, u, ufunc)
    j, x = jx
    dx = s.dxs[x]
    if II[j] == length(s, x)
        return _euler_integral(II, s, jx, u, ufunc, dx)
    end

    dist2max = length(s, x) - II[j]
    I1 = unitindex(ndims(u, s), j)
    Imax = II + dist2max * I1
    return _euler_integral(Imax, s, jx, u, ufunc, dx)
end

@inline function generate_euler_integration_rules(II::CartesianIndex, s::DiscreteSpace, depvars, indexmap, terms)
    ufunc(u, I, x) = s.discvars[u][I]

    eulerrules = reduce(safe_vcat, [[Integral(x in DomainSets.ClosedInterval(s.vars.intervals[x][1], Num(x)))(u) => euler_integral(Idx(II, s, u, indexmap), s, (x2i(s, u, x), x), u, ufunc)
                                for x in params(u, s)]
                               for u in depvars], init = [])
    return eulerrules
end

function wd_integral_Idx(II::CartesianIndex, s::DiscreteSpace, u, x, indexmap)
    # We need to construct a new index as indices may be of different size
    length(params(u, s)) == 0 && return CartesianIndex()
    # A hack using the boundary value re-indexing function to get an index that will work
    u_ = substitute(u, [x => s.axies[x][end]])
    II = newindex(u_, II, s, indexmap)
    return II
end

@inline function generate_whole_domain_integration_rules(II::CartesianIndex, s::DiscreteSpace, depvars, indexmap, terms, bvar = nothing)
    ufunc(u, I, x) = s.discvars[u][I]
    wholedomainrules = reduce(safe_vcat,
                              [[Integral(x in DomainSets.ClosedInterval(s.vars.intervals[x][1], s.vars.intervals[x][2]))(u) =>
                                    whole_domain_integral(wd_integral_Idx(II, s, u, x, indexmap), s, (x2i(s, u, x), x), u, ufunc)
                                for x in filter(x -> (!haskey(indexmap, x) | isequal(x, bvar)), params(u, s))]
                               for u in depvars],
                              init = [])
    return wholedomainrules
end
