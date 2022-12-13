#TODO: Add support for non-uniform dx
#TODO: Use the rectangle rule, trapezium rule and simpsons rule to approximate the integral.
function _euler_integral(II, s, jx, u, ufunc, dx::Number) #where {T,N,Wind,DX<:Number}
    j, x = jx
    # unit index in direction of the derivative
    I1 = unitindex(ndims(u, s), j)
    # dx for multiplication

    Itap = [(II - I1 * i) for i = 0:(II[j]-1)] # Sum from current position till CartesianIndex 2
    weights = [dx * (i != (II[j] - 1)) for i = 0:(II[j]-1)]

    return dot(weights, ufunc(u, Itap, x))
end

# Nonuniform dx
function _euler_integral(II, s, jx, u, ufunc, dx::AbstractVector) #where {T,N,Wind,DX<:Number}
    j, x = jx
    # unit index in direction of the derivative
    I1 = unitindex(ndims(u, s), j)
    # dx for multiplication

    Itap = [(II - I1 * i) for i = (II[j]-1):-1:0] # Sum from current position till CartesianIndex 2
    weights = [dx[i+1] * (i != 1) for i = 0:(II[j]-1)]

    return dot(weights, ufunc(u, Itap, x))
end

function euler_integral(II, s, jx, u, ufunc)
    dx = s.dxs[jx[2]]
    return _euler_integral(II, s, jx, u, ufunc, dx)
end

@inline function generate_integration_rules(II::CartesianIndex, s::DiscreteSpace, depvars, indexmap, terms)
    ufunc(u, I, x) = s.discvars[u][I]

    return reduce(vcat, [[Integral(x in DomainSets.ClosedInterval(s.vars.intervals[x][1], Num(x)))(u) => euler_integral(Idx(II, s, u, indexmap), s, (x2i(s, u, x), x), u, ufunc)
                          for x in params(u, s)]
                         for u in depvars])
end
