# use the trapezoid rule
function _euler_integral(II, s, jx, u, ufunc, dx) #where {T,N,Wind,DX<:Number}
    j, x = jx
    if II[j] == 1 # recursively arrived at lower end of the domain
        return Num(0)
    end
    # unit index in direction of the derivative
    I1 = unitindex(ndims(u, s), j)
    # dx for multiplication
    Itap = [II - I1, II]

    if dx isa Number # Uniform dx
        weights = [dx / 2, dx / 2]
    elseif dx isa AbstractVector # Nonuniform dx
        weights = fill(dx[II[j] - 1] / 2, 2)
    else
        error("Unsupported type of dx: $(type(dx))")
    end
    
    # sym_do computes from II to II - I1, 
    # and recursive call computes from II - I1 to lower end of domain
    return sym_dot(weights, ufunc(u, Itap, x)) +
           _euler_integral(II - I1, s, jx, u, ufunc, dx)
end
function _euler_integral_II_to_upper(II, s, jx, u, ufunc, dx) #where {T,N,Wind,DX<:Number}
    j, x = jx
    if II[j] == length(s, x) # recursively arrived at upper end of the domain
        return Num(0)
    end
    # unit index in direction of the derivative
    I1 = unitindex(ndims(u, s), j)
    # dx for multiplication
    Itap = [II, II + I1]

    if dx isa Number # Uniform dx
        weights = [dx / 2, dx / 2]
    elseif dx isa AbstractVector # Nonuniform dx
        weights = fill(dx[II[j] + 1] / 2, 2)
    else
        error("Unsupported type of dx: $(type(dx))")
    end
    
    # sym_do computes from II to II + I1, 
    # and recursive call computes from II + I1 to upper end of domain
    return sym_dot(weights, ufunc(u, Itap, x)) +
           _euler_integral_II_to_upper(II + I1, s, jx, u, ufunc, dx)
end

# An integral from II to end of domain
function euler_integral(method::Symbol, II, s, jx, u, ufunc)
    j, x = jx
    dx = s.dxs[x]
    if method == :lower_boundary_to_x
        return _euler_integral(II, s, jx, u, ufunc, dx)
    elseif method == :x_to_upper_boundary
        return _euler_integral_II_to_upper(II, s, jx, u, ufunc, dx)
        # error("Method :x_to_upper_boundary is not implemented for euler_integral.")
    else
        error("Unsupported method for euler_integral: $method")
    end
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

@inline function generate_euler_integration_rules(
        II::CartesianIndex, s::DiscreteSpace, depvars, indexmap, terms)
    ufunc(u, I, x) = s.discvars[u][I]

    eulerrules = reduce(safe_vcat, [
        reduce(safe_vcat, [
        [
            # integrals from lower domain end to x:
            Integral(x in DomainSets.ClosedInterval(s.vars.intervals[x][1], Num(x)))(u) =>  euler_integral(:lower_boundary_to_x, Idx(II, s, u, indexmap), s, (x2i(s, u, x), x), u, ufunc), 
            # integrals from x to upper domain end:
            Integral(x in DomainSets.ClosedInterval(Num(x), s.vars.intervals[x][2]))(u) =>  euler_integral(:x_to_upper_boundary, Idx(II, s, u, indexmap), s, (x2i(s, u, x), x), u, ufunc),
            # TODO: # any other arbitrary integrals???
            # TODO...
        ]
    for x in ivs(u, s)]) for u in depvars] )

    return eulerrules
end

function wd_integral_Idx(II::CartesianIndex, s::DiscreteSpace, u, x, indexmap)
    # We need to construct a new index as indices may be of different size
    length(ivs(u, s)) == 0 && return CartesianIndex()
    # A hack using the boundary value re-indexing function to get an index that will work
    u_ = substitute(u, [x => s.axies[x][end]])
    II = newindex(u_, II, s, indexmap)
    return II
end

@inline function generate_whole_domain_integration_rules(
        II::CartesianIndex, s::DiscreteSpace, depvars, indexmap, terms, bvar = nothing)
    ufunc(u, I, x) = s.discvars[u][I]
    wholedomainrules = reduce(safe_vcat,
        [[Integral(x in DomainSets.ClosedInterval(s.vars.intervals[x][1],
              s.vars.intervals[x][2]))(u) => whole_domain_integral(
              wd_integral_Idx(II, s, u, x, indexmap), s, (x2i(s, u, x), x), u, ufunc)
          for x in filter(x -> (!haskey(indexmap, x) | isequal(x, bvar)), ivs(u, s))]
         for u in depvars],
        init = [])
    return wholedomainrules
end
