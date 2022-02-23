@inline function generate_bc_rules!(bceqs, derivweights::DifferentialDiscretizer, s, interiormap, boundary::AbstractTruncatingBoundary)
    bc = boundary.eq
    push!(bceqs, vec(map(edge(s, boundary, interiormap)) do II
        rules = generate_bc_rules(II, derivweights, s, boundary)
        
        substitute(bc.lhs, rules) ~ substitute(bc.rhs, rules)
    end))
end


function generate_bc_rules(II, derivweights, s::DiscreteSpace{N,M,G}, boundary::AbstractTruncatingBoundary) where {N, M, G<:CenterAlignedGrid}
    # depvarbcmaps will dictate what to replace the variable terms with in the bcs
    # replace u(t,0) with u₁, etc
    u_, x_ = getvars(boundary)
    ufunc(v, I, x) = s.discvars[v][I]

    depvarderivbcmaps = []
    depvarbcmaps = []
    
    # * Assume that the BC is in terms of an explicit expression, not containing references to variables other than u_ at the boundary
    u = depvar(u_, s)
    depvarderivbcmaps = [(Differential(x_)^d)(u_) => central_difference(derivweights.map[Differential(x_)^d], II, s, (x2i(s, u, x_), x_), u, ufunc) for d in derivweights.orders[x_]]
    # ? Does this need to be done for all variables at the boundary?
    depvarbcmaps = [u_ => s.discvars[u][II]]

    fd_rules = generate_finite_difference_rules(II, s, boundary.depvars, boundary.eq, derivweights)
    varrules = axiesvals(s, depvar(u_,s), x_, II)

    # valrules should be caught by depvarbcmaps and varrules if the above assumption holds
    #valr = valrules(s, II)
    #for condition in fd_rules.conditions
    return vcat(depvarderivbcmaps, depvarbcmaps, fd_rules, varrules)
end

function generate_bc_rules(II, derivweights, s::DiscreteSpace{N,M,G}, boundary::AbstractTruncatingBoundary) where {N, M, G<:EdgeAlignedGrid}
    
    u_, x_ = getvars(boundary)
    ufunc(v, I, x) = s.discvars[v][I]

    depvarderivbcmaps = []
    depvarbcmaps = []

    # depvarbcmaps will dictate what to replace the variable terms with in the bcs
    # replace u(t,0) with u₁, etc
    # * Assume that the BC is in terms of an explicit expression, not containing references to variables other than u_ at the boundary
    u = depvar(u_, s)
    j = x2i(s, u, x_)
    # Shift depending on the boundary
    shift(::LowerBoundary) = zero(II)
    shift(::UpperBoundary) = unitindex(N, j)

    depvarderivbcmaps = [(Differential(x_)^d)(u_) => half_offset_centered_difference(derivweights.halfoffsetmap[Differential(x_)^d], II-shift(boundary), s, (j,x_), u, ufunc) for d in derivweights.orders[x_]]

    depvarbcmaps = [u_ => half_offset_centered_difference(derivweights.interpmap[x_], II-shift(boundary), s, (j,x_), u, ufunc)]

    fd_rules = generate_finite_difference_rules(II, s, boundary.depvars, boundary.eq, derivweights)
    varrules = axiesvals(s, u, x_, II)

    # valrules should be caught by depvarbcmaps and varrules if the above assumption holds
    #valr = valrules(s, II)
    
    return vcat(depvarderivbcmaps, depvarbcmaps, fd_rules, varrules)
end

function generate_bc_rules!(bceqs, derivweights, s::DiscreteSpace{N}, interiormap, boundary::PeriodicBoundary) where N
    # depvarbcmaps will dictate what to replace the variable terms with in the bcs
    # replace u(t,0) with u₁, etc
    u_, x_ = getvars(boundary)
    j = x2i(s, depvar(u_, s), x_)
    # * Assume that the periodic BC is of the simple form u(t,0) ~ u(t,1)
    Ioffset = unitindex(N, j)*(length(s, x_) - 1)
    local disc
    for u in s.ū
        if isequal(operation(u), operation(u_))
            disc =  s.discvars[u]
            break
        end
    end

    
    push!(bceqs, vec(map(edge(s, boundary, interiormap)) do II
        disc[II] ~ disc[II + Ioffset]
    end))

end

#TODO: Benchmark and optimize this

@inline function generate_corner_eqs!(bceqs, s, interiormap, N, u)
    interior = interiormap.I[interiormap.pde[u]]
    sd(i, j) = selectdim(interior, j, i)
    domain = setdiff(s.Igrid[u], interior)
    II1 = unitindices(N)
    for j in 1:N
        I1 = II1[j]
        edge = sd(1, j)
        offset = edge[1][j]-1
        for k in 1:offset
            setdiff!(domain, vec(copy(edge).-[I1*k]))
        end
        edge = sd(size(interior, j), j)
        offset = size(s.discvars[u], j) - size(interior, j)
        for k in 1:offset
            setdiff!(domain, vec(copy(edge).+[I1*k]))
        end
    end
    push!(bceqs, s.discvars[u][domain] .~ 0)
end

"""
Create a vector containing indices of the corners of the domain.
"""
@inline function findcorners(s::DiscreteSpace, lower, upper, u) where {M}
    args = remove(arguments(u), s.time)
    if any(lower.==0) && any(upper.==0)
        return CartesianIndex{2}[]
    end
    return reduce(vcat, vec.(map(0 : 3) do n
        dig = digits(n, base = 2, pad = 2)
        CartesianIndices(Tuple(map(enumerate(dig)) do (i, b)
            x = args[i]
            if b == 1
                1:lower[i]
            elseif b == 0
                length(s, x)-upper[i]+1:length(s, x)
            end
        end))
    end))
end

@inline function generate_corner_eqs!(bceqs, s, interiormap, pde)
    u = interiormap.var[pde]
    N = ndims(u, s)
    if N <= 1
        return
    elseif N == 2
        Icorners = findcorners(s, interiormap.lower[pde], interiormap.upper[pde], u)
        push!(bceqs, s.discvars[u][Icorners] .~ 0)
    else
        generate_corner_eqs!(bceqs, s, interiormap, N, u)
    end
end

