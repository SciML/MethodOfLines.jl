@inline function generate_bc_rules!(bceqs, derivweights::DifferentialDiscretizer, s, interiormap, boundary::AbstractTruncatingBoundary)
    bc = boundary.eq
    push!(bceqs, vec(map(edge(s, boundary, interiormap)) do II
        rules = generate_bc_rules(II, derivweights, s, bc, boundary)
        
        substitute(bc.lhs, rules) ~ substitute(bc.rhs, rules)
    end))
end


function generate_bc_rules(II, derivweights, s::DiscreteSpace{N,M,G}, bc, boundary::AbstractTruncatingBoundary) where {N, M, G<:CenterAlignedGrid}
    # depvarbcmaps will dictate what to replace the variable terms with in the bcs
    # replace u(t,0) with u₁, etc
    u_, x_ = getvars(boundary)
    ufunc(v, I, x) = s.discvars[v][I]

    depvarderivbcmaps = []
    depvarbcmaps = []
    
    # * Assume that the BC is in terms of an explicit expression, not containing references to variables other than u_ at the boundary
    for u in s.ū
        if isequal(operation(u), operation(u_))
            # What to replace derivatives at the boundary with
            depvarderivbcmaps = [(Differential(x_)^d)(u_) => central_difference(derivweights.map[Differential(x_)^d], II, s, (s.x2i[x_], x_), u, ufunc) for d in derivweights.orders[x_]]
            # ? Does this need to be done for all variables at the boundary?
            depvarbcmaps = [u_ => s.discvars[u][II]]
            break
        end
    end
    fd_rules = generate_finite_difference_rules(II, s, bc, derivweights)
    varrules = axiesvals(s, u_, x_, II)

    # valrules should be caught by depvarbcmaps and varrules if the above assumption holds
    #valr = valrules(s, II)
    #for condition in fd_rules.conditions
    return vcat(depvarderivbcmaps, depvarbcmaps, fd_rules, varrules)
end

function generate_bc_rules(II, derivweights, s::DiscreteSpace{N,M,G}, bc, boundary::AbstractTruncatingBoundary) where {N, M, G<:EdgeAlignedGrid}
    
    u_, x_ = getvars(boundary)
    ufunc(v, I, x) = s.discvars[v][I]

    depvarderivbcmaps = []
    depvarbcmaps = []

    # depvarbcmaps will dictate what to replace the variable terms with in the bcs
    # replace u(t,0) with u₁, etc
    # * Assume that the BC is in terms of an explicit expression, not containing references to variables other than u_ at the boundary
    j = s.x2i[x_]
    shift(::LowerBoundary) = zero(II)
    shift(::UpperBoundary) = unitindex(N, j)
    for u in s.ū
        if isequal(operation(u), operation(u_))
            depvarderivbcmaps = [(Differential(x_)^d)(u_) => half_offset_centered_difference(derivweights.halfoffsetmap[Differential(x_)^d], II-shift(boundary), s, (j,x_), u, ufunc) for d in derivweights.orders[x_]]
    
            depvarbcmaps = [u_ => half_offset_centered_difference(derivweights.interpmap[x_], II-shift(boundary), s, (j,x_), u, ufunc)]
            break
        end
    end
    
    fd_rules = generate_finite_difference_rules(II, s, bc, derivweights)
    varrules = axiesvals(s, u_, x_, II)

    # valrules should be caught by depvarbcmaps and varrules if the above assumption holds
    #valr = valrules(s, II)
    
    return vcat(depvarderivbcmaps, depvarbcmaps, fd_rules, varrules)
end

function generate_bc_rules!(bceqs, derivweights, s::DiscreteSpace{N}, interiormap, boundary::PeriodicBoundary) where N
    # depvarbcmaps will dictate what to replace the variable terms with in the bcs
    # replace u(t,0) with u₁, etc
    u_, x_ = getvars(boundary)
    j = s.x2i[x_]
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

@inline function generate_corner_eqs!(bceqs, s, interior, u)
    sd(i, j) = selectdim(interior, j, i)
    domain = vec(copy(interior))
    N = ndims(u, s)
    II1 = unitindices(N)
    for j in 1:N
        I1 = II1[j]
        edge = sd(1, j)
        offset = edge[1][j]-1
        for k in 1:offset
            vcat(domain, vec(copy(edge).-[I1*k]))
        end
        edge = sd(size(interior, j), j)
        offset = size(s.discvars[u], j) - size(interior, j)
        for k in 1:offset
            vcat(domain, vec(copy(edge).+[I1*k]))
        end
    end
    corners = setdiff(domain, vec(s.Igrid[u]))
    push!(bceqs, s.discvars[u][corners] .~ 0)
end
