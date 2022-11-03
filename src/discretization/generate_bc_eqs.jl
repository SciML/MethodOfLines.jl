@inline function generate_bc_eqs!(bceqs, s, boundaryvalfuncs, interiormap, boundary::AbstractTruncatingBoundary)
    args = params(depvar(boundary.u, s), s)
    indexmap = Dict([args[i]=>i for i in 1:length(args)])
    push!(bceqs, generate_bc_eqs(s, boundaryvalfuncs, boundary, interiormap, indexmap))
end

function generate_bc_eqs!(bceqs, s::DiscreteSpace{N}, boundaryvalfuncs, interiormap, boundary::PeriodicBoundary) where N
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

function generate_boundary_val_funcs(s, depvars, boundarymap, indexmap, derivweights)
    return mapreduce(vcat, values(boundarymap)) do boundaries
        map(mapreduce(x -> boundaries[x], vcat, s.x̄)) do b
            if b isa PeriodicBoundary
                II -> []
            else
                II -> boundary_value_maps(II, s, b, derivweights, indexmap)
            end
        end
    end
end

function boundary_value_maps(II::CartesianIndex, s::DiscreteSpace{N,M,G}, boundary, derivweights, indexmap) where {N,M,G<:EdgeAlignedGrid}
    u_, x_ = getvars(boundary)

    ufunc(v, I, x) = s.discvars[v][I]

    # depvarbcmaps will dictate what to replace the variable terms with in the bcs
    # replace u(t,0) with u₁, etc

    u = depvar(u_, s)
    args = params(u, s)
    j = findfirst(isequal(x_), args)

    # We need to construct a new index in case the value at the boundary appears in an equation one dimension lower
    is = [II[indexmap[x]] for x in filter(!isequal(x_), args)]

    is = [is[1:j-1]..., idx(boundary, s), is[j:end]...]
    II = CartesianIndex(is...)

    # Shift depending on the boundary
    shift(::LowerBoundary) = zero(II)
    shift(::UpperBoundary) = unitindex(N, j)

    depvarderivbcmaps = [(Differential(x_)^d)(u_) => half_offset_centered_difference(derivweights.halfoffsetmap[1][Differential(x_)^d], II-shift(boundary), s, isperiodic(boundary), (j,x_), u, ufunc) for d in derivweights.orders[x_]]

    depvarbcmaps = [u_ => half_offset_centered_difference(derivweights.interpmap[x_], II-shift(boundary), s, isperiodic(boundary), (j,x_), u, ufunc)]

    return vcat(depvarderivbcmaps, depvarbcmaps)
end

function boundary_value_maps(II::CartesianIndex, s::DiscreteSpace{N,M,G}, boundary, derivweights, indexmap) where {N,M,G<:CenterAlignedGrid}
    u_, x_ = getvars(boundary)
    ufunc(v, I, x) = s.discvars[v][I]

    depvarderivbcmaps = []
    depvarbcmaps = []

    # * Assume that the BC is in terms of an explicit expression, not containing references to variables other than u_ at the boundary
    u = depvar(u_, s)
    args = params(u, s)
    j = findfirst(isequal(x_), args)

    # We need to construct a new index in case the value at the boundary appears in an equation one dimension lower
    is = [II[indexmap[x]] for x in filter(!isequal(x_), args)]

    is = [is[1:j-1]..., idx(boundary, s), is[j:end]...]
    II = CartesianIndex(is...)

    depvarderivbcmaps = [(Differential(x_)^d)(u_) => central_difference(derivweights.map[Differential(x_)^d], II, s,isperiodic(boundary), (x2i(s, u, x_), x_), u, ufunc) for d in derivweights.orders[x_]]
    # ? Does this need to be done for all variables at the boundary?
    depvarbcmaps = [u_ => s.discvars[u][II]]

    return vcat(depvarderivbcmaps, depvarbcmaps)
end


function generate_bc_eqs(s::DiscreteSpace{N,M,G}, boundaryvalfuncs, boundary::AbstractTruncatingBoundary, interiormap, indexmap) where {N, M, G}
    bc = boundary.eq
    return vec(map(edge(s, boundary, interiormap)) do II
        boundaryvalrules = mapreduce(f -> f(II), vcat, boundaryvalfuncs)
        varrules = varmaps(s, boundary.depvars, II, indexmap)
        valrules = axiesvals(s, depvar(boundary.u, s), boundary.x, II)
        rules = vcat(boundaryvalrules, varrules, valrules)

        substitute(bc.lhs, rules) ~ substitute(bc.rhs, rules)
    end)
end

"""
`generate_extrap_eqs`

Pads the boundaries with extrapolation equations, extrapolated with 6th order lagrangian polynomials.
Reuses `central_difference` as this already dispatches the correct stencil, given a `DerivativeOperator` which contains the correct weights.
"""
function generate_extrap_eqs!(eqs, pde, u, s, derivweights, interiormap, periodicmap)
    args = remove(arguments(u), s.time)
    extents = interiormap.stencil_extents[pde]
    vlower = interiormap.lower[pde]
    vupper = interiormap.upper[pde]
    pmap = periodicmap.map[operation(u)]
    ufunc(u, I, x) = s.discvars[u][I]

    eqmap = [[] for _ in CartesianIndices(s.discvars[u])]
    for (j, x) in enumerate(args)
        pmap[x] isa Val{true} && continue
        ninterp = extents[j] - vlower[j]
        I1 = unitindex(length(args), j)
        while ninterp >= vlower[j]
            for Il in (edge(interiormap, s, u, j, true) .+ (ninterp * I1,))
                expr = central_difference(derivweights.boundary[x], Il, s, pmap[x], (j, x), u, ufunc)
                push!(eqmap[Il], expr)
            end
            ninterp = ninterp - 1
        end
        ninterp = extents[j] - vupper[j]
        while ninterp >= vupper[j]
            for Iu in (edge(interiormap, s, u, j, false) .- (ninterp * I1,))
                expr = central_difference(derivweights.boundary[x], Iu, s, pmap[x], (j, x), u, ufunc)
                push!(eqmap[Iu], expr)
            end
            ninterp = ninterp - 1
        end
    end
    # Overlap handling
    for II in setdiff(collect(CartesianIndices(eqmap)), interiormap.I[pde])
        rhss = eqmap[II]
        if length(rhss) == 0
            continue
        elseif length(rhss) == 1
            push!(eqs, s.discvars[u][II] ~ rhss[1])
        else
            n = length(rhss)
            push!(eqs, s.discvars[u][II] ~ sum(rhss)/n)
        end
    end
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
@inline function findcorners(s::DiscreteSpace, lower, upper, u)
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
