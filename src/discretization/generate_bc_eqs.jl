idx(b::LowerBoundary, s) = 1
idx(b::UpperBoundary, s) = length(s, b.x)
idx(b::HigherOrderInterfaceBoundary, s) = length(s, b.x)

@inline function edge(interiormap, s, u, j, islower)
    I = interiormap.I[interiormap.pde[depvar(u, s)]]
    # check needed on v1.6
    length(I) == 0 && return CartesianIndex{0}[]
    sd(i) = selectdim(I, j, i)
    I1 = unitindex(ndims(u, s), j)
    if islower
        edge = sd(1)
        # cast the edge of the interior to the edge of the boundary
        edge = edge .- [I1 * (edge[1][j] - 1)]
    else
        edge = sd(size(interiormap.I[interiormap.pde[depvar(u, s)]], j))
        edge = edge .+ [I1 * (size(s.discvars[depvar(u, s)], j) - edge[1][j])]
    end
    return edge
end

edge(s, b, interiormap) = edge(interiormap, s, b.u, x2i(s, b.u, b.x), !isupper(b))

@inline function generate_bc_eqs!(disc_state, s, boundaryvalfuncs, interiormap, boundary::AbstractTruncatingBoundary)
    args = ivs(depvar(boundary.u, s), s)
    indexmap = Dict([args[i] => i for i in 1:length(args)])
    vcat!(disc_state.bceqs, generate_bc_eqs(s, boundaryvalfuncs, boundary, interiormap, indexmap))
end

function generate_bc_eqs!(disc_state, s::DiscreteSpace, boundaryvalfuncs, interiormap, boundary::InterfaceBoundary)
    isupper(boundary) && return
    u_ = boundary.u
    x_ = boundary.x
    u__ = boundary.u2
    x__ = boundary.x2
    N = ndims(u_, s)
    j = x2i(s, depvar(u_, s), x_)
    # * Assume that the interface BC is of the simple form u(t,0) ~ u(t,1)
    Ioffset = unitindex(N, j) * (length(s, x__) - 1)
    disc1 = s.discvars[depvar(u_, s)]
    disc2 = s.discvars[depvar(u__, s)]

    vcat!(disc_state.bceqs, vec(map(edge(s, boundary, interiormap)) do II
        disc1[II] ~ disc2[II+Ioffset]
    end))

end

function generate_boundary_val_funcs(s, depvars, boundarymap, indexmap, derivweights)
    return mapreduce(vcat, values(boundarymap)) do boundaries
        map(mapreduce(x -> boundaries[x], vcat, s.x̄)) do b
            # No interface values in equations
            if b isa InterfaceBoundary
                II -> []
                # Only make a map if it is actually possible to substitute in the boundary value given the indexmap
            elseif all(x -> haskey(indexmap, x), filter(x -> !(safe_unwrap(x) isa Number), b.indvars))
                II -> boundary_value_maps(II, s, b, derivweights, indexmap)
            else
                II -> []
            end
        end
    end
end

function boundary_value_maps(II, s::DiscreteSpace{N,M,G}, boundary, derivweights, indexmap) where {N,M,G<:EdgeAlignedGrid}
    u_, x_ = getvars(boundary)

    ufunc(v, I, x) = s.discvars[v][I]

    # depvarbcmaps will dictate what to replace the variable terms with in the bcs
    # replace u(t,0) with u₁, etc

    u = depvar(u_, s)
    args = ivs(u, s)
    j = findfirst(isequal(x_), args)
    IIold = II
    # We need to construct a new index in case the value at the boundary appears in an equation one dimension lower
    II = newindex(u_, II, s, indexmap)

    val = filter(z -> z isa Number, arguments(u_))[1]
    r = x_ => val
    othervars = map(boundary.depvars) do v
        substitute(v, r)
    end
    othervars = filter(v -> (length(arguments(v)) != 1) && any(isequal(x_), arguments(depvar(v, s))), othervars)

    # Shift depending on the boundary
    shift(::LowerBoundary) = zero(II)
    shift(::UpperBoundary) = unitindex(N, j)

    depvarderivbcmaps = [(Differential(x_)^d)(u_) => half_offset_centered_difference(derivweights.halfoffsetmap[1][Differential(x_)^d], II - shift(boundary), s, [], (j, x_), u, ufunc) for d in derivweights.orders[x_]]

    depvarbcmaps = [v_ => half_offset_centered_difference(derivweights.interpmap[x_], II - shift(boundary), s, [], (x2i(s, depvar(v_, s), s), x_), depvar(v_, s), ufunc) for v_ in [u_; othervars]]

    # Only make a map if the integral will actually come out to the same number of dimensions as the boundary value
    integralvs = filter(v -> !any(x -> safe_unwrap(x) isa Number, arguments(v)), boundary.depvars)
    # @show integralvs

    integralbcmaps = generate_whole_domain_integration_rules(IIold, s, integralvs, indexmap, nothing, x_)

    if boundary isa HigherOrderInterfaceBoundary
        u__ = boundary.u2
        x__ = boundary.x2
        otheru = depvar(u__, s)

        j = x2i(s, otheru, x__)
        is = [II[i] for i in 1:length(II)]
        is = [is[1:j-1]..., 1, is[j:end]...]
        II = CartesianIndex(is...)

        depvarderivbcmaps = [(Differential(x__)^d)(u__) => half_offset_centered_difference(derivweights.halfoffsetmap[1][Differential(x__)^d], II, s, [], (j, x__), otheru, ufunc) for d in derivweights.orders[x_]]

        depvarbcmaps = [u__ => half_offset_centered_difference(derivweights.interpmap[x__], II, s, [], (j, x__), otheru, ufunc)]

        depvarderivbcmaps = vcat(depvarderivbcmaps, otherderivmaps)
        depvarbcmaps = vcat(depvarbcmaps, otherbcmaps)
    end


    return vcat(depvarderivbcmaps, integralbcmaps, depvarbcmaps)
end

function boundary_value_maps(II, s::DiscreteSpace{N,M,G}, boundary, derivweights, indexmap) where {N,M,G<:CenterAlignedGrid}
    u_, x_ = getvars(boundary)
    ufunc(v, I, x) = s.discvars[v][I]

    depvarderivbcmaps = []
    depvarbcmaps = []

    # * Assume that the BC is in terms of an explicit expression, not containing references to variables other than u_ at the boundary
    u = depvar(u_, s)
    args = ivs(u, s)
    j = findfirst(isequal(x_), args)
    IIold = II
    # We need to construct a new index in case the value at the boundary appears in an equation one dimension lower
    II = newindex(u_, II, s, indexmap)
    val = filter(z -> z isa Number, arguments(u_))[1]
    r = x_ => val
    othervars = map(boundary.depvars) do v
        substitute(v, r)
    end
    othervars = filter(v -> (length(arguments(v)) != 1) && any(isequal(x_), arguments(depvar(v, s))), othervars)

    depvarderivbcmaps = [(Differential(x_)^d)(u_) => central_difference(derivweights.map[Differential(x_)^d], II, s, [], (x2i(s, u, x_), x_), u, ufunc) for d in derivweights.orders[x_]]
    depvarbcmaps = [v_ => s.discvars[depvar(v_, s)][II] for v_ in [u_; othervars]]

    # Only make a map if the integral will actually come out to the same number of dimensions as the boundary value
    integralvs = unwrap.(filter(v -> !any(x -> safe_unwrap(x) isa Number, arguments(v)), boundary.depvars))

    integralbcmaps = generate_whole_domain_integration_rules(IIold, s, integralvs, indexmap, nothing, x_)

    # Deal with the other relevant variables if boundary isa InterfaceBoundary
    if boundary isa HigherOrderInterfaceBoundary
        u__ = boundary.u2
        x__ = boundary.x2
        otheru = depvar(u__, s)

        is = [II[i] for i in setdiff(1:length(II), [j])]
        j = x2i(s, otheru, x__)

        is = vcat(is[1:j-1], 1, is[j:end])
        II = CartesianIndex(is...)

        otherderivmaps = [(Differential(x__)^d)(u__) => central_difference(derivweights.map[Differential(x__)^d], II, s, [], (x2i(s, otheru, x__), x__), otheru, ufunc) for d in derivweights.orders[x__]]
        otherbcmaps = [u__ => s.discvars[otheru][II]]

        depvarderivbcmaps = vcat(depvarderivbcmaps, otherderivmaps)
        depvarbcmaps = vcat(depvarbcmaps, otherbcmaps)
    end

    return vcat(depvarderivbcmaps, integralbcmaps, depvarbcmaps)
end


function generate_bc_eqs(s::DiscreteSpace{N,M,G}, boundaryvalfuncs, boundary::AbstractTruncatingBoundary, interiormap, indexmap) where {N,M,G}
    bc = boundary.eq
    return vec(map(edge(s, boundary, interiormap)) do II
        boundaryvalrules = mapreduce(f -> f(II), vcat, boundaryvalfuncs)
        vmaps = varmaps(s, boundary.depvars, II, indexmap)
        varrules = axiesvals(s, depvar(boundary.u, s), boundary.x, II)
        rules = vcat(boundaryvalrules, vmaps, varrules)

        substitute(bc.lhs, rules) ~ substitute(bc.rhs, rules)
    end)
end

"""
`generate_extrap_eqs`

Pads the boundaries with extrapolation equations, extrapolated with 6th order lagrangian polynomials.
Reuses `central_difference` as this already dispatches the correct stencil, given a `DerivativeOperator` which contains the correct weights.
"""
function generate_extrap_eqs!(disc_state, pde, u, s, derivweights, interiormap, bcmap)
    args = ivs(u, s)
    length(args) == 0 && return

    lowerextents, upperextents = interiormap.stencil_extents[pde]
    vlower = interiormap.lower[pde]
    vupper = interiormap.upper[pde]
    ufunc(u, I, x) = s.discvars[u][I]

    eqmap = [[] for _ in CartesianIndices(s.discvars[u])]
    for (j, x) in enumerate(args)
        ninterp = lowerextents[j] - vlower[j]
        I1 = unitindex(length(args), j)
        bs = bcmap[operation(u)][x]
        haslower, hasupper = haslowerupper(bs, x)
        while ninterp >= vlower[j]
            if haslower
                break
            end
            for Il in (edge(interiormap, s, u, j, true) .+ (ninterp * I1,))
                expr = central_difference(derivweights.boundary[x], Il, s, filter_interfaces(bcmap[operation(u)][x]), (j, x), u, ufunc)
                push!(eqmap[Il], expr)
            end
            ninterp = ninterp - 1
        end
        ninterp = upperextents[j] - vupper[j]
        while ninterp >= vupper[j]
            if hasupper
                break
            end
            for Iu in (edge(interiormap, s, u, j, false) .- (ninterp * I1,))
                expr = central_difference(derivweights.boundary[x], Iu, s, filter_interfaces(bcmap[operation(u)][x]), (j, x), u, ufunc)
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
            push!(disc_state.bceqs, s.discvars[u][II] ~ rhss[1])
        else
            n = length(rhss)
            push!(disc_state.bceqs, s.discvars[u][II] ~ sum(rhss) / n)
        end
    end
end

#TODO: Benchmark and optimize this

@inline function generate_corner_eqs!(disc_state, s, interiormap, N, u)
    interior = interiormap.I[interiormap.pde[u]]
    ndims(u, s) == 0 && return
    sd(i, j) = selectdim(interior, j, i)
    domain = setdiff(s.Igrid[u], interior)
    II1 = unitindices(N)
    for j in 1:N
        I1 = II1[j]
        edge = sd(1, j)
        offset = edge[1][j] - 1
        for k in 1:offset
            setdiff!(domain, vec(copy(edge) .- [I1 * k]))
        end
        edge = sd(size(interior, j), j)
        offset = size(s.discvars[u], j) - size(interior, j)
        for k in 1:offset
            setdiff!(domain, vec(copy(edge) .+ [I1 * k]))
        end
    end
    append!(disc_state.bceqs, s.discvars[u][domain] .~ 0)
end

"""
Create a vector containing indices of the corners of the domain.
"""
@inline function findcorners(s::DiscreteSpace, lower, upper, u)
    args = remove(arguments(u), s.time)
    if any(lower .== 0) && any(upper .== 0)
        return CartesianIndex{2}[]
    end
    return reduce(vcat, vec.(map(0:3) do n
        dig = digits(n, base=2, pad=2)
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

@inline function generate_corner_eqs!(disc_state, s, interiormap, pde)
    u = interiormap.var[pde]
    N = ndims(u, s)
    if N <= 1
        return
    elseif N == 2
        Icorners = findcorners(s, interiormap.lower[pde], interiormap.upper[pde], u)
        append!(disc_state.bceqs, s.discvars[u][Icorners] .~ 0)
    else
        generate_corner_eqs!(disc_state, s, interiormap, N, u)
    end
end
