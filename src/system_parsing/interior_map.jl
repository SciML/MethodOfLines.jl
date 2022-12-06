struct InteriorMap
    var
    pde
    I
    lower
    upper
    stencil_extents
end

#to get an equal mapping, you want to associate every equation to a unique dependent variable that it's solving for
# this is tearing
# so there's u and v
# first equation has second derivative of u so we assign that one to u
# second equation, assume it has second derivative of both
# then we assign v to it because u is already assigned somewhere else.
# and use the interior based on the assignment

function InteriorMap(pdes, boundarymap, s::DiscreteSpace{N,M}, discretization) where {N,M}
    @assert length(pdes) == M "There must be the same number of equations and unknowns, got $(length(pdes)) equations and $(M) unknowns"
    m = buildmatrix(pdes, s)
    varmap = Dict(build_variable_mapping(m, s.ū, pdes))

    # Determine the interiors for each pde
    vlower = []
    vupper = []
    extents = []

    interior = map(pdes) do pde
        u = varmap[pde]
        boundaries = mapreduce(x -> boundarymap[operation(u)][x], vcat, s.x̄)
        n = ndims(u, s)
        lower = zeros(Int, n)
        upper = zeros(Int, n)
        # Determine thec number of points to remove from each end of the domain for each dimension
        for b in boundaries
            #@show b
            clip_interior!!(lower, upper, s, b)
        end
        push!(vlower, pde => lower)
        push!(vupper, pde => upper)
        #TODO: Allow assymmetry
        pdeorders = Dict(map(x -> x => d_orders(x, [pde]), s.x̄))

        # Add ghost points to pad stencil extents
        lowerextents, upperextents = calculate_stencil_extents(s, u, discretization, pdeorders, boundarymap)
        push!(extents, pde => (lowerextents, upperextents))
        lower = [max(e, l) for (e, l) in zip(lowerextents, lower)]
        upper = [max(e, u) for (e, u) in zip(upperextents, upper)]

        # Don't update this x2i, it is correct.
        pde => generate_interior(lower, upper, u, s, discretization)
    end


    pdemap = [k.second => k.first for k in varmap]
    return InteriorMap(varmap, Dict(pdemap), Dict(interior), Dict(vlower), Dict(vupper), Dict(extents))
end

function generate_interior(lower, upper, u, s, disc::MOLFiniteDifference{G,D}) where {G, D<:ScalarizedDiscretization}
    args = remove(arguments(u), s.time)
    return s.Igrid[u][[(1+lower[x2i(s, u, x)]:length(s.grid[x])-upper[x2i(s, u, x)]) for x in args]...]
end

function generate_interior(lower, upper, u, s, disc::MOLFiniteDifference{G, D}) where {G, D<:ArrayDiscretization}
    args = remove(arguments(u), s.time)
    return [(1+lower[x2i(s, u, x)], length(s.grid[x])-upper[x2i(s, u, x)]) for x in args]
end

function calculate_stencil_extents(s, u, discretization, orders, bcmap)
    aorder = discretization.approx_order
    advection_scheme = discretization.advection_scheme

    args = remove(arguments(u), s.time)
    lowerextents = zeros(Int, length(args))
    upperextents = zeros(Int, length(args))

    for (j,x) in enumerate(args)
        # Skip if periodic in x
        haslower, hasupper = haslowerupper(filter_interfaces(bcmap[operation(u)][x]), x)
        for dorder in filter(isodd, orders[x])
            if !haslower
                lowerextents[j] = max(lowerextents[j], extent(advection_scheme, dorder))
            end
            if !hasupper
                upperextents[j] = max(upperextents[j], extent(advection_scheme, dorder))
            end
        end
    end
    return lowerextents, upperextents
end

function buildmatrix(pdes, s::DiscreteSpace{N,M}) where {N,M}
    m = zeros(Int, M, M)
    elegiblevars = [getvars(pde, s) for pde in pdes]
    u2i = Dict([u => k for (k, u) in enumerate(s.ū)])
    #@show elegiblevars, s.ū
    for (i, varmap) in enumerate(elegiblevars)
        for var in keys(varmap)
            m[i, u2i[var]] = varmap[var]
        end
    end
    return m
end

function build_variable_mapping(m, vars, pdes)
    notzero(x) = x > 0 ? 1 : 0
    varpdemap = []
    N = length(pdes)
    rows = sum(m, dims=2)
    cols = sum(m, dims=1)
    i = findfirst(isequal(0), rows)
    j = findfirst(isequal(0), cols)
    @assert i === nothing "Equation $(pdes[i[1]]) is not an equation for any of the dependent variables."
    @assert j === nothing "Variable $(vars[j[2]]) does not appear in any equation, therefore cannot be solved for"
    for k in 1:N
        # Check if any of the pdes only have one valid variable
        m_ones = notzero.(m)
        cols = sum(m_ones, dims=1)
        j = findfirst(isequal(1), cols)
        if j !== nothing
            j = j[2]
            for i in 1:N
                if m[i, j] > 0
                    push!(varpdemap, pdes[i] => vars[j])
                    m[i, :] .= 0
                    m[:, j] .= 0
                    break
                end
            end
            continue
        end
        # Check if any of the variables only have one valid pde
        rows = sum(m_ones, dims=2)
        i = findfirst(isequal(1), rows)
        if i !== nothing
            i = i[1]
            for j in 1:N
                if m[i, j] > 0
                    push!(varpdemap, pdes[i] => vars[j])
                    m[i, :] .= 0
                    m[:, j] .= 0
                    break
                end
            end
            continue
        end
        # Check if any of the variables have more than one valid pde, and pick one
        I = findmax(m)[2]
        i = I[1]
        j = I[2]
        push!(varpdemap, pdes[i] => vars[j])
        m[i, :] .= 0
        m[:, j] .= 0
    end
    @assert length(varpdemap) == N "Could not map all PDEs to variables to solve for, the system is unbalanced."
    return varpdemap
end

@inline function split(children)
    count = [child[1] for child in children]
    # if there are more than one depvars at that order, include all of them
    is = findall(a -> a == maximum(count), count)
    vars = unique(reduce(vcat, [child[2] for child in children[is]]))
    return (sum(count), vars)
end
"""
Creates a ranking of the variables in the term, based on their derivative order.
The heuristic that should work is, if there's a time derivative then use that variable, otherwise use the highest derivative for that variable. If there are two with the highest derivative, pick first from the list that hasn't been chosen for another equation
"""
function get_ranking!(varmap, term, x, s)
    if !istree(term)
        return (0, [])
    end
    S = Symbolics
    SU = SymbolicUtils
    #@show term
    if findfirst(isequal(term), s.ū) !== nothing
        if varmap[term] < 1
            varmap[term] = 1
        end
        return (1, [term])
    else
        op = SU.operation(term)
        children = map(arg -> get_ranking!(varmap, arg, x, s), SU.arguments(term))
        count, vars = split(children)
        if op isa Differential && isequal(op.x, x)
            for var in vars
                if varmap[var] < count + 1
                    varmap[var] = count + 1
                end
            end
            return (1 + count, vars)
        end
        return (count, vars)
    end
end

function getvars(pde, s)
    ct = 0
    ut = []
    # Create ranking for each variable
    varmap = Dict([u => 0 for u in s.ū])
    if s.time !== nothing
        l = get_ranking!(varmap, pde.lhs, s.time, s)
        r = get_ranking!(varmap, pde.rhs, s.time, s)
        for u in s.ū
            if varmap[u] > 1
                varmap[u] += div(typemax(Int), 2) #Massively weight derivatives in time
            end
        end
    end
    for x in s.x̄
        l = get_ranking!(varmap, pde.lhs, x, s)
        r = get_ranking!(varmap, pde.rhs, x, s)
    end
    return varmap
end
