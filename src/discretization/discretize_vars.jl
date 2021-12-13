#Base.getindex(s::DiscreteSpace, i::Integer) = DiscreteVar(vars[i], ...)

@inline function get_edgevals(params, axies, i)
    return [params[i] => first(axies[i]), params[i] => last(axies[i])]
end


@inline function get_edgevals(s, i)
    return [s.params[i] => first(s.axies[i]), s.params[i] => last(s.axies[i])]
end

@inline function subvar(depvar, edge_vals)
    return substitute.((depvar,), edge_vals)
end
struct DiscreteSpace{N,M}
    vars
    discvars
    params # Note that these aren't necessarily @params
    axies
    grid
    dxs
    Iaxies
    Igrid
    Iedge
end

function DiscreteSpace(domain, depvars, nottime, grid_align, discretization)
    t = discretization.time
    nspace = length(nottime)
    # Discretize space
    axies = map(nottime) do x
        xdomain = domain[findfirst(d -> isequal(x, d.variables), domain)]
        dx = discretization.dxs[findfirst(dxs -> isequal(x, dxs[1].val), discretization.dxs)][2]
        dx isa Number ? (DomainSets.infimum(xdomain.domain):dx:DomainSets.supremum(xdomain.domain)) : dx
    end
    dxs = map(nottime) do x
        dx = discretization.dxs[findfirst(dxs -> isequal(x, dxs[1].val), discretization.dxs)][2]
    end

    # Define the grid on which the dependent variables will be evaluated (see #378)
    # center_align is recommended for Dirichlet BCs
    # edge_align is recommended for Neumann BCs (spatial discretization is conservative)
    if grid_align == center_align
        grid = axies
    elseif grid_align == edge_align
        # boundary conditions implementation assumes centered_order=2
        @assert discretization.centered_order == 2
        # construct grid including ghost nodes beyond outer edges
        # e.g. space 0:dx:1 goes to grid -dx/2:dx:1+dx/2
        space_ext = map(s -> vcat(2s[1] - s[2], s, 2s[end] - s[end-1]), axies)
        grid = map(s -> (s[1:end-1] + s[2:end]) / 2, space_ext)
        # TODO: allow depvar-specific center/edge choice?
    end

    # Build symbolic variables
    Iaxies = CartesianIndices(((axes(s)[1] for s in axies)...,))
    Igrid = CartesianIndices(((axes(g)[1] for g in grid)...,))
    depvarsdisc = map(depvars) do u
        if t === nothing
            sym = nameof(operation(u))
            collect(first(@variables $sym[collect(axes(g)[1] for g in grid)...]))
        elseif isequal(arguments(u), [t])
            [u for II in s.Igrid]
        else
            sym = nameof(operation(u))
            collect(first(@variables $sym[collect(axes(g)[1] for g in grid)...](t)))
        end
    end


    # Build symbolic maps for boundaries
    Iedge = reduce(vcat, [[vcat([Colon() for j = 1:i-1], 1, [Colon() for j = i+1:nspace]),
        vcat([Colon() for j = 1:i-1], length(axies[i]), [Colon() for j = i+1:nspace])] for i = 1:nspace])

    return DiscreteSpace{nspace,length(depvars)}(depvars, depvarsdisc, nottime, axies, grid, dxs, Iaxies, Igrid, Iedge)
end

nparams(::DiscreteSpace{N,M}) where {N,M} = N
nvars(::DiscreteSpace{N,M}) where {N,M} = M

grid_idxs(s::DiscreteSpace) = CartesianIndices(((axes(g)[1] for g in s.grid)...,))
edge_idxs(s::DiscreteSpace{N}) where {N} = reduce(vcat, [[vcat([Colon() for j = 1:i-1], 1, [Colon() for j = i+1:N]),
    vcat([Colon() for j = 1:i-1], length(s.axies[i]), [Colon() for j = i+1:N])] for i = 1:N])

axiesvals(s::DiscreteSpace{N}) where {N} = map(y -> [s.params[i] => s.axies[i][y.I[i]] for i = 1:N], s.Iaxies)
gridvals(s::DiscreteSpace{N}) where {N} = map(y -> [s.params[i] => s.grid[i][y.I[i]] for i = 1:N], s.Igrid)

## Boundary methods ##
edgevals(s::DiscreteSpace{N}) where {N} = reduce(vcat, [get_edgevals(s.params, s.axies, i) for i = 1:N])
edgevars(s::DiscreteSpace) = [[d[e...] for e in s.Iedge] for d in s.discvars]

@inline function edgemaps(t, s::DiscreteSpace)
    bclocs(s::DiscreteSpace) = map(e -> substitute.(vcat(t, s.params), e), edgevals(s))
    return Dict(bclocs(s) .=> [axiesvals(s)[e...] for e in s.Iedge])
end

map_symbolic_to_discrete(II::CartesianIndex, s::DiscreteSpace{N,M}) where {N,M} = vcat([s.vars[k] => s.discvars[k][II] for k = 1:M], [s.params[j] => s.grid[j][II[j]] for j = 1:N])


#varmap(s::DiscreteSpace{N,M}) where {N,M} = [s.vars[i] => i for i = 1:M]

"""
Returns a list of index offsets from the boundary, where 1 means that the index is on the lower boundary, and a value of 0 means that the index is on the interior. Likewise, -1 means that the index is on the upper boundary. -2 would be 1 index off of the boundary etc. The list is of length N, where N is the number of dimensions. 
"""
function edgeoffset(s::DiscreteSpace{N}, I::CartesianIndex{N}, padding::T) where {N,T<:Integer}
    Ioffset = zeros(T, N)
    for (i, x) in s.grid
        if I[i] <= padding
            Ioffset[i] = padding - I[i] + 1
        elseif I[i] > length(x) - padding
            Ioffset[i] = I[i] - length(x) - 1
        end
    end
    return Ioffset
end