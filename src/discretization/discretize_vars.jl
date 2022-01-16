#Base.getindex(s::DiscreteSpace, i::Integer) = DiscreteVar(vars[i], ...)
"""
Return a map of of variables to the gridpoints at the edge of the domain
"""
@inline function get_edgevals(params, axies, i)
    return [params[i] => first(axies[i]), params[i] => last(axies[i])]
end

"""
Return a map of of variables to the gridpoints at the edge of the domain
"""
@inline function get_edgevals(s, i)
    return [s.nottime[i] => first(s.axies[nottime[i]]), s.nottime[i] => last(s.axies[nottime[i]])]
end

@inline function subvar(depvar, edge_vals)
    return substitute.((depvar,), edge_vals)
end
struct DiscreteSpace{N,M}
    vars
    discvars
    time
    nottime # Note that these aren't necessarily @parameters
    params
    axies
    grid
    dxs
    Iaxies
    Igrid
    Iedge
    x2i
end

nparams(::DiscreteSpace{N,M}) where {N,M} = N
nvars(::DiscreteSpace{N,M}) where {N,M} = M

Base.length(s::DiscreteSpace, x) = length(s.grid[x])
Base.size(s::DiscreteSpace) = Tuple(length(s.grid[z]) for z in s.nottime)

params(s::DiscreteSpace{N,M}) where {N,M}= s.params

grid_idxs(s::DiscreteSpace) = CartesianIndices(((axes(g)[1] for g in s.grid)...,))
edge_idxs(s::DiscreteSpace{N}) where {N} = reduce(vcat, [[vcat([Colon() for j = 1:i-1], 1, [Colon() for j = i+1:N]), vcat([Colon() for j = 1:i-1], length(s.axies[i]), [Colon() for j = i+1:N])] for i = 1:N])

axiesvals(s::DiscreteSpace{N}) where {N} = map(y -> [s.nottime[i] => s.axies[s.nottime[i]][y.I[i]] for i = 1:N], s.Iaxies)
gridvals(s::DiscreteSpace{N}) where {N} = map(y -> [s.nottime[i] => s.grid[s.nottime[i]][y.I[i]] for i = 1:N], s.Igrid)

## Boundary methods ##
edgevals(s::DiscreteSpace{N}) where {N} = reduce(vcat, [get_edgevals(s.nottime, s.axies, i) for i = 1:N])
edgevars(s::DiscreteSpace) = [[d[e...] for e in s.Iedge] for d in s.discvars]

@inline function edgemaps(s::DiscreteSpace)
    bclocs(s::DiscreteSpace) = map(e -> substitute.(s.params, e), edgevals(s))
    return Dict(bclocs(s) .=> [axiesvals(s)[e...] for e in s.Iedge])
end

Iinterior(s::DiscreteSpace) = s.Igrid[[2:(length(s, x)-1) for x in s.nottime]...]

map_symbolic_to_discrete(II::CartesianIndex, s::DiscreteSpace{N,M}) where {N,M} = vcat([s.vars[k] => s.discvars[k][II] for k = 1:M], [s.nottime[j] => s.grid[j][II[j]] for j = 1:N])

# ? How rude is this? Makes Iedge work

# TODO: Allow other grids

function DiscreteSpace(domain, depvars, indvars, nottime, discretization)
    grid_align = discretization.grid_align
    t = discretization.time
    nspace = length(nottime)
    # Discretize space
    axies = map(nottime) do x
        xdomain = domain[findfirst(d -> isequal(x, d.variables), domain)]
        dx = discretization.dxs[findfirst(dxs -> isequal(x, dxs[1].val), discretization.dxs)][2]
        dx isa Number ? x => (DomainSets.infimum(xdomain.domain):dx:DomainSets.supremum(xdomain.domain)) : x => dx
    end
    dxs = map(nottime) do x
        x => discretization.dxs[findfirst(dxs -> isequal(x, dxs[1].val), discretization.dxs)][2]
    end

    # Define the grid on which the dependent variables will be evaluated (see #378)
    # center_align is recommended for Dirichlet BCs
    # edge_align is recommended for Neumann BCs (spatial discretization is conservative)
    if grid_align == center_align
        grid = axies
    elseif grid_align == edge_align
        # construct grid including ghost nodes beyond outer edges
        # e.g. space 0:dx:1 goes to grid -dx/2:dx:1+dx/2
        grid =  map(nottime) do x
            xdomain = domain[findfirst(d -> isequal(x, d.variables), domain)]
            dx = discretization.dxs[findfirst(dxs -> isequal(x, dxs[1].val), discretization.dxs)][2]
            dx isa Number ? (x => ((DomainSets.infimum(xdomain.domain)-dx/2):dx:(DomainSets.supremum(xdomain.domain)+dx/2))) : x => dx
        end 
        # TODO: allow depvar-specific center/edge choice?
    end

    # Build symbolic variables
    Iaxies = CartesianIndices(((axes(s.second)[1] for s in axies)...,))
    Igrid = CartesianIndices(((axes(g.second)[1] for g in grid)...,))

    depvarsdisc = map(depvars) do u
        if t === nothing
            sym = nameof(SymbolicUtils.operation(u))
            u => collect(first(@variables $sym[collect(axes(g.second)[1] for g in grid)...]))
        elseif isequal(SymbolicUtils.arguments(u), [t])
            u => [u for II in s.Igrid]
        else
            sym = nameof(SymbolicUtils.operation(u))
            u => collect(first(@variables $sym[collect(axes(g.second)[1] for g in grid)...](t))) 
        end
    end

    # Build symbolic maps for boundaries
    Iedge = vcat((vcat(vec(selectdim(Igrid, dim, 1)), vec(selectdim(Igrid, dim, length(grid[dim].second)))) for dim in 1:nspace)...)

    nottime2dim = [nottime[i] => i for i in 1:nspace]
    dim2nottime = [i => nottime[i] for i in 1:nspace]
    return DiscreteSpace{nspace,length(depvars)}(depvars, Dict(depvarsdisc), discretization.time, nottime, indvars, Dict(axies), Dict(grid), Dict(dxs), Iaxies, Igrid, Iedge, Dict(nottime2dim))
end



#varmap(s::DiscreteSpace{N,M}) where {N,M} = [s.vars[i] => i for i = 1:M]

