#Base.getindex(s::DiscreteSpace, i::Integer) = DiscreteVar(vars[i], ...)
"""
Return a map of of variables to the gridpoints at the edge of the domain
"""
@inline function get_edgevals(params, axies, i)
    return [params[i] => first(axies[params[i]]), params[i] => last(axies[params[i]])]
end

"""
Return a map of of variables to the gridpoints at the edge of the domain
"""
@inline function get_edgevals(s, i)
    return [s.x̄[i] => first(s.axies[x̄[i]]), s.x̄[i] => last(s.axies[x̄[i]])]
end

@inline function subvar(depvar, edge_vals)
    return substitute.((depvar,), edge_vals)
end
struct DiscreteSpace{N,M,G}
    vars
    discvars
    time
    x̄ # Note that these aren't necessarily @parameters
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
Base.size(s::DiscreteSpace) = Tuple(length(s.grid[z]) for z in s.x̄)

params(s::DiscreteSpace{N,M}) where {N,M}= s.params

grid_idxs(s::DiscreteSpace) = CartesianIndices(((axes(g)[1] for g in s.grid)...,))
edge_idxs(s::DiscreteSpace{N}) where {N} = reduce(vcat, [[vcat([Colon() for j = 1:i-1], 1, [Colon() for j = i+1:N]), vcat([Colon() for j = 1:i-1], length(s.axies[i]), [Colon() for j = i+1:N])] for i = 1:N])

@inline function axiesvals(s::DiscreteSpace{N,M,G}, x_, I) where {N,M,G} 
    map(enumerate(s.x̄)) do (j, x)
        if isequal(x, x_)
            x => (I[j] == 1 ? first(s.axies[x]) : last(s.axies[x]))
        else
            x => s.grid[x][I[j]]
        end
    end
end

gridvals(s::DiscreteSpace{N}) where N = map(y-> [Pair(x, s.grid[x][y.I[j]]) for (j,x) in enumerate(s.x̄)],s.Igrid)
gridvals(s::DiscreteSpace{N}, I::CartesianIndex) where N = [Pair(x, s.grid[x][I[j]]) for (j,x) in enumerate(s.x̄)]

## Boundary methods ##
edgevals(s::DiscreteSpace{N}) where {N} = reduce(vcat, [get_edgevals(s.x̄, s.axies, i) for i = 1:N])
edgevars(s::DiscreteSpace, I) = [u => s.discvars[u][I] for u in ū]

"""
Generate a map of variables to the gridpoints at the edge of the domain
"""
@inline function edgemaps(s::DiscreteSpace, ::LowerBoundary)
    return [x => first(s.axies[x]) for x in s.x̄]
end
@inline function edgemaps(s::DiscreteSpace, ::UpperBoundary)
    return [x => last(s.axies[x]) for x in s.x̄]
end

varmaps(s::DiscreteSpace, II) = [u => s.discvars[u][II] for u in ū]

valmaps(s::DiscreteSpace, II) = vcat(varmaps(s,II), gridvals(s,II))



Iinterior(s::DiscreteSpace) = s.Igrid[[2:(length(s, x)-1) for x in s.x̄]...]

map_symbolic_to_discrete(II::CartesianIndex, s::DiscreteSpace{N,M}) where {N,M} = vcat([ū[k] => s.discvars[k][II] for k = 1:M], [s.x̄[j] => s.grid[j][II[j]] for j = 1:N])

# ? How rude is this? Makes Iedge work

# TODO: Allow other grids
# TODO: allow depvar-specific center/edge choice?
 
@inline function generate_grid(x̄, axies, domain, discretization::MOLFiniteDifference{G}) where {G<:CenterAlignedGrid}
    return axies    
end

@inline function generate_grid(x̄, axies, domain, discretization::MOLFiniteDifference{G}) where {G<:EdgeAlignedGrid}
    return map(x̄) do x
        xdomain = domain[findfirst(d -> isequal(x, d.variables), domain)]
        dx = discretization.dxs[findfirst(dxs -> isequal(x, dxs[1].val), discretization.dxs)][2]
        dx isa Number ? (x => ((DomainSets.infimum(xdomain.domain)-dx/2):dx:(DomainSets.supremum(xdomain.domain)+dx/2))) : x => dx
    end
end

function DiscreteSpace(domain, depvars, indvars, x̄, discretization::MOLFiniteDifference{G}) where {G}
    t = discretization.time
    nspace = length(x̄)
    # Discretize space
    axies = map(x̄) do x
        xdomain = domain[findfirst(d -> isequal(x, d.variables), domain)]
        dx = discretization.dxs[findfirst(dxs -> isequal(x, dxs[1].val), discretization.dxs)][2]
        dx isa Number ? x => (DomainSets.infimum(xdomain.domain):dx:DomainSets.supremum(xdomain.domain)) : x => dx
    end
    dxs = map(x̄) do x
        x => discretization.dxs[findfirst(dxs -> isequal(x, dxs[1].val), discretization.dxs)][2]
    end

    # Define the grid on which the dependent variables will be evaluated (see #378)
    # center_align is recommended for Dirichlet BCs
    # edge_align is recommended for Neumann BCs (spatial discretization is conservative)

    grid = generate_grid(x̄, axies, domain, discretization)

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
    Iedge = Dict([x => [vec(selectdim(Igrid, dim, 1)), vec(selectdim(Igrid, dim, length(grid[dim].second)))] for (dim, x) in enumerate(x̄)])

    x̄2dim = [x̄[i] => i for i in 1:nspace]
    dim2x̄ = [i => x̄[i] for i in 1:nspace]
    return DiscreteSpace{nspace,length(depvars), G}(depvars, Dict(depvarsdisc), discretization.time, x̄, indvars, Dict(axies), Dict(grid), Dict(dxs), Iaxies, Igrid, Iedge, Dict(x̄2dim))
end



#varmap(s::DiscreteSpace{N,M}) where {N,M} = [ū[i] => i for i = 1:M]

