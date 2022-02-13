struct DiscreteSpace{N,M,G}
    ū
    discvars
    time
    x̄ # Note that these aren't necessarily @parameters
    axies
    grid
    dxs
    Iaxies
    Igrid
    Iedges
    x2i
end

nparams(::DiscreteSpace{N,M}) where {N,M} = N
nvars(::DiscreteSpace{N,M}) where {N,M} = M

Base.length(s::DiscreteSpace, x) = length(s.grid[x])
Base.length(s::DiscreteSpace, j::Int) = length(s.grid[s.x̄[j]])
Base.size(s::DiscreteSpace) = Tuple(length(s.grid[z]) for z in s.x̄)

"""
A function that returns wat to replace independent variables with in boundary equations
"""
@inline function axiesvals(s::DiscreteSpace{N,M,G}, x_, I) where {N,M,G} 
    map(enumerate(s.x̄)) do (j, x)
        if isequal(x, x_)
            x => (I[j] == 1 ? first(s.axies[x]) : last(s.axies[x]))
        else
            x => s.grid[x][I[j]]
        end
    end
end

gridvals(s::DiscreteSpace{N}) where N = map(y-> [x => s.grid[x][y.I[j]] for (j,x) in enumerate(s.x̄)],s.Igrid)
gridvals(s::DiscreteSpace{N}, I::CartesianIndex) where N = [x => s.grid[x][I[j]] for (j,x) in enumerate(s.x̄)]


varmaps(s::DiscreteSpace, II) = [u => s.discvars[u][II] for u in s.ū]

valmaps(s::DiscreteSpace, II) = vcat(varmaps(s,II), gridvals(s,II))


interior(Igrid, N) = Igrid[[2:(size(Igrid, i)-1) for i in 1:N]...]


map_symbolic_to_discrete(II::CartesianIndex, s::DiscreteSpace{N,M}) where {N,M} = vcat([s.ū[k] => s.discvars[k][II] for k = 1:M], [s.x̄[j] => s.grid[j][II[j]] for j = 1:N])

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

function DiscreteSpace(domain, depvars, x̄, discretization::MOLFiniteDifference{G}) where {G}
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
    Iaxies = [u => CartesianIndices(((axes(axies[x])[1] for x in remove(arguments(u), t))...,)) for u in depvars]
    Igrid = [u => CartesianIndices(((axes(grid[x])[1] for x in remove(arguments(u), t))...,)) for u in depvars]

    depvarsdisc = map(depvars) do u
        if t === nothing
            sym = nameof(SymbolicUtils.operation(u))
            uaxes = collect(axes(grid[x])[1] for x in arguments(u))
            u => collect(first(@variables $sym[uaxes...]))
        elseif isequal(SymbolicUtils.arguments(u), [t])
            u => [u for II in s.Igrid[u]]
        else
            sym = nameof(SymbolicUtils.operation(u))
            uaxes = collect(axes(grid[x])[1] for x in remove(arguments(u), t))
            u => collect(first(@variables $sym[collect(axes(g.second)[1] for g in grid)...](t))) 
        end
    end

    x̄2dim = [x̄[i] => i for i in 1:nspace]
    dim2x̄ = [i => x̄[i] for i in 1:nspace]
    return DiscreteSpace{nspace,length(depvars), G}(depvars, Dict(depvarsdisc), discretization.time, x̄, Dict(axies), Dict(grid), Dict(dxs), Iaxies, Igrid, Dict(x̄2dim))
end

@inline function edges(x̄, Igrid, N)
    sd(A::AbstractArray{T,N}, d, i) where {T,N} = selectdim(interior(A, N), d, i)
    return Dict(map(enumerate(x̄)) do (dim, x)
        x =>[sd(Igrid, dim, 1)              .- [unitindex(N, dim)], 
             sd(Igrid, dim, size(Igrid, dim)-2) .+ [unitindex(N, dim)]]
    end)
end

@inline function edge(s::Discretespace, u, x, islower)
    sd(A::AbstractArray{T,N}, x, i) where {T,N} = selectdim(interior(A, N), s.x2i[x], i)
    if islower
        return sd(s.Igrid[u], x, 1)
    else
        return sd(s.Igrid[u], x, length(s, x))
    end
end 

edge(s::Discretespace, b::AbstractBoundary) = edge(s, b.u, b.x, !isupper(b))
# """
# Create a vector containing indices of the corners of the domain.
# """
# @inline function removecorners(s::DiscreteSpace{2,M}) where {M}
    
#     Icorners = map(0:3) do n
#         dig = digits(n, base = 2, pad = 2)
#         CartesianIndex(b == 1 ? length(s, i) : 1 for (i, b) in enumerate(dig))
#     end
#     Ivalid = setdiff(s.Igrid, Icorners)
#     return mapreduce(vcat, u-> vec(s.discvars[u][Ivalid]), s.ū)
# end


#varmap(s::DiscreteSpace{N,M}) where {N,M} = [s.ū[i] => i for i = 1:M]
# ! needs rewrite
@inline function cornereqs(s::DiscreteSpace{N,M}) where {N,M}
    Iedges = edges(s)
    # Flatten Iedges
    Iedges = mapreduce(x->mapreduce(i -> vec(Iedges[x][i]), vcat, 1:2), vcat, s.x̄)
    Iinterior = interior(s.Igrid, nparams(s))
    Ivalid = setdiff(s.Igrid, vcat(vec(Iinterior), vec(Iedges)))
    #TODO: change this when vars have different domains
    return mapreduce(u -> vec(s.discvars[u][Ivalid]) .~ 0., vcat, s.ū)::Vector
end
