struct DiscreteSpace{N,M,G}
    ū
    args
    discvars
    time
    x̄ # Note that these aren't necessarily @parameters
    axies
    grid
    dxs
    Iaxies
    Igrid
    x2i
end

# * The move to DiscretizedVariable with a smart recursive getindex and custom dict based index type (?) will allow for sampling whole expressions at once, leading to much greater flexibility. Both Sym and Array interfaces will be implemented. Derivatives become the demarcation between different types of sampling => Derivatives are a custom subtype of DiscretizedVariable, with special subtypes for Nonlinear laplacian/spherical/ other types of derivatives with special handling. There is a pre discretized equation step that recognizes and replaces these with rules, and then the resulting equation is simply indexed into to generate the interior/BCs.

function DiscreteSpace(domain, depvars, x̄, discretization::MOLFiniteDifference{G}) where {G}
    t = discretization.time
    nspace = length(x̄)
    # Discretize space
    dxs = Dict(map(x̄) do x
        xdomain = domain[findfirst(d -> isequal(x, d.variables), domain)]
        dx = discretization.dxs[findfirst(dxs -> isequal(x, dxs[1].val), discretization.dxs)][2]
        xmin = DomainSets.infimum(xdomain.domain)
        xmax = DomainSets.supremum(xdomain.domain)
        dx = dx isa Number ? dx : dx[2] - dx[1]
        N = (xmax - xmin) / dx
        if !isinteger(N)
            N = ceil(N)
            dx = (xmax - xmin) / N
            @warn "d$x does not give integer point count for that dimension, reducing to $dx"
        end
        x => dx
    end)
    axies = map(x̄) do x
        xdomain = domain[findfirst(d -> isequal(x, d.variables), domain)]
        xmin = DomainSets.infimum(xdomain.domain)
        xmax = DomainSets.supremum(xdomain.domain)
        x => xmin:dxs[x]:xmax
    end

    # Define the grid on which the dependent variables will be evaluated (see #378)
    # center_align is recommended for Dirichlet BCs
    # edge_align is recommended for Neumann BCs (spatial discretization is conservative)

    grid = generate_grid(x̄, dxs, axies, domain, discretization)

    axies = Dict(axies)
    grid = Dict(grid)

    # Build symbolic variables
    Iaxies = [u => CartesianIndices(((axes(axies[x])[1] for x in remove(arguments(u), t))...,)) for u in depvars]
    Igrid = [u => CartesianIndices(((axes(grid[x])[1] for x in remove(arguments(u), t))...,)) for u in depvars]

    depvarsdisc = map(depvars) do u
        op = SymbolicUtils.operation(u)
        if op isa SymbolicUtils.Term{SymbolicUtils.FnType{Tuple,Real},Nothing}
            sym = Symbol(string(op))
        else
            sym = nameof(op)
        end
        if t === nothing
            uaxes = collect(axes(grid[x])[1] for x in arguments(u))
            u => collect(first(@variables $sym[uaxes...]))
        elseif isequal(SymbolicUtils.arguments(u), [t])
            u => fill(first(@variables($sym(t))), ()) #Create a 0-dimensional array
        else
            uaxes = collect(axes(grid[x])[1] for x in remove(arguments(u), t))
            u => collect(first(@variables $sym[uaxes...](t)))
        end
    end

    args = [operation(u) => arguments(u) for u in depvars]

    x̄2dim = [x̄[i] => i for i in 1:nspace]
    dim2x̄ = [i => x̄[i] for i in 1:nspace]
    return DiscreteSpace{nspace,length(depvars),G}(depvars, Dict(args), Dict(depvarsdisc), discretization.time, x̄, axies, grid, dxs, Dict(Iaxies), Dict(Igrid), Dict(x̄2dim))
end

nparams(::DiscreteSpace{N,M}) where {N,M} = N
nvars(::DiscreteSpace{N,M}) where {N,M} = M

params(u,s) = remove(s.args[operation(u)], s.time)
Base.ndims(u,s::DiscreteSpace) = length(params(u,s))

Base.length(s::DiscreteSpace, x) = length(s.grid[x])
Base.length(s::DiscreteSpace, j::Int) = length(s.grid[s.x̄[j]])
Base.size(s::DiscreteSpace) = Tuple(length(s.grid[z]) for z in s.x̄)

@inline function Idx(II, s, u, indexmap)
    # We need to construct a new index as indices may be of different size
    length(params(u,s)) == 0 && return CartesianIndex()
    is = [II[indexmap[x]] for x in params(u, s)]


    II = CartesianIndex(is...)
    return II
end

"""
A function that returns what to replace independent variables with in boundary equations
"""
@inline function axiesvals(s::DiscreteSpace{N,M,G}, u_, x_, I) where {N,M,G}
    u = depvar(u_,s)
    map(params(u, s)) do x
        if isequal(x, x_)
            x => (I[x2i(s, u, x)] == 1 ? first(s.axies[x]) : last(s.axies[x]))
        else
            x => s.grid[x][I[x2i(s, u, x)]]
        end
    end
end

gridvals(s::DiscreteSpace{N}, u) where N = ndims(u,s) == 0 ? [] : map(y-> [x => s.grid[x][y.I[x2i(s, u, x)]] for x in params(u,s)],s.Igrid[u])
gridvals(s::DiscreteSpace{N}, u, I::CartesianIndex) where N = ndims(u,s) == 0 ? [] : [x => s.grid[x][I[x2i(s,u,x)]] for x in params(u, s)]


varmaps(s::DiscreteSpace, depvars, II, indexmap) = [u => s.discvars[u][Idx(II, s, u, indexmap)] for u in depvars]

valmaps(s::DiscreteSpace, u, depvars, II, indexmap) = length(II) == 0 ? [] : vcat(varmaps(s, depvars, II, indexmap), gridvals(s, u, II))

valmaps(s, u, depvars, indexmap) = valmaps.([s], [u], [depvars], s.Igrid[u], [indexmap])

map_symbolic_to_discrete(II::CartesianIndex, s::DiscreteSpace{N,M}) where {N,M} = vcat([s.ū[k] => s.discvars[k][II] for k = 1:M], [s.x̄[j] => s.grid[j][II[j]] for j = 1:N])

# TODO: Allow other grids

@inline function generate_grid(x̄, dxs, axies, domain, discretization::MOLFiniteDifference{G}) where {G<:CenterAlignedGrid}
    return axies
end

@inline function generate_grid(x̄, dxs, axies, domain, discretization::MOLFiniteDifference{G}) where {G<:EdgeAlignedGrid}
    return map(x̄) do x
        xdomain = domain[findfirst(d -> isequal(x, d.variables), domain)]
        dx = dxs[x]
        dx isa Number ? (x => ((DomainSets.infimum(xdomain.domain)-dx/2):dx:(DomainSets.supremum(xdomain.domain)+dx/2))) : x => dx
    end
end


depvar(u, s::DiscreteSpace) = operation(u)(s.args[operation(u)]...)

x2i(s::DiscreteSpace, u, x) = findfirst(isequal(x), remove(s.args[operation(u)], s.time))
