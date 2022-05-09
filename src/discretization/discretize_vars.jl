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

# TODO: Allow other grids

@inline function generate_axies(superdomains, domain, discretization)
    subaxies = []
    axies = map(superdomains) do x
        xdomain = domain[findfirst(d -> isequal(x, d.variables), domain)]
        dx = discretization.dxs[findfirst(dxs -> isequal(x, dxs[1].val), discretization.dxs)][2]
        discx = dx isa Number ? (DomainSets.infimum(xdomain.domain):dx:DomainSets.supremum(xdomain.domain)) : dx
        xhigh = DomainSets.supremum(xdomain.domain)
        if discx[end] != xhigh
            @warn "d$x for $x does not divide domain exactly, adding grid point at $x = $(xhigh))."
            discx = collect(discx)
            push!(discx, xhigh)
        end
        # Handle overlaps
        try
            xsub = discretization.overlap_map[x]
            if length(xsub) > 0
                highs = Dict(map(xsub) do x
                    x => DomainSets.supremum(domain[findfirst(d -> isequal(x, d.variables), domain)].domain)
                end)
                lows = Dict(map(xsub) do x
                    x => DomainSets.infimum(domain[findfirst(d -> isequal(x, d.variables), domain)].domain)
                end)
                for high in values(highs)
                    if !(high ∈ discx)
                        discx = vcat(discx[discx.<high], high, discx[discx.>high])
                        @warn "Due to overlap, adding grid point at $x = $(high))."
                    end
                end
                for low in values(lows)
                    if !(low ∈ discx)
                        discx = vcat(discx[discx.<low], low, discx[discx.>low])
                        @warn "Due to overlap, adding grid point at $x = $(low))."
                    end
                end
            end
            vcat!(subaxies, map(y -> y.val => discx[lows[y].<=discx.<=highs[y]], xsub))
        catch e
            # Catch missing keys for no overlap
        end

        x => discx
    end
    return Dict(vcat(subaxies, axies))
end

@inline function generate_grid(superdomains, axies, domain, discretization::MOLFiniteDifference{G}) where {G<:CenterAlignedGrid}
    imap = []
    for x in superdomains
        try
            xsub = discretization.overlap_map[x]
            for y in xsub
                high = DomainSets.supremum(domain[findfirst(d -> isequal(x, d.variables), domain)].domain)
                low = DomainSets.infimum(domain[findfirst(d -> isequal(x, d.variables), domain)].domain)
                ilow = findlast(l -> l < low, axies[x])
                ihigh = findlast(l -> l > high, discx)
                push!(imap, y => (ilow, ihigh))
            end
        catch e
        end
        push!(imap, x => (1, length(discx)))
    end

    return axies, Dict(imap)
end


@inline function generate_grid(superdomains, axies, domain, discretization::MOLFiniteDifference{G}) where {G<:EdgeAlignedGrid}
    subgrid = []
    imap = []
    grid = map(superdomains) do x
        xdomain = domain[findfirst(d -> isequal(x, d.variables), domain)]
        dx = discretization.dxs[findfirst(dxs -> isequal(x, dxs[1].val), discretization.dxs)][2]
        if dict[x] isa StepRangeLen
            discx = (DomainSets.infimum(xdomain.domain)-dx/2):dx:(DomainSets.supremum(xdomain.domain)+dx/2)
        else
            discx = [(axies[x][i] + axies[x][i+1]) / 2 for i in 1:length(axies[x])-1]
            pushfirst!(discx, discx[1] - 2 * (discx[1] - infimum(xdomain.domain)))
            push!(discx, discx[end] + 2 * (supremum(xdomain.domain) - discx[end]))
        end
        # Handle overlap
        try
            xsub = discretization.overlap_map[x]
            for y in xsub
                high = DomainSets.supremum(domain[findfirst(d -> isequal(x, d.variables), domain)].domain)
                low = DomainSets.infimum(domain[findfirst(d -> isequal(x, d.variables), domain)].domain)
                ilow = findlast(l -> l < low, discx)
                ihigh = findlast(l -> l > high, discx)
                push!(imap, y => (ilow, ihigh))
                push!(subgrid, y.val => discx[ilow:ihigh])
            end
        catch e
        end
        push!(imap, x => (1, length(discx)))
        x => discx
    end
    return (Dict(vcat(subgrid, grid)), Dict(imap))
end

@inline function generate_discrete_depvars(depvars, grid, domain, discretization)
    sub2sup = Dict(mapreduce(vcat, keys(discretization.overlap_map)) do x
        #? needs .val?
        [y => x for y in overlap_map[x]]
    end)
    superdepvars = filter(v -> isequal(v, substitute(v, sub2sup)), depvars)
    subdepvars = setdiff(depvars, superdepvars)

    sub2sup = Dict(map(depvars) do u
        if !isequal(u, substitute(u, sub2sup))
            u => substitute(u, sub2sup)
        else
            nothing => nothing
        end
    end)

    depvarsdisc = Dict(map(superdepvars) do u
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
    end)

    subdepvarsdisc = Dict(map(subdepvars) do u
        args = arguments(u)
        I = CartesianIndices(([imap[x][1]:imap[x][2] for x in args]...))
        u => @view(depvarsdisc[sub2sup[u]][I])
    end)

    return merge(depvarsdisc, subdepvarsdisc)
end

function DiscreteSpace(domain, depvars, x̄, discretization::MOLFiniteDifference{G}) where {G}
    t = discretization.time
    nspace = length(x̄)
    # Discretize space

    # Needed to allow variables in seperate but overlapping domains
    subdomains = filter(y -> any(isequal((y,), unique(reduce(vcat, [discretization.overlap_map[x] for x in keys(discretization.overlap_map)])))), x̄)
    superdomains = filter(y -> !any(isequal.(subdomains, (y,))), x̄)

    # Define the grid on which the dependent variables will be evaluated (see #378)
    # center_align is recommended for Dirichlet BCs
    # edge_align is recommended for Neumann BCs (spatial discretization is conservative)

    axies = generate_axies(superdomains, domain, discretization)
    grid = generate_grid(superdomains, axies, domain, discretization)

    dxs = map(x̄) do x
        discx = grid[x]
        if discx isa StepRangeLen
            x => discretization.dxs[findfirst(dxs -> isequal(x, dxs[1].val), discretization.dxs)][2]
        elseif discx isa AbstractVector # is an abstract vector but not StepRangeLen
            x => [discx[i+1] - discx[i] for i in 1:length(x)]
        else
            throw(ArgumentError("Supplied d$x is not a Number or AbstractVector, got $(typeof(discretization.dxs[findfirst(dxs -> isequal(x, dxs[1].val), discretization.dxs)][2])) for $x"))
        end
    end

    # Build symbolic variables
    Iaxies = [u => CartesianIndices(((axes(axies[x])[1] for x in remove(arguments(u), t))...,)) for u in depvars]
    Igrid = [u => CartesianIndices(((axes(grid[x])[1] for x in remove(arguments(u), t))...,)) for u in depvars]



    args = [operation(u) => arguments(u) for u in depvars]

    x̄2dim = [x̄[i] => i for i in 1:nspace]
    dim2x̄ = [i => x̄[i] for i in 1:nspace]
    return DiscreteSpace{nspace,length(depvars),G}(depvars, Dict(args), Dict(depvarsdisc), discretization.time, x̄, axies, grid, Dict(dxs), Dict(Iaxies), Dict(Igrid), Dict(x̄2dim))
end

nparams(::DiscreteSpace{N,M}) where {N,M} = N
nvars(::DiscreteSpace{N,M}) where {N,M} = M

params(u, s) = remove(s.args[operation(u)], s.time)
Base.ndims(u, s::DiscreteSpace) = length(params(u, s))

Base.length(s::DiscreteSpace, x) = length(s.grid[x])
Base.length(s::DiscreteSpace, j::Int) = length(s.grid[s.x̄[j]])
Base.size(s::DiscreteSpace) = Tuple(length(s.grid[z]) for z in s.x̄)

@inline function Idx(II, s, u, indexmap)
    # We need to construct a new index as indices may be of different size
    length(params(u, s)) == 0 && return CartesianIndex()
    is = [II[indexmap[x]] for x in params(u, s)]


    II = CartesianIndex(is...)
    return II
end

"""
A function that returns what to replace independent variables with in boundary equations
"""
@inline function axiesvals(s::DiscreteSpace{N,M,G}, u_, x_, I) where {N,M,G}
    u = depvar(u_, s)
    map(params(u, s)) do x
        if isequal(x, x_)
            x => (I[x2i(s, u, x)] == 1 ? first(s.axies[x]) : last(s.axies[x]))
        else
            x => s.grid[x][I[x2i(s, u, x)]]
        end
    end
end

gridvals(s::DiscreteSpace{N}, u) where {N} = ndims(u, s) == 0 ? [] : map(y -> [x => s.grid[x][y.I[x2i(s, u, x)]] for x in params(u, s)], s.Igrid[u])
gridvals(s::DiscreteSpace{N}, u, I::CartesianIndex) where {N} = ndims(u, s) == 0 ? [] : [x => s.grid[x][I[x2i(s, u, x)]] for x in params(u, s)]


varmaps(s::DiscreteSpace, depvars, II, indexmap) = [u => s.discvars[u][Idx(II, s, u, indexmap)] for u in depvars]

valmaps(s::DiscreteSpace, u, depvars, II, indexmap) = length(II) == 0 ? [] : vcat(varmaps(s, depvars, II, indexmap), gridvals(s, u, II))

valmaps(s, u, depvars, indexmap) = valmaps.([s], [u], [depvars], s.Igrid[u], [indexmap])

map_symbolic_to_discrete(II::CartesianIndex, s::DiscreteSpace{N,M}) where {N,M} = vcat([s.ū[k] => s.discvars[k][II] for k = 1:M], [s.x̄[j] => s.grid[j][II[j]] for j = 1:N])



depvar(u, s::DiscreteSpace) = operation(u)(s.args[operation(u)]...)

x2i(s::DiscreteSpace, u, x) = findfirst(isequal(x), remove(s.args[operation(u)], s.time))
