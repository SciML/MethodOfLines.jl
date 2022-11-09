"""
    DiscreteSpace(domain, depvars, indepvars, discretization::MOLFiniteDifference)

A type that stores informations about the discretized space. It takes each independent variable
defined on the space to be discretized and create a corresponding range. It then takes each dependant
variable and create an array of symbolic variables to represent it in its discretized form.

## Arguments

- `domain`: The domain of the space.
- `vars`: A `VariableMap` object that contains the dependant and independent variables and
    other important values.
- `discretization`: The discretization algorithm.

## Properties

- `ū`: The vector of dependant variables.
- `args`: The dictionary of the operations of dependant variables and the corresponding arguments,
    which include the time variable if given.
- `discvars`: The dictionary of dependant variables and the discrete symbolic representation of them.
    Note that this includes the boundaries. See the example below.
- `time`: The time variable. `nothing` for steady state problems.
- `x̄`: The vector of symbolic spatial variables.
- `axies`: The dictionary of symbolic spatial variables and their numerical discretizations.
- `grid`: Same as `axies` if `CenterAlignedGrid` is used. For `EdgeAlignedGrid`, interpolation will need
    to be defined `±dx/2` above and below the edges of the simulation domain where dx is the step size in the direction of that edge.
- `dxs`: The discretization symbolic spatial variables and their step sizes.
- `Iaxies`: The dictionary of the dependant variables and their `CartesianIndices` of the discretization.
- `Igrid`: Same as `axies` if `CenterAlignedGrid` is used. For `EdgeAlignedGrid`, one more index will be needed for extrapolation.
- `x2i`: The dictionary of symbolic spatial variables their ordering.

## Examples

```julia
julia> using MethodOfLines, DomainSets, ModelingToolkit
julia> using MethodOfLines:DiscreteSpace

julia> @parameters t x
julia> @variables u(..)
julia> Dt = Differential(t)
julia> Dxx = Differential(x)^2

julia> eq  = [Dt(u(t, x)) ~ Dxx(u(t, x))]
julia> bcs = [u(0, x) ~ cos(x),
              u(t, 0) ~ exp(-t),
              u(t, 1) ~ exp(-t) * cos(1)]

julia> domain = [t ∈ Interval(0.0, 1.0),
                 x ∈ Interval(0.0, 1.0)]

julia> dx = 0.1
julia> discretization = MOLFiniteDifference([x => dx], t)
julia> ds = DiscreteSpace(domain, [u(t,x).val], [x.val], discretization)

julia> ds.discvars[u(t,x)]
11-element Vector{Num}:
  u[1](t)
  u[2](t)
  u[3](t)
  u[4](t)
  u[5](t)
  u[6](t)
  u[7](t)
  u[8](t)
  u[9](t)
 u[10](t)
 u[11](t)

julia> ds.axies
Dict{Sym{Real, Base.ImmutableDict{DataType, Any}}, StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}} with 1 entry:
  x => 0.0:0.1:1.0
```
"""
struct DiscreteSpace{N,M,G}
    vars
    discvars
    axies
    grid
    dxs
    Iaxies
    Igrid
end

# * The move to DiscretizedVariable with a smart recursive getindex and custom dict based index type (?) will allow for sampling whole expressions at once, leading to much greater flexibility. Both Sym and Array interfaces will be implemented. Derivatives become the demarcation between different types of sampling => Derivatives are a custom subtype of DiscretizedVariable, with special subtypes for Nonlinear laplacian/spherical/ other types of derivatives with special handling. There is a pre discretized equation step that recognizes and replaces these with rules, and then the resulting equation is simply indexed into to generate the interior/BCs.

function DiscreteSpace(vars, discretization::MOLFiniteDifference{G}) where {G}
    x̄ = vars.x̄
    t = vars.time
    depvars = vars.ū
    nspace = length(x̄)
    # Discretize space
    axies = map(x̄) do x
        xdomain = vars.intervals[x]
        dx = prepare_dx(discretization.dxs[x], xdomain, discretization.grid_align)
        discx = dx isa Number ? (xdomain[1]:dx:xdomain[2]) : dx
        xhigh = xdomain[2]
        if discx[end] != xhigh
            @warn "d$x for $x does not divide domain exactly, adding grid point at $x = $(xhigh))."
            discx = collect(discx)
            push!(discx, xhigh)
        end
        x => discx
    end

    # Define the grid on which the dependent variables will be evaluated (see #378)
    # center_align is recommended for Dirichlet BCs
    # edge_align is recommended for Neumann BCs (spatial discretization is conservative)

    grid = generate_grid(x̄, axies, vars.intervals, discretization)

    dxs = map(x̄) do x
        discx = Dict(grid)[x]
        if discx isa StepRangeLen
            xdomain = vars.intervals[x]
            x => prepare_dx(discretization.dxs[x], xdomain, discretization.grid_align)
        elseif discx isa AbstractVector # is an abstract vector but not StepRangeLen
            x => [discx[i+1] - discx[i] for i in 1:length(x)]
        else
            throw(ArgumentError("Supplied d$x is not a Number or AbstractVector, got $(typeof(discretization.dxs[x])) for $x"))
        end
    end

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
            u => collect(first(@variables $sym(t)[uaxes...]))
        end
    end


    return DiscreteSpace{nspace,length(depvars),G}(vars, Dict(depvarsdisc), axies, grid, Dict(dxs), Dict(Iaxies), Dict(Igrid))
end

import Base.getproperty

function Base.getproperty(s::DiscreteSpace, p::Symbol)
    if p in [:ū, :x̄, :time, :args, :x2i, :i2x]
        getfield(s.vars, p)
    else
        getfield(s, p)
    end
end

prepare_dx(dx::Integer, xdomain, ::CenterAlignedGrid) = (xdomain[2] - xdomain[1])/(dx - 1)
prepare_dx(dx::Integer, xdomain, ::EdgeAlignedGrid) = (xdomain[2] - xdomain[1])/dx
prepare_dx(dx, xdomain, ::AbstractGrid) = dx

nparams(::DiscreteSpace{N,M}) where {N,M} = N
nvars(::DiscreteSpace{N,M}) where {N,M} = M

"""
    params(u, s::DiscreteSpace)

Fillter out the time variable and get the spatial variables of `u` in `s`.
"""
params(u, s::DiscreteSpace) = remove(s.args[operation(u)], s.time)
Base.ndims(u, s::DiscreteSpace) = length(params(u, s))

Base.length(s::DiscreteSpace, x) = length(s.grid[x])
Base.length(s::DiscreteSpace, j::Int) = length(s.grid[s.x̄[j]])
Base.size(s::DiscreteSpace) = Tuple(length(s.grid[z]) for z in s.x̄)

"""
    Idx(II::CartesianIndex, s::DiscreteSpace, u, indexmap)

Here `indexmap` maps the arguments of `u` in `s` to the their ordering. Return a subindex
of `II` that corresponds to only the spatial arguments of `u`.
"""
@inline function Idx(II::CartesianIndex, s::DiscreteSpace, u, indexmap)
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

# TODO: Allow other grids

@inline function generate_grid(x̄, axies, domain, discretization::MOLFiniteDifference{G}) where {G<:CenterAlignedGrid}
    return axies
end

@inline function generate_grid(x̄, axies, intervals, discretization::MOLFiniteDifference{G}) where {G<:EdgeAlignedGrid}
    dict = Dict(axies)
    return map(x̄) do x
        xdomain = intervals[x]
        dx = discretization.dxs[x]
        if dict[x] isa StepRangeLen
            x => (xdomain[1]-dx/2):dx:(xdomain[2]+dx/2)
        else
            discx = [(dict[x][i] + dict[x][i+1]) / 2 for i in 1:length(dict[x])-1]
            pushfirst!(discx, discx[1] - 2 * (discx[1] - xdomain[1]))
            push!(discx, discx[end] + 2 * (xdomain[2] - discx[end]))
            x => discx
        end
    end
end


depvar(u, s::DiscreteSpace) = depvar(u, s.vars)

x2i(s::DiscreteSpace, u, x) = x2i(s.vars, u, x)
