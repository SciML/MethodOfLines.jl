### INITIAL AND BOUNDARY CONDITIONS ###

#TODO: Retire DiscreteSpace and move to a DiscreteEquation class with corresponding DiscreteVariables which carry all the required information, define substitute methods for these.

abstract type AbstractBoundary end

abstract type AbstractTruncatingBoundary <: AbstractBoundary end

abstract type AbstractInterfaceBoundary <: AbstractTruncatingBoundary end

abstract type AbstractLowerBoundary <: AbstractTruncatingBoundary end

abstract type AbstractUpperBoundary <: AbstractTruncatingBoundary end

abstract type AbstractExtendingBoundary <: AbstractBoundary end

struct LowerBoundaryTrait end

struct UpperBoundaryTrait end

trait(b::AbstractBoundary) = isupper(b) ? UpperBoundaryTrait() : LowerBoundaryTrait()

struct LowerBoundary <: AbstractLowerBoundary
    u
    x
    depvars
    indvars
    eq
    order
    function LowerBoundary(u, t, x, order, eq, v)
        depvars_lhs = get_depvars(eq.lhs, v.depvar_ops)
        depvars_rhs = get_depvars(eq.rhs, v.depvar_ops)
        depvars = collect(depvars_lhs ∪ depvars_rhs)
        #depvars =  filter(u -> !any(map(x-> x isa Number, arguments(u))), depvars)
        allx̄ = Set(filter(!isempty, map(u -> filter(x -> t === nothing || !isequal(x, t), arguments(u)), depvars)))
        return new(u, x, depvar.(depvars, [v]), first(allx̄), eq, order)
    end
end

struct UpperBoundary <: AbstractUpperBoundary
    u
    x
    depvars
    indvars
    eq
    order
    function UpperBoundary(u, t, x, order, eq, v)
        depvars_lhs = get_depvars(eq.lhs, v.depvar_ops)
        depvars_rhs = get_depvars(eq.rhs, v.depvar_ops)
        depvars = collect(depvars_lhs ∪ depvars_rhs)
        allx̄ = Set(filter(!isempty, map(u -> filter(x -> t === nothing || !isequal(x, t), arguments(u)), depvars)))
        return new(u, x, depvar.(depvars, [v]), first(allx̄), eq, order)
    end
end


struct LowerInterpolatingBoundary <: AbstractLowerBoundary
    u
    x
end

struct UpperInterpolatingBoundary <: AbstractUpperBoundary
    u
    x
end



# Note that it is assumed throughout MOL that the variables in an inteface BC have the same argument signature,
# differing in only one variable which is that of the interface. This is not checked here, but will cause errors if it is not true.
# Interfaces are assumed to be on the lower boundary of the domain.

struct InterfaceBoundary{IsUpper_u,IsUpper_u2} <: AbstractInterfaceBoundary
    u
    u2
    x
    x2
    eq
end

struct HigherOrderInterfaceBoundary <: AbstractInterfaceBoundary
    u
    u2
    x
    x2
    depvars
    indvars
    eq
    order
    function HigherOrderInterfaceBoundary(u, u2, x, x2, t, order, eq, v)
        depvars_lhs = get_depvars(eq.lhs, v.depvar_ops)
        depvars_rhs = get_depvars(eq.rhs, v.depvar_ops)
        depvars = collect(depvars_lhs ∪ depvars_rhs)

        allx̄ = Set(filter(!isempty, map(u -> filter(x -> t === nothing || !isequal(x, t), arguments(u)), depvars)))
        return new(u, u2, x, x2, depvar.(depvars, [v]), first(allx̄), eq, order)
    end
end
const AbstractEquationBoundary = Union{LowerBoundary, UpperBoundary, HigherOrderInterfaceBoundary}
const AbstractInterpolatingBoundary = Union{LowerInterpolatingBoundary, UpperInterpolatingBoundary}

no_interp(bs) = filter(b -> !(b isa AbstractInterpolatingBoundary), bs)

function Base.isequal(i1::InterfaceBoundary, i2::InterfaceBoundary)
    front = (isequal(i1.u, i2.u) & isequal(i1.u2, i2.u2) & isequal(i1.x, i2.x) & isequal(i1.x2, i2.x2))
    back = (isequal(i1.u, i2.u2) & isequal(i1.u2, i2.u) & isequal(i1.x, i2.x2) & isequal(i1.x2, i2.x))
    return front | back
end

getvars(b::AbstractBoundary) = (b.u, b.x)

function isperiodic(b1::InterfaceBoundary{b1u,b1u2}, b2::InterfaceBoundary{b2u,b2u2}) where {b1u,b1u2,b2u,b2u2}
    us_equal = isequal(operation(b1.u), operation(b2.u2)) && isequal(operation(b2.u), operation(b1.u2))
    xs_equal = issequal(b1.x, b2.x2) && isequal(b1.x2, b2.x)
    return us_equal && xs_equal
end

@inline function isinterface(b)
    if b isa InterfaceBoundary
        return Val(true)
    else
        return Val(false)
    end
end

filter_interfaces(bs) = filter(b -> b isa InterfaceBoundary, bs)

function haslowerupper(bs, x)
    haslower = false
    hasupper = false
    for b in filter_interfaces(bs)
        if isequal(b.x, x)
            if isupper(b)
                hasupper = true
            else
                haslower = true
            end
        end
    end
    return haslower, hasupper
end

has_interfaces(bmps) = any(b -> b isa InterfaceBoundary, reduce(vcat, reduce(vcat, collect.(values.(collect(values(bmps)))))))


idx(b::LowerBoundary, s) = 1
idx(b::UpperBoundary, s) = length(s, b.x)
idx(b::HigherOrderInterfaceBoundary, s) = length(s, b.x)

# indexes for Iedge depending on boundary type
isupper(::AbstractLowerBoundary) = false
isupper(::AbstractUpperBoundary) = true
isupper(::InterfaceBoundary{IsUpper_u}) where {IsUpper_u} = IsUpper_u isa Val{true} ? true : false
isupper(::HigherOrderInterfaceBoundary) = true

flatten_vardict(bmps) = reduce(vcat, reduce(vcat, collect.(values.(collect(values(bmps))))))

@inline function clip_interior!!(lower, upper, s, b::AbstractBoundary)
    # This x2i is correct
    dim = x2i(s, depvar(b.u, s), b.x)
    @assert dim !== nothing "Internal Error: Variable $(b.x) not found in $(depvar(b.u, s)), when parsing boundary condition $(b)"
    if b isa InterfaceBoundary && isupper(b)
        return
    end
    lower[dim] = lower[dim] + !isupper(b)
    upper[dim] = upper[dim] + isupper(b)
end

ordering(::LowerBoundary) = 1
ordering(::UpperBoundary) = 1
ordering(::LowerInterpolatingBoundary) = 2
ordering(::UpperInterpolatingBoundary) = 2

offset(::AbstractLowerBoundary, i, len) = i
offset(::AbstractUpperBoundary, i, len) = len - i + 1


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

function _boundary_rules(v, orders, u, x, val)
    args = v.args[operation(u)]
    args = substitute.(args, (x => val,))
    varrule = operation(u)(args...) => [operation(u)(args...), x, 0]

    spacerules = [(Differential(x)^d)(operation(u)(args...)) => [operation(u)(args...), x, d] for d in reverse(orders[x])]

    if v.time !== nothing && x !== v.time
        timerules = [Differential(v.time)(operation(u)(args...)) => [operation(u)(args...), x, d] for d in reverse(orders[v.time])]
        return vcat(spacerules, timerules, varrule)
    else
        return vcat(spacerules, varrule)
    end
end

function generate_boundary_matching_rules(v, orders)
    # TODO: Check for bc equations of multiple variables
    lowerboundary(x) = v.intervals[x][1]
    upperboundary(x) = v.intervals[x][2]

    # Rules to match boundary conditions on the lower boundaries
    lower = Dict([operation(u) => Dict([x => _boundary_rules(v, orders, u, x, lowerboundary(x)) for x in all_params(u, v)]) for u in v.ū])

    upper = Dict([operation(u) => Dict([x => _boundary_rules(v, orders, u, x, upperboundary(x)) for x in all_params(u, v)]) for u in v.ū])

    return (lower, upper)
end

"""
Creates a map of boundaries for each variable to be used later when discretizing the boundary condition equations
"""
function parse_bcs(bcs, v::VariableMap, orders)
    t = v.time

    ## BC matching rules, returns the variable and parameter the bc concerns

    lower_boundary_rules, upper_boundary_rules = generate_boundary_matching_rules(v, orders)

    boundarymap = Dict([operation(u) => Dict([x => [] for x in all_ivs(v)]) for u in v.ū])

    # Generate initial conditions and bc equations
    for bc in bcs
        boundary = nothing
        # Split out additive terms
        terms = split_terms(bc)
        # * Assume that the BC is defined on the edge of the domain
        # Check whether the bc is on the lower boundary, or periodic, we don't care which depvar/var
        local u_, u__, x_, x__
        isinterface = false
        interface_orders = []
        for term in terms, r in flatten_vardict(lower_boundary_rules)
            # Need to check the order, higher order should be discretized as a normal upper index
            # As zeroth order goes on the lower index. Needs a zeroth order interface to make sense,
            # but I assume that this will always be the case where there are higher order interfaces.
            #Check if the rule changes the expression
            if subsmatch(term, r)
                # Get the matched variables from the rule
                u_, x_, order = r.second
                # Mark the boundary
                if boundary === nothing
                    boundary = LowerBoundary(u_, t, x_, order, bc, v)
                end
                # do it again for the upper end to check for periodic, but only check the current depvar and indvar
                push!(interface_orders, order)
                for term_ in setdiff(terms, [term]), r_ in flatten_vardict(upper_boundary_rules)
                    if subsmatch(term_, r_)
                        isinterface = true
                        u__, x__, order__ = r_.second
                        @assert ndims(u_, v) == ndims(u__, v) "Invalid Interface Boundary $bc: Dependent variables $(u_) and $(u__) have different numbers of dimensions."

                        push!(interface_orders, order__)
                    end
                end
                # Handle flux condition at interface
            end
        end
        # If this is a condition on flux at an interface, leave it to become an upper boundary
        # Else if it is a simple interface, make interface boundaries
        if isinterface
            if all(==(0), interface_orders)
                boundary = (InterfaceBoundary{Val(false),Val(true)}(u_, u__, x_, x__, bc),
                    InterfaceBoundary{Val(true),Val(false)}(u__, u_, x__, x_, bc))
            else
                boundary = HigherOrderInterfaceBoundary(u__, u_, x__, x_, t, maximum(interface_orders), bc, v)
            end
        end
        # repeat for upper boundary
        if boundary === nothing
            for term in terms, r in flatten_vardict(upper_boundary_rules)
                if subsmatch(term, r)
                    u_, x_, order = r.second
                    boundary = UpperBoundary(u_, t, x_, order, bc, v)
                    break
                end
            end
        end
        @assert boundary !== nothing "Boundary condition $bc is not on a boundary of the domain, or is not a valid boundary condition"
        if boundary isa Tuple
            push!(boundarymap[operation(boundary[1].u)][boundary[1].x], boundary[1])
            push!(boundarymap[operation(boundary[2].u)][boundary[2].x], boundary[2])
        else
            push!(boundarymap[operation(boundary.u)][boundary.x], boundary)
        end
    end
    return boundarymap
end
