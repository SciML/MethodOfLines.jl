### INITIAL AND BOUNDARY CONDITIONS ###

#TODO: Retire DiscreteSpace and move to a DiscreteEquation class with corresponding DiscreteVariables which carry all the required information, define substitute methods for these.

abstract type AbstractBoundary end

abstract type AbstractTruncatingBoundary <: AbstractBoundary end

abstract type AbstractExtendingBoundary <: AbstractBoundary end

struct LowerBoundary <: AbstractTruncatingBoundary
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

        allx̄ = Set(filter(!isempty, map(u->filter(x-> t === nothing || !isequal(x, t), arguments(u)), depvars)))
        return new(u, x, depvar.(depvars, [v]), first(allx̄), eq, order)
    end
end

struct UpperBoundary <: AbstractTruncatingBoundary
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

        allx̄ = Set(filter(!isempty, map(u->filter(x-> t === nothing || !isequal(x, t), arguments(u)), depvars)))
        return new(u, x, depvar.(depvars, [v]), first(allx̄), eq, order)
    end
end

struct PeriodicBoundary <: AbstractBoundary
    u
    x
    eq
end

getvars(b::AbstractBoundary) = (b.u, b.x)

@inline function isperiodic(bmps, u, x)
    if !isempty(bmps[operation(u)][x])
        # Being explicit hoping the compiler will optimize this away
        if first(bmps[operation(u)][x]) isa PeriodicBoundary
            return Val(true)
        else
            return Val(false)
        end
    else
        return Val(false)
    end
end

@inline function isperiodic(b)
    if b isa PeriodicBoundary
        return Val(true)
    else
        return Val(false)
    end
end

@inline function clip_interior!!(lower, upper, s, b::AbstractBoundary)
    # This x2i is correct
    dim = x2i(s, depvar(b.u, s), b.x)
    @assert dim !== nothing "Internal Error: Variable $(b.x) not found in $(depvar(b.u, s)), when parsing boundary condition $(b)"

    lower[dim] = lower[dim] + !isupper(b)
    upper[dim] = upper[dim] + isupper(b)
end

idx(b::LowerBoundary, s) = 1
idx(b::UpperBoundary, s) = length(s, b.x)


# indexes for Iedge depending on boundary type
isupper(::LowerBoundary) = false
isupper(::UpperBoundary) = true
isupper(::PeriodicBoundary) = false

@inline function edge(interiormap, s, u, j, islower)
    I = interiormap.I[interiormap.pde[depvar(u, s)]]
    # check needed on v1.6
    length(I) == 0 && return CartesianIndex{0}[]
    sd(i) = selectdim(I, j, i)
    I1 = unitindex(ndims(u, s), j)
    if islower
        edge = sd(1)
        # cast the edge of the interior to the edge of the boundary
        edge = edge .- [I1*(edge[1][j]-1)]
    else
        edge = sd(size(interiormap.I[interiormap.pde[depvar(u,s)]], j))
        edge = edge .+ [I1*(size(s.discvars[depvar(u, s)], j)-edge[1][j])]
    end
    return edge
end

edge(s, b, interiormap) = edge(interiormap, s, b.u, x2i(s, b.u, b.x), !isupper(b))

function _boundary_rules(v, orders, u, x, val)
        args = v.args[operation(u)]
        args = substitute.(args, (x=>val,))
        varrule = operation(u)(args...) => [operation(u)(args...), x, 0]

        spacerules = [(Differential(x)^d)(operation(u)(args...)) => [operation(u)(args...), x, d] for d in reverse(orders[x])]

        if v.time !== nothing && x !== v.time
            timerules = [Differential(v.time)(operation(u)(args...)) => [operation(u)(args...), v.time, d] for d in reverse(orders[v.time])]
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
    depvar_ops = v.depvar_ops

    # Create some rules to match which bundary/variable a bc concerns
    # * Assume that the term of the condition is applied additively and has no multiplier/divisor/power etc.
    u0 = []
    bceqs = []
    ## BC matching rules, returns the variable and parameter the bc concerns

    lower_boundary_rules, upper_boundary_rules = generate_boundary_matching_rules(v, orders)

    boundarymap = Dict([operation(u)=>Dict([x => [] for x in all_ivs(v)]) for u in v.ū])

    # Generate initial conditions and bc equations
    for bc in bcs
        boundary = nothing
        # Split out additive terms
        terms = split_terms(bc)
        # * Assume that the BC is defined on the edge of the domain
        # Check whether the bc is on the lower boundary, or periodic, we don't care which depvar/var
        for term in terms, r in reduce(vcat, reduce(vcat, collect.(values.(collect(values(lower_boundary_rules))))))
            #Check if the rule changes the expression
            if subsmatch(term, r)
                # Get the matched variables from the rule
                u_, x_, order = r.second
                # Mark the boundary
                boundary = LowerBoundary(u_, t, x_, order, bc, v)
                # do it again for the upper end to check for periodic, but only check the current depvar and indvar
                for term_ in setdiff(terms, [term]), r_ in upper_boundary_rules[operation(u_)][x_]
                    if subsmatch(term_, r_)
                        boundary = PeriodicBoundary(u_, x_, bc)
                    end
                end

                break
            end
        end
        # repeat for upper boundary
        if boundary === nothing
            for term in terms, r in reduce(vcat, reduce(vcat, collect.(values.(collect(values(upper_boundary_rules))))))
                if subsmatch(term, r)
                    u_, x_, order = r.second
                    boundary = UpperBoundary(u_, t, x_, order, bc, v)
                    break
                end
            end
        end
        @assert boundary !== nothing "Boundary condition $bc is not on a boundary of the domain, or is not a valid boundary condition"

        push!(boundarymap[operation(boundary.u)][boundary.x], boundary)
    end
    return boundarymap
end
