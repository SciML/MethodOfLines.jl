### INITIAL AND BOUNDARY CONDITIONS ###

#TODO: Retire DiscreteSpace and move to a DiscreteEquation class with corresponding DiscreteVariables which carry all the required information, define substitute methods for these.

abstract type AbstractBoundary end

abstract type AbstractTruncatingBoundary <: AbstractBoundary end

abstract type AbstractExtendingBoundary <: AbstractBoundary end

struct LowerBoundary <: AbstractTruncatingBoundary
    u::Any
    x::Any
    depvars::Any
    indvars::Any
    eq::Any
    function LowerBoundary(u, t, x, eq, s, depvar_ops)
        depvars_lhs = get_depvars(eq.lhs, depvar_ops)
        depvars_rhs = get_depvars(eq.rhs, depvar_ops)
        depvars = collect(depvars_lhs ∪ depvars_rhs)
        #depvars =  filter(u -> !any(map(x-> x isa Number, arguments(u))), depvars)

        allx̄ = Set(
            filter(
                !isempty,
                map(
                    u -> filter(x -> t === nothing || !isequal(x, t.val), arguments(u)),
                    depvars,
                ),
            ),
        )
        return new(u, x, depvar.(depvars, [s]), first(allx̄), eq)
    end
end

struct UpperBoundary <: AbstractTruncatingBoundary
    u::Any
    x::Any
    depvars::Any
    indvars::Any
    eq::Any
    function UpperBoundary(u, t, x, eq, s, depvar_ops)
        depvars_lhs = get_depvars(eq.lhs, depvar_ops)
        depvars_rhs = get_depvars(eq.rhs, depvar_ops)
        depvars = collect(depvars_lhs ∪ depvars_rhs)

        allx̄ = Set(
            filter(
                !isempty,
                map(
                    u -> filter(x -> t === nothing || !isequal(x, t.val), arguments(u)),
                    depvars,
                ),
            ),
        )
        return new(u, x, depvar.(depvars, [s]), first(allx̄), eq)
    end
end
struct PeriodicBoundary <: AbstractBoundary
    u::Any
    x::Any
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
        edge = edge .- [I1 * (edge[1][j] - 1)]
    else
        edge = sd(size(interiormap.I[interiormap.pde[depvar(u, s)]], j))
        edge = edge .+ [I1 * (size(s.discvars[depvar(u, s)], j) - edge[1][j])]
    end
    return edge
end

edge(s, b, interiormap) = edge(interiormap, s, b.u, x2i(s, b.u, b.x), !isupper(b))

function _boundary_rules(s, orders, u, x, val)
    args = s.args[operation(u)]
    args = substitute.(args, (x => val,))
    varrule = operation(u)(args...) => [operation(u)(args...), x]
    return vcat(
        [
            (Differential(x)^d)(operation(u)(args...)) => [operation(u)(args...), x] for
            d in reverse(orders[x])
        ],
        Differential(s.time)(operation(u)(args...)) => [operation(u)(args...), x],
        varrule,
    )
end

function generate_boundary_matching_rules(s, orders)
    # TODO: Check for bc equations of multiple variables
    lowerboundary(x) = first(s.axies[x])
    upperboundary(x) = last(s.axies[x])

    # Rules to match boundary conditions on the lower boundaries
    lower = Dict([
        operation(u) => Dict([
            x => _boundary_rules(s, orders, u, x, lowerboundary(x)) for x in params(u, s)
        ]) for u in s.ū
    ])

    upper = Dict([
        operation(u) => Dict([
            x => _boundary_rules(s, orders, u, x, upperboundary(x)) for x in params(u, s)
        ]) for u in s.ū
    ])

    return (lower, upper)
end

"""
Creates a map of boundaries for each variable to be used later when discretizing the boundary condition equations, and 
"""
function BoundaryHandler(
    bcs,
    s::DiscreteSpace,
    depvar_ops,
    tspan,
    derivweights::DifferentialDiscretizer,
)

    t = s.time

    if t === nothing
        initmaps = s.ū
    else
        initmaps = substitute.(s.ū, [t => tspan[1]])
    end

    # Create some rules to match which bundary/variable a bc concerns
    # * Assume that the term of the condition is applied additively and has no multiplier/divisor/power etc.
    u0 = []
    bceqs = []
    ## BC matching rules, returns the variable and parameter the bc concerns

    lower_boundary_rules, upper_boundary_rules =
        generate_boundary_matching_rules(s, derivweights.orders)

    boundarymap = Dict([operation(u) => Dict([x => [] for x in s.x̄]) for u in s.ū])

    # Generate initial conditions and bc equations
    for bc in bcs
        # * Assume in the form `u(...) ~ ...` for now
        depvarslhs = get_depvars(bc.lhs, depvar_ops)
        bcdepvar = first(depvarslhs)

        depvars = depvar.(collect(union(depvarslhs, get_depvars(bc.rhs, depvar_ops))), (s,))
        if any(u -> isequal(operation(u), operation(bcdepvar)), s.ū)
            if t !== nothing &&
               ((operation(bc.lhs) isa Sym) | (operation(bc.lhs) isa Term)) &&
               !any(x -> isequal(x, t.val), arguments(bc.lhs))
                # initial condition
                # * Assume that the initial condition is not in terms of the initial derivative i.e. equation is first order in time
                initindex = findfirst(isequal(bc.lhs), initmaps)
                if initindex !== nothing
                    #@show bcdepvar, bc, depvar(bcdepvar, s)
                    args = params(depvar(bcdepvar, s), s)
                    indexmap = Dict([args[i] => i for i = 1:length(args)])
                    push!(
                        u0,
                        vec(
                            s.discvars[s.ū[initindex]] .=>
                                substitute.(
                                    (bc.rhs,),
                                    valmaps(s, depvar(bcdepvar, s), depvars, indexmap),
                                ),
                        ),
                    )
                end
            else
                # Split out additive terms
                terms = split_terms(bc)

                local u_, x_
                boundary = nothing
                # * Assume that the BC is defined on the edge of the domain
                # Check whether the bc is on the lower boundary, or periodic, we don't care which depvar/var
                for term in terms,
                    r in reduce(
                        vcat,
                        reduce(
                            vcat,
                            collect.(values.(collect(values(lower_boundary_rules)))),
                        ),
                    )
                    #Check if the rule changes the expression
                    if subsmatch(term, r)
                        # Get the matched variables from the rule
                        u_, x_ = r.second
                        # Mark the boundary                        
                        boundary = LowerBoundary(u_, s.time, x_, bc, s, depvar_ops)
                        # do it again for the upper end to check for periodic, but only check the current depvar and indvar
                        for term_ in setdiff(terms, [term]),
                            r_ in upper_boundary_rules[operation(u_)][x_]

                            if subsmatch(term_, r_)
                                boundary = PeriodicBoundary(u_, x_)
                            end
                        end

                        break
                    end
                end
                # repeat for upper boundary
                if boundary === nothing
                    for term in terms,
                        r in reduce(
                            vcat,
                            reduce(
                                vcat,
                                collect.(values.(collect(values(upper_boundary_rules)))),
                            ),
                        )

                        if subsmatch(term, r)
                            u_, x_ = r.second
                            boundary = UpperBoundary(u_, s.time, x_, bc, s, depvar_ops)
                            break
                        end
                    end
                end
                @assert boundary !== nothing "Boundary condition $bc is not on a boundary of the domain, or is not a valid boundary condition"

                push!(boundarymap[operation(boundary.u)][boundary.x], boundary)
                #generate_bc_rules!(bceqs, Iedge, derivweights, s, bc, boundary)
            end
        end
    end
    return boundarymap, u0
end
