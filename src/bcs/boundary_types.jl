### INITIAL AND BOUNDARY CONDITIONS ###

abstract type AbstractBoundary end

abstract type AbstractTruncatingBoundary <: AbstractBoundary end

abstract type AbstractExtendingBoundary <: AbstractBoundary end

struct LowerBoundary <: AbstractTruncatingBoundary
    u
    x
    eq
end

struct UpperBoundary <: AbstractTruncatingBoundary
    u
    x
    eq
end

struct PeriodicBoundary <: AbstractBoundary
    u
    x
end

getvars(b::AbstractBoundary) = (b.u, b.x)

struct BoundaryHandler{hasperiodic}
    boundaries::Dict{Num, AbstractBoundary}
end

# Which interior end to remove
whichboundary(::LowerBoundary) = (1, 0)
whichboundary(::UpperBoundary) = (0, 1)
whichboundary(::PeriodicBoundary) = (1, 0)

@inline function clip_interior!!(lower, upper, b::AbstractBoundary, x2i)
    clip = whichboundary(b)
    # This x2i is correct
    dim = x2i[b.x]

    lower[dim] = lower[dim] + clip[1]
    upper[dim] = upper[dim] + clip[2]
end


# indexes for Iedge depending on boundary type
isupper(::LowerBoundary) = false
isupper(::UpperBoundary) = true
isupper(::PeriodicBoundary) = false

@inline function edge(interiormap, s, u, j, islower)
    sd(i) = selectdim(interiormap.I[interiormap.pde[depvar(u, s)]], j, i)
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

function _boundary_rules(s, orders, x, val)
    rules = map(s.ū) do u
        args = s.args[operation(u)]
        args = substitute.(args, (x=>val,))
        operation(u)(args...) => [operation(u)(args...), x]
    end
    return vcat(rules, mapreduce(vcat, s.ū) do u
        args = s.args[operation(u)]
        args = substitute.(args, (x=>val,))
        [(Differential(x)^d)(operation(u)(args...)) => [operation(u)(args...), x] for d in orders[x]]
    end)
end

function generate_boundary_matching_rules(s, orders)
    # TODO: Check for bc equations of multiple variables
    lowerboundary(x) = first(s.axies[x])
    upperboundary(x) = last(s.axies[x])

    # Rules to match boundary conditions on the lower boundaries
    lower = reduce(vcat, [_boundary_rules(s, orders, x, lowerboundary(x)) for x in s.x̄])

    upper = reduce(vcat, [_boundary_rules(s, orders, x, upperboundary(x)) for x in s.x̄])

    return (lower, upper)
end

"""
Mutates bceqs and u0 by finding relevant equations and discretizing them.
TODO: return a handler for use with generate_finite_difference_rules and pull out initial condition. Important to remember that BCs can have 
"""
function BoundaryHandler(bcs, s::DiscreteSpace, depvar_ops, tspan, derivweights::DifferentialDiscretizer) 
    
    t=s.time
    
    if t === nothing
        initmaps = s.ū
    else
        initmaps = substitute.(s.ū,[t=>tspan[1]])
    end

    # Create some rules to match which bundary/variable a bc concerns
    # * Assume that the term of the condition is applied additively and has no multiplier/divisor/power etc.
    u0 = []
    bceqs = []
    ## BC matching rules, returns the variable and parameter the bc concerns

    lower_boundary_rules, upper_boundary_rules = generate_boundary_matching_rules(s, derivweights.orders)

    boundarymap = Dict([operation(u)=>[] for u in s.ū])

    # Generate initial conditions and bc equations
    for bc in bcs
        # * Assume in the form `u(...) ~ ...` for now
        bcdepvar = first(get_depvars(bc.lhs, depvar_ops))
        
        if any(u -> isequal(operation(u), operation(bcdepvar)), s.ū)
            if t !== nothing && operation(bc.lhs) isa Sym && !any(x -> isequal(x, t.val), arguments(bc.lhs))
                # initial condition
                # * Assume that the initial condition is not in terms of the initial derivative i.e. equation is first order in time
                initindex = findfirst(isequal(bc.lhs), initmaps) 
                if initindex !== nothing
                    push!(u0,vec(s.discvars[s.ū[initindex]] .=> substitute.((bc.rhs,),gridvals(s, depvar(bcdepvar,s)))))
                end
            else
                # Split out additive terms
                terms = split_additive_terms(bc)

                local u_, x_ 
                boundary = nothing
                # * Assume that the BC is defined on the edge of the domain
                # Check whether the bc is on the lower boundary, or periodic
                for term in terms, r in lower_boundary_rules
                    #Check if the rule changes the expression
                    if subsmatch(term, r)
                        # Get the matched variables from the rule
                        u_, x_ = r.second
                        # Mark the boundary                        
                        boundary = LowerBoundary(u_, x_, bc)
                        # do it again for the upper end to check for periodic
                        for term_ in setdiff(terms, [term]), r_ in upper_boundary_rules
                            if subsmatch(term_, r_)
                                boundary = PeriodicBoundary(u_, x_)
                            end
                        end

                        break
                    end
                end
                # repeat for upper boundary
                if boundary === nothing
                    for term in terms, r in upper_boundary_rules
                        if subsmatch(term, r)
                            u_, x_ = r.second 
                            boundary = UpperBoundary(u_, x_, bc)
                            break
                        end
                    end
                end
                @assert boundary !== nothing "Boundary condition $bc is not on a boundary of the domain, or is not a valid boundary condition"
                
                push!(boundarymap[operation(boundary.u)], boundary)
                #generate_bc_rules!(bceqs, Iedge, derivweights, s, bc, boundary)
            end
        end
    end
    return boundarymap, u0
end
