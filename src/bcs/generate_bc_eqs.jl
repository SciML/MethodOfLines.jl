function _boundary_rules(s, orders, x, val)
    #TODO: Change this around to detect boundary variables anywhere in the eq
   args = copy(s.params)

    args = substitute.(args, (x=>val,))

    rules = [operation(u)(args...) => [operation(u)(args...), x] for u in s.ū]

    return vcat(rules, vec([(Differential(x)^d)(operation(u)(args...)) => [operation(u)(args...), x] for d in orders[x], u in s.ū]))
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


#---- Count Boundary Equations --------------------
# Count the number of boundary equations that lie at the spatial boundary on
# both the left and right side. This will be used to determine number of
# interior equations s.t. we have a balanced system of equations.

# get the depvar boundary terms for given depvar and indvar index.

# return the counts of the boundary-conditions that reference the "left" and
# "right" edges of the given independent variable. Note that we return the
# max of the count for each depvar in the system of equations.
@inline function get_bc_counts(boundarymap, s, pde)
    timederivs = [Differential(s.time)(u) => 0 for u in s.ū]
    left = 0
    right = 0
    for (k,u) in enumerate(s.ū)
        if subsmatch(pde, timederivs[k])
            for boundary in boundarymap[operation(u)]
                wb = whichboundary(boundary)
                left += wb[1]
                right += wb[2]
            end
            break
        end
    end
    return [left, right]
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
                # * Assume that the initial condition is not in terms of the initial derivative
                initindex = findfirst(isequal(bc.lhs), initmaps) 
                if initindex !== nothing
                    push!(u0,vec(s.discvars[s.ū[initindex]] .=> substitute.((bc.rhs,),gridvals(s))))
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
                        boundary = LowerBoundary(u_, x_)
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
                            boundary = UpperBoundary(u_, x_)
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

@inline function generate_bc_rules!(bceqs, derivweights::DifferentialDiscretizer, s, bc, boundary::AbstractTruncatingBoundary)
    push!(bceqs, vec(map(edge(s,boundary)) do II
        rules = generate_bc_rules(II, derivweights, s, bc, boundary)
        
        substitute(bc.lhs, rules) ~ substitute(bc.rhs, rules)
    end))
end


function generate_bc_rules(II, derivweights, s::DiscreteSpace{N,M,G}, bc, boundary::AbstractTruncatingBoundary) where {N, M, G<:CenterAlignedGrid}
    # depvarbcmaps will dictate what to replace the variable terms with in the bcs
    # replace u(t,0) with u₁, etc
    u_, x_ = getvars(boundary)
    ufunc(v, I, x) = s.discvars[v][I]

    depvarderivbcmaps = []
    depvarbcmaps = []
    
    # * Assume that the BC is in terms of an explicit expression, not containing references to variables other than u_ at the boundary
    for u in s.ū
        if isequal(operation(u), operation(u_))
            # What to replace derivatives at the boundary with
            depvarderivbcmaps = [(Differential(x_)^d)(u_) => central_difference(derivweights.map[Differential(x_)^d], II, s, (s.x2i[x_], x_), u, ufunc) for d in derivweights.orders[x_]]
            # ? Does this need to be done for all variables at the boundary?
            depvarbcmaps = [u_ => s.discvars[u][II]]
            break
        end
    end
    fd_rules = generate_finite_difference_rules(II, s, bc, derivweights)
    varrules = axiesvals(s, x_, II)

    # valrules should be caught by depvarbcmaps and varrules if the above assumption holds
    #valr = valrules(s, II)
    #for condition in fd_rules.conditions
    return vcat(depvarderivbcmaps, depvarbcmaps, fd_rules, varrules)
end

function generate_bc_rules(II, derivweights, s::DiscreteSpace{N,M,G}, bc, boundary::AbstractTruncatingBoundary) where {N, M, G<:EdgeAlignedGrid}
    
    u_, x_ = getvars(boundary)
    ufunc(v, I, x) = s.discvars[v][I]

    depvarderivbcmaps = []
    depvarbcmaps = []

    # depvarbcmaps will dictate what to replace the variable terms with in the bcs
    # replace u(t,0) with u₁, etc
    # * Assume that the BC is in terms of an explicit expression, not containing references to variables other than u_ at the boundary
    j = s.x2i[x_]
    shift(::LowerBoundary) = zero(II)
    shift(::UpperBoundary) = unitindex(N, j)
    for u in s.ū
        if isequal(operation(u), operation(u_))
            depvarderivbcmaps = [(Differential(x_)^d)(u_) => half_offset_centered_difference(derivweights.halfoffsetmap[Differential(x_)^d], II-shift(boundary), s, (j,x_), u, ufunc) for d in derivweights.orders[x_]]
    
            depvarbcmaps = [u_ => half_offset_centered_difference(derivweights.interpmap[x_], II-shift(boundary), s, (j,x_), u, ufunc)]
            break
        end
    end
    
    fd_rules = generate_finite_difference_rules(II, s, bc, derivweights)
    varrules = axiesvals(s, x_, II)

    # valrules should be caught by depvarbcmaps and varrules if the above assumption holds
    #valr = valrules(s, II)
    
    return vcat(depvarderivbcmaps, depvarbcmaps, fd_rules, varrules)
end

function generate_bc_rules!(bceqs, derivweights, s::DiscreteSpace{N}, bc, boundary::PeriodicBoundary) where N
    # depvarbcmaps will dictate what to replace the variable terms with in the bcs
    # replace u(t,0) with u₁, etc
    u_, x_ = getvars(boundary)
    j = s.x2i[x_]
    # * Assume that the periodic BC is of the simple form u(t,0) ~ u(t,1)
    Ioffset = unitindex(N, j)*(length(s, x_) - 1)
    local disc
    for u in s.ū
        if isequal(operation(u), operation(u_))
            disc =  s.discvars[u]
            break
        end
    end

    
    push!(bceqs, vec(map(edge(s, boundary)) do II
        disc[II] ~ disc[II + Ioffset]
    end))

end
