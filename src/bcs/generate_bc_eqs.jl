function _boundary_rules(s, orders, val)
    args = copy(s.params)

    if isequal(val, floor(val))
        args = [substitute.(args, (x=>val,)), substitute.(args, (x=>Int(val),))]
    else
        args = [substitute.(args, (x=>val,))]
    end
    substitute.(args, (x=>lowerboundary(x),))
    
    rules = [@rule operation(u)(arg...) => (u, x) for u in s.vars, arg in args]

    return vcat(rules, vec([@rule (Differential(x)^d)(operation(u)(arg...)) => (u, x) for d in orders[x], u in s.vars, arg in args]))
end

function generate_boundary_matching_rules(s, orders)
    # TODO: Check for bc equations of multiple variables
    lowerboundary(x) = first(s.axies[x])
    upperboundary(x) = last(s.axies[x])

    # Rules to match boundary conditions on the lower boundaries
    lower = reduce(vcat, [_boundary_rules(s, orders, lowerboundary(x)) for x in s.vars])

    upper = reduce(vcat, [_boundary_rules(s, orders, upperboundary(x)) for x in s.vars])

    return (lower, upper)
end

"""
Mutates bceqs and u0 by finding relevant equations and discretizing them.
TODO: return a handler for use with generate_finite_difference_rules and pull out initial condition. Important to remember that BCs can have 
"""
function BoundaryHandler!!(u0, bceqs, bcs, s::DiscreteSpace, depvar_ops, tspan, derivweights::DifferentialDiscretizer) 
    
    t=s.time
    
    if t === nothing
        initmaps = s.vars
    else
        initmaps = substitute.(s.vars,[t=>tspan[1]])
    end

    # Create some rules to match which bundary/variable a bc concerns
    # * Assume that the term of the condition is applied additively and has no multiplier/divisor/power etc.
    
    ## BC matching rules, returns the variable and parameter the bc concerns

    lower_boundary_rules, upper_boundary_rules = generate_boundary_matching_rules(s, derivweights.orders)

    # indexes for Iedge depending on boundary type
    idx(::LowerBoundary) = 1
    idx(::UpperBoundary) = 2

    # Generate initial conditions and bc equations
    for bc in bcs
        # * Assume in the form `u(...) ~ ...` for now
        bcdepvar = first(get_depvars(bc.lhs, depvar_ops))
        
        if any(u -> isequal(operation(u), operation(bcdepvar)), s.vars)
            if t !== nothing && operation(bc.lhs) isa Sym && !any(x -> isequal(x, t.val), arguments(bc.lhs))
                # initial condition
                # * Assume that the initial condition is not in terms of the initial derivative
                initindex = findfirst(isequal(bc.lhs), initmaps) 
                if initindex !== nothing
                    push!(u0,vec(s.discvars[s.vars[initindex]] .=> substitute.((bc.rhs,),gridvals(s))))
                end
            else
                # Split out additive terms
                terms = split_additive_terms(bc)

                u_, x_ = (nothing, nothing)
                boundary = nothing
                # Check whether the bc is on the lower boundary, or periodic
                for term in terms, r in lower_boundary_rules
                    if r(term) !== nothing
                        u_, x_ = (term, r(term))
                        boundary = LowerBoundary()
                        for term_ in setdiff(terms, term)
                            for r in upper_boundary_rules
                                if r(term_) !== nothing
                                    # boundary = PeriodicBoundary()
                                    #TODO: Add handling for perioodic boundary conditions here
                                end
                            end
                        end
                        break
                    end
                end
                for term in terms, r in upper_boundary_rules
                    if r(term) !== nothing
                        u_, x_ = (term, r(term))
                        boundary = UpperBoundary()
                        break
                    end
                end

                @assert boundary !== nothing "Boundary condition $bc is not on a boundary of the domain, or is not a valid boundary condition"
                
                push!(bceqs, vec(map(s.Iedge[x_][idx(boundary)]) do II
                    rules = generate_bc_rules(II, derivweights, s, bc, u_, x_, boundary)
                    
                    substitute(bc.lhs, rules) ~ substitute(bc.rhs, rules)
                end))
            end
        end
    end
end

function generate_bc_rules(II, derivweights, s::DiscreteSpace{N,M,G}, bc, u_, x_, ::AbstractTruncatingBoundary) where {N, M, G<:CenterAlignedGrid}
    # depvarbcmaps will dictate what to replace the variable terms with in the bcs
    # replace u(t,0) with u₁, etc
    ufunc(v, I, x) = s.discvars[v][I]

    depvarderivbcmaps = []
    depvarbcmaps = []

    # * Assume that the BC is in terms of an explicit expression, not containing references to variables other than u_ at the boundary
    for u in s.vars
        if isequal(operation(u), operation(u_))
            # What to replace derivatives at the boundary with
            depvarderivbcmaps = [(Differential(x)^d)(u_) => central_difference(derivweights.map[Differential(x_)^d], II, s, (s.x2i[x_], x_), u, ufunc) for d in derivweights.orders[x_]]
            # ? Does this need to be done for all variables at the boundary?
            depvarbcmaps = [u_ => s.discvars[u][II]]
            break
        end
    end
    
    fd_rules = generate_finite_difference_rules(II, s, bc, derivweights)
    varrules = axiesvals(s, x_, II)

    # valrules should be caught by depvarbcmaps and varrules if the above assumption holds
    #valr = valrules(s, II)
    
    return vcat(depvarderivbcmaps, depvarbcmaps, fd_rules, varrules)
end

function generate_bc_rules(II, derivweights, s::DiscreteSpace{N,M,G}, bc, u_, x_, boundary::AbstractTruncatingBoundary) where {N, M, G<:EdgeAlignedGrid}
    
    offset(::LowerBoundary) = 1/2
    offset(::UpperBoundary) = -1/2
    ufunc(v, I, x) = s.discvars[v][I]

    depvarderivbcmaps = []
    depvarbcmaps = []

    # depvarbcmaps will dictate what to replace the variable terms with in the bcs
    # replace u(t,0) with u₁, etc
    # * Assume that the BC is in terms of an explicit expression, not containing references to variables other than u_ at the boundary
    for u in s.vars
        if isequal(operation(u), operation(u_))
            depvarderivbcmaps = [(Differential(x)^d)(u_) => half_offset_centered_difference(derivweights.halfoffsetmap[Differential(x_)^d], II, s, offset(boundary), (s.x2i[x_],x_), u, ufunc) for d in derivweights.orders[x_]]
    
            depvarbcmaps = [u_ => half_offset_centered_difference(derivweights.interpmap[x_], II, s, offset(boundary), (s.x2i[x_],x_), u, ufunc)]
            break
        end
    end
    
    fd_rules = generate_finite_difference_rules(II, s, bc, derivweights)
    varrules = axiesvals(s, x_, II)

    # valrules should be caught by depvarbcmaps and varrules if the above assumption holds
    #valr = valrules(s, II)
    
    return vcat(depvarderivbcmaps, depvarbcmaps, fd_rules, varrules)
end