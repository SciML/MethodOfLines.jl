### INITIAL AND BOUNDARY CONDITIONS ###

abstract type AbstractBoundary end

abstract type AbstractTruncatingBoundary <: AbstractBoundary end

abstract type AbstractExtendingBoundary <: AbstractBoundary end

struct LowerBoundary <: AbstractTruncatingBoundary
end

struct UpperBoundary<: AbstractTruncatingBoundary
end

struct CompleteBoundary <: AbstractTruncatingBoundary
end#

struct PeriodicBoundary <: AbstractBoundary
end

struct BoundaryHandler{hasperiodic}
    boundaries::Dict{Num, AbstractBoundary}
end

"""
Mutates bceqs and u0 by finding relevant equations and discretizing them.
TODO: return a handler for use with generate_finite_difference_rules and pull out initial condition
"""
function BoundaryHandler!!(u0, bceqs, bcs, s, depvar_ops, tspan, derivweights)
    
    t=s.time
    
    if t === nothing
        initmaps = s.vars
    else
        initmaps = substitute.(s.vars,[t=>tspan[1]])
    end

    # Create some rules to match which bundary/variable a bc concerns
    # ? Is it nessecary to check whether all other args are present?
    lower_boundary_rules = vec([@rule operation(u)(~~a, lowerboundary(x), ~~b) => IfElse.ifelse(all(y-> y in vcat(~~a, ~~b), setdiff(x, arguments(u))), x, nothing) for x in setdiff(arguments(u), t), u in s.vars])

    upper_boundary_rules = vec([@rule operation(u)(~~a, upperboundary(x), ~~b) => IfElse.ifelse(all(y-> y in vcat(~~a, ~~b), setdiff(x, arguments(u))), x, nothing) for x in setdiff(arguments(u), t), u in s.vars])

    # Generate initial conditions and bc equations
    for bc in bcs
        bcdepvar = first(get_depvars(bc.lhs, depvar_ops))
        if any(u -> isequal(operation(u), operation(bcdepvar)), s.vars)
            if t !== nothing && operation(bc.lhs) isa Sym && !any(x -> isequal(x, t.val), arguments(bc.lhs))
                # initial condition
                # * Assume in the form `u(...) ~ ...` for now
                # * Assume that the initial condition is not in terms of the initial derivative
                initindex = findfirst(isequal(bc.lhs), initmaps) 
                if initindex !== nothing
                    push!(u0,vec(s.discvars[s.vars[initindex]] .=> substitute.((bc.rhs,),gridvals(s))))
                end
            else
                # Split out additive terms
                rhs_arg = istree(pde.rhs) && (SymbolicUtils.operation(pde.rhs) == +) ? SymbolicUtils.arguments(pde.rhs) : [pde.rhs]
                lhs_arg = istree(pde.lhs) && (SymbolicUtils.operation(pde.lhs) == +) ? SymbolicUtils.arguments(pde.lhs) : [pde.lhs]

                u_, x_ = (nothing, nothing)
                terms = vcat(lhs_arg,rhs_arg)
                # * Assume that the term of the condition is applied additively and has no multiplier/divisor/power etc.
                boundary = nothing
                # Check whether the bc is on the lower boundary, or periodic
                for term in terms, r in lower_boundary_rules
                    if r(term) !== nothing
                        u_, x_ = (term, r(term))
                        boundary = LowerBoundary()
                        for term_ in setdiff(terms, term)
                            for r in upper_boundary_rules
                                if r(term_) !== nothing
                                    # boundary = :periodic
                                    #TODO: Add handling for perioodic boundary conditions here
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

                @assert boundary !== nothing "Boundary condition ${bc} is not on a boundary of the domain, or is not a valid boundary condition"
                
                push!(bceqs, vec(map(s.Iedge[x_][boundary]) do II
                    rules = generate_bc_rules(II, s, bc, u_, boundary, derivweights)
                    rules = vcat(rules, )
                    
                    substitute(bc.lhs, rules) ~ substitute(bc.rhs, rules)
                end))
            end
        else 
            throw(ArgumentError("No active variables in boundary condition $bc lhs, please ensure that bcs are "))
    end
end

function generate_bc_rules(II, dim, s, bc, u_, x_, boundary::AbstractTruncatingBoundary, derivweights, G::CenterAlignedGrid) 
    # depvarbcmaps will dictate what to replace the variable terms with in the bcs
    # replace u(t,0) with u₁, etc
    ufunc(v, I, x) = s.discvars[v][I]

    for u in s.vars
        if isequal(operation(u), operation(u_))
            # What to replace derivatives at the boundary with
            depvarderivbcmaps = [(Differential(x)^d)(u_) => central_difference(derivweights.map[Differential(x_)^d], II, s, (s.x2i[x_],x_), u, ufunc) for d in derivweights.orders[x_]]
            # ? Does this need to be done for all variables at the boundary?
            depvarbcmaps = [u_ => s.discvars[u][II]]
        end
    end
    
    fd_rules = generate_finite_difference_rules(II, s, bc, derivweights)
    varrules = axiesvals(s, II)
    
    return vcat(depvarderivbcmaps, depvarbcmaps, fd_rules, varrules)
end

function generate_bc_rules(II, dim, s, bc, u_, x_, boundary::AbstractTruncatingBoundary, derivweights, G::EdgeAlignedGrid) 
    
    boundaryoffset(::LowerBoundary) = 1/2
    boundaryoffset(::UpperBoundary) = -1/2
    ufunc(v, I, x) = s.discvars[v][I]

    # depvarbcmaps will dictate what to replace the variable terms with in the bcs
    # replace u(t,0) with u₁, etc
    for u in s.vars
        if isequal(operation(u), operation(u_))
            depvarderivbcmaps = [(Differential(x)^d)(u_) => half_offset_centered_difference(derivweights.halfoffsetmap[Differential(x_)^d], II, s, offset(boundary), (s.x2i[x_],x_), u, ufunc) for d in derivweights.orders[x_]]
    
            depvarbcmaps = [u_ => half_offset_centered_difference(derivweights.interpmap[x_], II, s, offset(boundary), (s.x2i[x_],x_), u, ufunc)]
        end
    end
    
    fd_rules = generate_finite_difference_rules(II, s, bc, derivweights)
    varrules = axiesvals(s, II)
    valr = valrules(s, II)
    
    return vcat(depvarderivbcmaps, depvarbcmaps, fd_rules, varrules)
end