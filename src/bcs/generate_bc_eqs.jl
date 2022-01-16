### INITIAL AND BOUNDARY CONDITIONS ###
"""
Mutates bceqs and u0 by finding relevant equations and discretizing them
"""
function generate_u0_and_bceqs!!(u0, bceqs, bcs, s, depvar_ops, tspan, derivweights)
    t=s.time
    
    if t === nothing
        initmaps = s.vars
    else
        initmaps = substitute.(s.vars,[t=>tspan[1]])
    end

    # Generate initial conditions and bc equations
    for bc in bcs
        bcdepvar = first(get_depvars(bc.lhs, depvar_ops))
        if any(u->isequal(operation(u),operation(bcdepvar)), s.vars)
            if t !== nothing && operation(bc.lhs) isa Sym && !any(x -> isequal(x, t.val), arguments(bc.lhs))
                # initial condition
                # Assume in the form `u(...) ~ ...` for now
                initindex = findfirst(isequal(bc.lhs), initmaps) 
                if initindex !== nothing
                    push!(u0,vec(s.discvars[s.vars[initindex]] .=> substitute.((bc.rhs,),gridvals(s))))
                end
            else
                # boundary condition
                # TODO: Seperate out Iedge in to individual boundaries and map each seperately to save time and catch the periodic case, have a look at the old maps for how to match the bcs
                push!(bceqs, vec(map(s.Iedge) do II
                    rules = generate_finite_difference_rules(II, s, bc, derivweights)
                    rules = vcat(rules, generate_bc_rules(II, s, bc , derivweights))
                    
                    substitute(bc.lhs, rules) ~ substitute(bc.rhs, rules)
                end))
            end
        end
    end
end

#function generate_u0_and_bceqs_with_rules!!(u0, bceqs, bcs, t, s, depvar_ops)