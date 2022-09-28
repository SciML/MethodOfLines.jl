function transform_pde_system(sys::PDESystem)
    eqs = [eq.lhs - eq.rhs ~ 0 for eq in sys.eqs]
    bcs = sys.bcs
    domains = sys.domains
    independent_variables = sys.iv
    dependent_variables = sys.dv


end

"""
Returns the number of nested derivatives with the same iv, the term if it is incompatible, and whether to expand the term.
"""
function filter_equivalent_differentials(term, count, differential, depvar_ops)
    S = Symbolics
    SU = SymbolicUtils
    if S.istree(term)
        op = SU.operation(term)
        if op isa Differential && isequal(op.x, differential.x)
            return filter_equivalent_differentials(SU.arguments(term)[1], count + 1, differential, depvar_ops)
        elseif any(isequal.((op,), depvar_ops))
            return count, nothing, false
        else
            if length(get_depvars(eq, depvar_ops)) == 0
                return count, term, true
            else
                return count, term, false
            end
        end
    else
        return 0, term, true
    end
end

"""
Finds incompatible terms in the equations and replaces them with auxiliary variables, or expanded derivatives.
"""
function descend_to_incompatible!(rules, term, ivs)
    S = Symbolics
    SU = SymbolicUtils
    if S.istree(term)
        op = SU.operation(term)
        if op isa Differential
            if op.x in ivs
                count, badterm, shouldexpand = filter_equivalent_differentials(term, 0, op, depvar_ops)
                if shouldexpand
                    rule = term => expand_derivatives(term)
                    for (i, eq) in enumerate(eqs)
                        eqs[i] = substitute(eq.lhs, rule) ~ 0
                    end
                elseif badterm !== nothing
                    create_aux_variable!(eqs, bcs, badterm)

            else
                throw(ArgumentError("Variable derived with respect to is not an independent variable in ivs, got $(op.x) in $(term)"))
            end
        end
        for arg in SU.arguments(term)
            res = descend_to_incompatible(out, arg, ivs)
            return res
        end
    else
        return (nothing, nothing)
    end
end

function create_aux_variable!(eqs, bcs, v, term)
    S = Symbolics
    SU = SymbolicUtils

    # generate replacement rules for incompatible terms
    # create a new variable
    newvar = S.diff2term(term)
    # generate the replacement rule
    rule = term => newvar
    # apply the replacement rule to the equations and boundary conditions
    for (i, eq) in enumerate(eqs)
        eqs[i] = substitute(eq.lhs, rule) ~ substitute(eq.rhs, rule)
    end
    for (i, bc) in enumerate(bcs)
        bcs[i] = substitute(bc.lhs, rule) ~ substitute(bc.rhs, rule)
    end
    # add the new equation
    push!(eqs, newvar - term ~ 0)
    # add the new variable to the list of dependent variables
    push!(dvs, newvar)
    # generate replacement rules for initial conditions
    sorted_ics = reverse(sort(mapreduce(dv -> boundarymap[dv][t], vcat, v.ū), by=tc->tc.order))
    ic_rules =  map(sorted_ics) do ic
        if ic.order !== 0
            (Differential(ic.x)^ic.order)(operation(ic.u)(v.args[operation(ic.u)])) => ic.rhs
        else
            operation(ic.u)(v.args[operation(ic.u)]...) => ic.rhs
        end
    end
    # ! Work out how many boundaries are in the map and create rules for each
    # ! Calculate how many new bcs are needed and add them to the list of bcs
end

function check_mixed_differentials(ivs)
    S = Symbolics
    SU = SymbolicUtils
    for iv in ivs, eq in eqs
        if S.istree(eq)
            op = SU.operation(eq)
            if op isa Differential
                if !isequal(op.x, iv)
                    return true
                end
            end
        end
    end
