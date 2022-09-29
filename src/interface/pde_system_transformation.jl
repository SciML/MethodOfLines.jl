function transform_pde_system(sys::PDESystem, discretization, v)
    eqs = [eq.lhs - eq.rhs ~ 0 for eq in sys.eqs]
    bcs = sys.bcs
    done = false
    # Replace bad terms until the system comes up clean
    while !done
        done = true
        for eq in eqs
            term, badterm, shouldexpand = descend_to_incompatible(eq.lhs, v)
            # Expand derivatives where possible
            if shouldexpand
                rule = term => expand_derivatives(term)
                subs_alleqs!(eqs, bcs, rule)
                done = false
                break
            # Replace incompatible terms with auxiliary variables
            elseif badterm !== nothing
                # mutates eqs, bcs and v, we remake a fresh v at the end
                create_aux_variable!(eqs, bcs, v, badterm)
                done = false
                break
            end
        end
    end
    v = VariableMap(eqs, v.x̄, v.ū, sys.domains, discretization)
    sys = PDESystem(eqs, bcs, sys.domains, sys.ivs, Num.(v.ū), ps = sys.ps, name = sys.name)
    return sys, v
end

"""
Returns the number of nested derivatives with the same iv, the term if it is incompatible, and whether to expand the term.
"""
function filter_equivalent_differentials(term, differential, v)
    S = Symbolics
    SU = SymbolicUtils
    if S.istree(term)
        op = SU.operation(term)
        if op isa Differential && isequal(op.x, differential.x)
            return filter_equivalent_differentials(SU.arguments(term)[1], differential, v.depvar_ops)
        elseif any(isequal.((op,), v.depvar_ops))
            return nothing, false
        else
            if length(get_depvars(term, v.depvar_ops)) == 0
                return term, true
            else
                return term, false
            end
        end
    else
        return term, true
    end
end

"""
Finds incompatible terms in the equations and replaces them with auxiliary variables, or expanded derivatives.
"""
function descend_to_incompatible(term, v)
    S = Symbolics
    SU = SymbolicUtils
    if S.istree(term)
        op = SU.operation(term)
        if op isa Differential
            if op.x in ivs
                badterm, shouldexpand = filter_equivalent_differentials(term, op, v.depvar_ops)
                if badterm !== nothing
                    return (term, badterm, shouldexpand)
                else
                    return (nothing, nothing, false)
                end
            else
                throw(ArgumentError("Variable derived with respect to is not an independent variable in ivs, got $(op.x) in $(term)"))
            end
        end
        for arg in SU.arguments(term)
            res = descend_to_incompatible(arg, v)
            if res[2] !== nothing
                return res
            end
        end
        return (nothing, nothing, false)
    else
        return (nothing, nothing, false)
    end
end

function create_aux_variable!(eqs, bcs, v, term)
    S = Symbolics
    SU = SymbolicUtils
    t = v.time
    newbcs = []

    # generate replacement rules for incompatible terms
    # create a new variable
    newvar = S.diff2term(term)
    # generate the replacement rule
    rule = term => newvar
    # apply the replacement rule to the equations and boundary conditions
    for (i, eq) in enumerate(eqs)
        eqs[i] = substitute(eq.lhs, rule) ~ 0
    end
    for (i, bc) in enumerate(bcs)
        bcs[i] = substitute(bc.lhs, rule) ~ substitute(bc.rhs, rule)
    end
    # add the new equation
    neweq = newvar ~ term
    push!(eqs, neweq.lhs - neweq.rhs ~ 0)
    # add the new variable to the list of dependent variables
    push!(dvs, newvar)
    # generate replacement rules for initial conditions
    orders = Dict(map(x -> x => d_orders_with0(x, term), all_ivs(v)))
    sorted_ics = sort(mapreduce(dv -> boundarymap[dv][t], vcat, v.ū), by=tc->tc.order)

    icrules = map(reverse(sorted_ics)) do ic
        ic isa FinalCondition && throw(ArgumentError("Final conditions are not yet supported"))
        if ic.order !== 0
            (Differential(ic.x)^ic.order)(operation(ic.u)(v.args[operation(ic.u)])) => ic.rhs
        else
            operation(ic.u)(v.args[operation(ic.u)]...) => ic.rhs
        end
    end

    newics =  map(reverse(orders[t])) do o
        args = arguments(newvar)
        args = substitute.(args, (x=>v.intervals[t][1],))
        lhsvar = o == 0 ? operation(newvar)(args...) : (Derivative(t)^o)(operation(newvar)(args...))
        lhsvar ~ expand_derivatives(substitute(term, icrules))
    end

    # ! Work out how many boundaries are in the map and create rules for each
    # ! Calculate how many new bcs are needed and add them to the list of bcs
end
