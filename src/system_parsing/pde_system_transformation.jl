"""
Replace the PDESystem with an equivalent PDESystem which is compatible with MethodOfLines, mutates boundarymap, pmap and v
"""
function transform_pde_system!(v, boundarymap, pmap, sys::PDESystem)
    eqs = [eq.lhs - eq.rhs ~ 0 for eq in sys.eqs]
    bcs = sys.bcs
    local done
    # Replace bad terms with a greedy strategy until the system comes up clean
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
                create_aux_variable!(eqs, bcs, boundarymap, pmap, v, badterm)
                done = false
                break
            end
        end
    end

    sys = PDESystem(eqs, bcs, sys.domains, sys.ivs, Num.(v.ū), ps = sys.ps, name = sys.name)
    return sys
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

function create_aux_variable!(eqs, bcs, boundarymap, pmap, v, term)
    S = Symbolics
    SU = SymbolicUtils
    t = v.time
    newbcs = []

    oldū = v.ū
    old_depvar_ops = v.depvar_ops

    # create a new variable
    newvar = S.diff2term(term)
    newop = operation(newvar)
    newargs = arguments(newvar)

    # update_varmap
    update_varmap!(v, newvar)
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

    newbcs = []

    for dv in old_depvar_ops
        for iv in all_ivs(v)
            # if this is a periodic boundary, just add a new periodic condition
            if pmap[dv][iv] isa Val{true}
                args = substitute.(newargs, (x=>v.intervals[iv][1],))
                push!(newbcs, PeriodicBoundary(newop(args...), iv))
                continue
            end
            boundaries = boundarymap[dv][iv]
            generate_aux_bcs!(newbcs, newop, term, boundaries, v)
        end
    end

    newbcs = unique(newbcs)

    # add the new bc equations
    append!(bcs, map(bc -> bc.eq, newbcs))

    # Add the new boundary conditions and initial conditions to the boundarymap
    merge!(boundarymap, Dict(newop => Dict(iv => [] for iv in all_ivs(v))))
    merge!(pmap.map, Dict(newop => Dict(iv => Val{false}() for iv in all_ivs(v))))
    for bc in newbcs
        push!(boundarymap[newop][bc.x], bc)
        if bc isa PeriodicBoundary
            pmap[newop][bc.x] = Val{true}()
        end
    end
    # update pmap
end

function generate_bc_rules(bcs, v)
    bcs = reverse(sort(bcs, by = bc -> bc.order))
    map(bcs) do bc
        if bc.order !== 0
            bcrule_lhs = (Differential(bc.x)^bc.order)(operation(bc.u)(v.args[operation(bc.u)]))
            bcterm = (Differential(bc.x)^bc.order)(bc.u)
        else
            bcrule_lhs = operation(bc.u)(v.args[operation(bc.u)]...)
            bcterm = bc.u
        end
        rhs = solve_for(bc.eq, bcterm)
        bcrule_lhs => rhs
    end
end

function generate_aux_bcs!(newbcs, newop, term, bcs, v)
    t = v.time
    rules = generate_bc_rules(bcs, v)
    for bc in bcs
        iv = bc.x
        val = isupper(bc) ? v.intervals[iv][2] : v.intervals[iv][1]
        args = arguments(newvar)
        args = substitute.(args, (iv=>val,))
        bcdv = newop(args...)
        deriv = bc.order == 0 ? identity : (Derivative(iv)^bc.order)

        bclhs = deriv(bcdv)
        # ! catch faliures to expand and throw a better error
        bcrhs = expand_derivatives(substitute(deriv(term), rules))
        eq = bclhs ~ bcrhs

        newbc = if isupper(bc)
            UpperBoundary(bcdv, t, iv, bc.order, eq)
        else
            LowerBoundary(bcdv, t, iv, bc.order, eq)
        end
        push!(newbcs, newbc)
    end
end
