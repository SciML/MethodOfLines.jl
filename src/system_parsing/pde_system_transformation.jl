"""
Replace the PDESystem with an equivalent PDESystem which is compatible with MethodOfLines, mutates boundarymap, pmap and v

Modified copilot explanation:

"""
function transform_pde_system!(v, boundarymap, pmap, sys::PDESystem)
    eqs = [eq.lhs - eq.rhs ~ 0 for eq in sys.eqs]
    bcs = sys.bcs
    done = false
    # Replace bad terms with a greedy strategy until the system comes up clean
    while !done
        done = true
        for eq in eqs
            term, badterm, shouldexpand = descend_to_incompatible(eq.lhs, v)
            # Expand derivatives where possible
            if shouldexpand
                @show term
                @warn "Expanding derivatives in term $term."
                rule = term => expand_derivatives(term)
                subs_alleqs!(eqs, bcs, rule)
                done = false
                break
                # Replace incompatible terms with auxiliary variables
            elseif badterm !== nothing
                @show term
                # mutates eqs, bcs and v, we remake a fresh v at the end
                create_aux_variable!(eqs, bcs, boundarymap, pmap, v, badterm)
                done = false
                break
            end
        end
    end

    sys = PDESystem(eqs, bcs, sys.domain, sys.ivs, Num.(v.ū), sys.ps, name=sys.name)
    return sys
end

"""
Returns the term if it is incompatible, and whether to expand the term.
"""
function filter_equivalent_differentials(term, differential, v)
    S = Symbolics
    SU = SymbolicUtils
    if S.istree(term)
        op = SU.operation(term)
        if op isa Differential && isequal(op.x, differential.x)
            return filter_equivalent_differentials(SU.arguments(term)[1], differential, v)
        else
            return check_deriv_arg(term, v)
        end
    else
        return term, true
    end
end

"""
Check that term is a compatible derivative argument, and return the term if it is not and whether to expand.
"""
function check_deriv_arg(term, v)
    if istree(term)
        op = operation(term)
        if any(isequal(op), v.depvar_ops)
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
Check if term is a compatible part of a nonlinear laplacian, including spherical laplacian, and return the argument to the innermost derivative if it is.
"""
function nonlinlap_check(term, differential, v)
    if istree(term)
        op = operation(term)
        if op in [*, /]
            args = arguments(term)
            if operation(args[1]) == *
                term = args[1]
                args = arguments(term)
            end
            derivs = findall(args) do arg
                op = operation(arg)
                op isa Differential && isequal(op.x, differential.x)
            end
            if length(derivs) == 1
                return arguments(derivs[1])[1]
            end
        end
    end
    return nothing
end

"""
Finds incompatible terms in the equations and returns them with the incompatible part and whether to expand the term.
"""
function descend_to_incompatible(term, v)
    S = Symbolics
    SU = SymbolicUtils
    if S.istree(term)
        op = SU.operation(term)
        if op isa Differential
            if any(isequal(op.x), all_ivs(v))
                nonlinlapterm = nonlinlap_check(term, op, v)

                if nonlinlapterm !== nothing
                    badterm, shouldexpand = check_deriv_arg(nonlinlapterm, v)
                else
                    badterm, shouldexpand = filter_equivalent_differentials(term, op, v)
                end

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
    end
    return (nothing, nothing, false)
end

"""
Turn `term` in to an auxiliary variable, and replace it in the equations and boundary conditions.

Modified copilot explanation:
#= Here is the explanation for the code above:
1. First we create a new variable to represent our term, and replace the term with the new variable.
2. Then we generate the equation for the new variable, and add it to the equation list.
3. Then we generate the replacement rules for the boundary conditions, and substitute them into the new equation to infer auxiliary bcs.
4. Finally we add the new boundary conditions to the boundarymap and pmap. =#
"""
function create_aux_variable!(eqs, bcs, boundarymap, pmap, v, term)
    S = Symbolics
    SU = SymbolicUtils
    t = v.time
    newbcs = []

    # create a new variable
    newvar = diff2term(term)
    newop = operation(newvar)
    newargs = arguments(newvar)

    # update_varmap
    update_varmap!(v, newvar)
    # generate the replacement rule
    rule = term => newvar
    # apply the replacement rule to the equations and boundary conditions
    subs_alleqs!(eqs, bcs, rule)

    # add the new equation
    neweq = newvar ~ term

    @warn "Incompatible term found. Adding auxiliary equation $(neweq) to the system."

    newdepvars = [get_depvars(term, v.depvar_ops)...]
    neweqops = get_ops(newdepvars)

    # Add the new equation to the equation list
    push!(eqs, neweq.lhs - neweq.rhs ~ 0)

    newbcs = []
    # get a dict of each iv to all the bcs that depend on it
    bcivmap = reduce((d1, d2) -> mergewith(vcat, d1, d2), collect(values(boundarymap)))

    # Generate replacement rules for each individual boundary, in a closure for simpliciy
    function rulesforeachboundary(x, isupper)
        cond(b) = isupper ? b isa UpperBoundary : b isa LowerBoundary
        return generate_bc_rules(filter(cond, bcivmap[x]), v)
    end
    # Substitute the boundary conditions in to the new equation to infer the new boundary conditions
    # TODO: support robin/general boundary conditions somehow
    for dv in neweqops
        for iv in all_ivs(v)
            # if this is a periodic boundary, just add a new periodic condition
            if pmap.map[dv][iv] isa Val{true}
                args1 = substitute.(newargs, (iv => v.intervals[iv][1],))
                args2 = substitute.(newargs, (iv => v.intervals[iv][2],))
                push!(newbcs, PeriodicBoundary(newop(args1...), iv, newop(args1...) ~ newop(args2...)))
                continue
            end
            boundaries = boundarymap[dv][iv]
            length(bcs) == 0 && continue

            generate_aux_bcs!(newbcs, newvar, term, filter(isupper, boundaries), v, rulesforeachboundary(iv, true))
            generate_aux_bcs!(newbcs, newvar, term, filter(!isupper, boundaries), v, rulesforeachboundary(iv, false))
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
            pmap.map[newop][bc.x] = Val{true}()
        end
    end
    # update pmap
end

function generate_bc_rules(bcs, v)
    bcs = reverse(sort(bcs, by=bc -> bc.order))
    map(bcs) do bc
        deriv = bc.order == 0 ? identity : (Differential(bc.x)^bc.order)
        bcrule_lhs = deriv(operation(bc.u)(v.args[operation(bc.u)]...))
        bcterm = deriv(bc.u)
        rhs = solve_for(bc.eq, bcterm)
        bcrule_lhs => rhs
    end
end

function generate_aux_bcs!(newbcs, newvar, term, bcs, v, rules)
    t = v.time
    for bc in bcs
        x = bc.x
        val = isupper(bc) ? v.intervals[x][2] : v.intervals[x][1]
        newop = operation(newvar)
        args = arguments(newvar)
        args = substitute.(args, (x => val,))
        bcdv = newop(args...)
        deriv = bc.order == 0 ? identity : (Differential(x)^bc.order)

        bclhs = deriv(bcdv)
        # ! catch faliures to expand and throw a better error message
        bcrhs = expand_derivatives(substitute(deriv(term), rules))
        eq = bclhs ~ bcrhs

        newbc = if isupper(bc)
            UpperBoundary(bcdv, t, x, bc.order, eq, v)
        else
            LowerBoundary(bcdv, t, x, bc.order, eq, v)
        end
        push!(newbcs, newbc)
    end
end
