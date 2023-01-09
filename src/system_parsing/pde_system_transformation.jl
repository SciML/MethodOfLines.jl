"""
Replace the PDESystem with an equivalent PDESystem which is compatible with MethodOfLines, mutates boundarymap and v

Modified copilot explanation:

"""
function transform_pde_system!(v, boundarymap, sys::PDESystem)
    eqs = copy(sys.eqs)
    bcs = copy(sys.bcs)
    done = false
    # Replace bad terms with a greedy strategy until the system comes up clean
    while !done
        done = true
        for eq in eqs
            term, badterm, shouldexpand = descend_to_incompatible(eq.lhs, v)
            # Expand derivatives where possible
            if shouldexpand
                @warn "Expanding derivatives in term $term."
                rule = term => expand_derivatives(term)
                subs_alleqs!(eqs, bcs, rule)
                done = false
                break
                # Replace incompatible terms with auxiliary variables
            elseif badterm !== nothing
                # mutates eqs, bcs and v, we remake a fresh v at the end
                create_aux_variable!(eqs, bcs, boundarymap, v, badterm)
                done = false
                break
            end
        end
    end

    sys = PDESystem(eqs, bcs, sys.domain, sys.ivs, Num.(v.ū), sys.ps, name=sys.name)
    return sys
end

function cardinalize_eqs!(pdesys)
    for (i, eq) in enumerate(pdesys.eqs)
        pdesys.eqs[i] = eq.lhs - eq.rhs ~ 0
    end
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
function nonlinlap_check(term, differential)
    if istree(term)
        op = operation(term)
        if (op == *) || (op == /)
            args = arguments(term)
            if istree(args[1]) && operation(args[1]) == *
                term = args[1]
                args = arguments(term)
            elseif istree(args[1]) && operation(args[1]) == /
                term = args[1]
                denominator = arguments(term)[2]
                has_derivatives(denominator) && return nothing
                args = arguments(term)
                if istree(args[1]) && operation(args[1]) == *
                    term = args[1]
                    args = arguments(term)
                end
            end

            is = findall(args) do arg
                if istree(arg)
                    op = operation(arg)
                    op isa Differential && isequal(op.x, differential.x)
                else
                    false
                end
            end
            if length(is) == 1
                i = first(is)
                return arguments(args[i])[1]
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
                nonlinlapterm = nonlinlap_check(arguments(term)[1], op)

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
        elseif op isa Integral
            if any(isequal(op.domain.variables), v.x̄)
                euler = isequal(op.domain.domain.left, v.intervals[op.domain.variables][1]) && isequal(op.domain.domain.right, Num(op.domain.variables))
                whole = isequal(op.domain.domain.left, v.intervals[op.domain.variables][1]) && isequal(op.domain.domain.right, v.intervals[op.domain.variables][2])
                if any([euler, whole])
                    u = arguments(term)[1]
                    out = check_deriv_arg(u, v)
                    @assert out == (nothing, false) "Integral $term must be purely of a variable, got $u. Try wrapping the integral argument with an auxiliary variable."
                    return (nothing, nothing, false)
                else
                    throw(ArgumentError("Integration Domain only supported for integrals from start of iterval to the variable, got $(op.domain.domain) in $(term)"))
                end
            else
                throw(ArgumentError("Integral must be with respect to the independent variable in its upper-bound, got $(op.domain.variables) in $(term)"))
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
function create_aux_variable!(eqs, bcs, boundarymap, v, term)
    S = Symbolics
    SU = SymbolicUtils
    t = v.time
    newbcs = []

    # create a new variable
    if istree(term)
        op = operation(term)
        if op isa Differential
            newvar = diff2term(term)
        else
            newvar = ex2term(term, v)
        end
    else
        throw(ArgumentError("Term is not a tree, got $(term), this should never happen!"))
    end

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
    for dv in neweqops
        for iv in all_ivs(v)
            # if this is a periodic boundary, just add a new periodic condition
            interfaces = filter_interfaces(boundarymap[dv][iv])
            @assert length(interfaces) == 0 "Interface BCs like $(interfaces[1].eq) are not yet supported in conjunction with system transformation, please transform manually if needed and set `should_transform=false` in the discretization. If you need this feature, please open an issue on GitHub."

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
    update_boundarymap!(boundarymap, newbcs, newop, v)
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
    for bc in bcs
        generate_aux_bc!(newbcs, newvar, term, bc, v, rules)
    end
end

function generate_aux_bc!(newbcs, newvar, term, bc::AbstractTruncatingBoundary, v, rules)
    t = v.time
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

function update_boundarymap!(boundarymap, bcs, newop, v)
    merge!(boundarymap, Dict(newop => Dict(iv => [] for iv in all_ivs(v))))
    for bc1 in bcs
        for bc2 in setdiff(bcs, [bc1])
            if isequal(bc1.eq.lhs, bc2.eq.lhs) && isequal(bc1.eq.rhs, bc2.eq.rhs)
                bcs = setdiff(bcs, [bc2])
            end
        end
    end
    for bc in bcs
        push!(boundarymap[newop][bc.x], bc)
    end
end

function handle_complex(pdesys)
    eqs = pdesys.eqs
    if any(eq -> (eq isa Vector) || hascomplex(eq), eqs)
        dvmaps = map(pdesys.dvs) do dv
            args = arguments(dv)
            dv = operation(safe_unwrap(dv))
            resym = Symbol("Re"*string(dv))
            imsym = Symbol("Im"*string(dv))
            redv = collect(first(@variables $resym(..)))
            imdv = collect(first(@variables $imsym(..)))
            redv = operation(unwrap(redv(args...)))
            imdv = operation(unwrap(imdv(args...)))
            (dv => redv, dv => imdv)
        end
        redvmaps = map(dvmaps) do dvmap
            dvmap[1]
        end
        imdvmaps = map(dvmaps) do dvmap
            dvmap[2]
        end
        dvmaps = Dict(map(dvmaps) do dvmap
            dvmap[1].first => (dvmap[1].second, dvmap[2].second)
        end)

        eqs = mapreduce(vcat, eqs) do eq
            eq = split_complex(eq)
            if eq isa Vector
                eq1 = eq[1]
                eq2 = eq[2]
                eq1 = substitute(eq1, redvmaps)
                eq2 = substitute(eq2, imdvmaps)
            else
                eq1 = substitute(eq, redvmaps)
                eq2 = substitute(eq, imdvmaps)
            end
            [eq1, eq2]
        end

        bcs = mapreduce(vcat, pdesys.bcs) do eq
            eq = split_complex(eq)
            if eq isa Vector
                eq1 = eq[1]
                eq2 = eq[2]
                eq1 = substitute(eq1, redvmaps)
                eq2 = substitute(eq2, imdvmaps)
            else
                eq1 = substitute(eq, redvmaps)
                eq2 = substitute(eq, imdvmaps)
            end
            [eq1, eq2]
        end

        redvmaps = Dict(redvmaps)
        imdvmaps = Dict(imdvmaps)

        dvs = mapreduce(vcat, pdesys.dvs) do dv
            dv = safe_unwrap(dv)
            redv = redvmaps[operation(dv)](arguments(dv)...)
            imdv = imdvmaps[operation(dv)](arguments(dv)...)
            [redv, imdv]
        end

        pdesys = PDESystem(eqs, bcs, dvs, pdesys.ivs, pdesys.ps, name = pdesys.name)
    else
        dvmaps = nothing
    end
    return pdesys, dvmaps
end
