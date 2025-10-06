"""
Find all function-like calls in an expression that are not in the known depvar_ops list.
This helps detect when a user has forgotten to include a variable in the PDESystem.
"""
function find_unknown_variables(expr, v)
    S = Symbolics
    SU = SymbolicUtils
    unknown_vars = Set()

    # Common operators to ignore
    builtin_ops = Set([+, -, *, /, ^, sqrt, sin, cos, tan, exp, log, abs])

    function traverse(ex)
        ex = safe_unwrap(ex)
        if S.iscall(ex)
            op = SU.operation(ex)
            args = SU.arguments(ex)

            # Skip Differential, Integral, and built-in mathematical operators
            if !(op isa Differential || op isa Integral || op in builtin_ops)
                # Check if this is a function call with arguments that could be a dependent variable
                # Look for function-like operations (BasicSymbolic with FnType)
                if op isa SymbolicUtils.BasicSymbolic && !isempty(args) &&
                   !any(isequal(op), v.depvar_ops)
                    # Check if any argument is an independent variable
                    has_iv = false
                    for arg in args
                        arg_unwrap = safe_unwrap(arg)
                        if any(iv -> isequal(arg_unwrap, iv), all_ivs(v))
                            has_iv = true
                            break
                        end
                    end
                    if has_iv
                        push!(unknown_vars, op)
                    end
                end
            end

            # Recursively traverse arguments
            for arg in args
                traverse(arg)
            end
        end
    end

    traverse(expr)
    return unknown_vars
end

"""
Replace the PDESystem with an equivalent PDESystem which is compatible with MethodOfLines, mutates boundarymap and v

Modified copilot explanation:

"""
function PDEBase.transform_pde_system!(
        v::PDEBase.VariableMap, boundarymap, sys::PDESystem, disc::MOLFiniteDifference)
    eqs = copy(sys.eqs)
    bcs = copy(sys.bcs)

    # Pre-validate: check for unknown variables in equations before transformation
    all_unknown_vars = Set()
    for eq in eqs
        unknown_lhs = find_unknown_variables(eq.lhs, v)
        unknown_rhs = find_unknown_variables(eq.rhs, v)
        union!(all_unknown_vars, unknown_lhs, unknown_rhs)
    end
    for bc in bcs
        unknown_lhs = find_unknown_variables(bc.lhs, v)
        unknown_rhs = find_unknown_variables(bc.rhs, v)
        union!(all_unknown_vars, unknown_lhs, unknown_rhs)
    end

    if !isempty(all_unknown_vars)
        unknown_vars_str = join(string.(collect(all_unknown_vars)), ", ")
        throw(ArgumentError("Found unknown symbolic variable(s): $unknown_vars_str. These variables appear in the equations or boundary conditions but were not included in the dependent variables list of the PDESystem. Please add them to the dependent variables, e.g.: PDESystem(eqs, bcs, domains, ivs, [existing_vars..., $unknown_vars_str])"))
    end

    done = false
    # Replace bad terms with a greedy strategy until the system comes up clean.
    # Track previous terms to detect infinite loops
    seen_terms = Set()
    max_iterations = 1000
    iteration_count = 0

    while !done
        done = true
        iteration_count += 1

        if iteration_count > max_iterations
            throw(ArgumentError("Maximum iterations exceeded in system transformation. This likely indicates an infinite loop due to unhandled symbolic variables. Please ensure all variables appearing in equations are included in the dependent variables list of the PDESystem."))
        end

        for eq in eqs
            term, badterm, shouldexpand = descend_to_incompatible(eq.lhs, v)
            # Expand derivatives where possible
            if shouldexpand
                # Check if we've seen this term before - indicates we're stuck
                if term in seen_terms
                    # Detect unknown variables in the term
                    unknown_vars = find_unknown_variables(term, v)
                    if !isempty(unknown_vars)
                        unknown_vars_str = join(string.(unknown_vars), ", ")
                        throw(ArgumentError("Found unknown symbolic variable(s) $unknown_vars_str in equation $(eq). These variables appear in the equations but were not included in the dependent variables list of the PDESystem. Please add them to the dependent variables: PDESystem(eqs, bcs, domains, ivs, [existing_vars..., $unknown_vars_str])"))
                    else
                        throw(ArgumentError("Infinite loop detected while expanding derivatives in term $term. The term cannot be expanded or transformed properly. This may indicate a problem with the equation structure."))
                    end
                end
                push!(seen_terms, term)

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

    sys = PDESystem(eqs, bcs, sys.domain, sys.ivs, Num.(v.ū),
        sys.ps, name = sys.name, defaults = sys.defaults)
    return sys
end

function PDEBase.should_transform(pdesys::PDESystem, disc::MOLFiniteDifference, boundarymap)
    if !disc.should_transform
        return false
    end
    if has_interfaces(boundarymap)
        @warn "The system contains interface boundaries, which are not compatible with system transformation. The system will not be transformed. Please post an issue if you need this feature."
        return false
    end
    return true
end

"""
Returns the term if it is incompatible, and whether to expand the term.
"""
function filter_differentials(term, differential, v, depth = 0)
    S = Symbolics
    SU = SymbolicUtils
    if S.iscall(term)
        op = SU.operation(term)
        if op isa Differential && isequal(op.x, differential.x)
            return filter_differentials(SU.arguments(term)[1], differential, v, depth + 1)
        elseif op isa Differential && !isequal(op.x, differential.x) && depth <= 1
            return check_deriv_arg(arguments(term)[1], v)
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
    if iscall(term)
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
    if iscall(term)
        op = operation(term)
        if (op == *) || (op == /)
            args = arguments(term)
            if iscall(args[1]) && operation(args[1]) == *
                term = args[1]
                args = arguments(term)
            elseif iscall(args[1]) && operation(args[1]) == /
                term = args[1]
                denominator = arguments(term)[2]
                has_derivatives(denominator) && return nothing
                args = arguments(term)
                if iscall(args[1]) && operation(args[1]) == *
                    term = args[1]
                    args = arguments(term)
                end
            end

            is = findall(args) do arg
                if iscall(arg)
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
    if S.iscall(term)
        op = SU.operation(term)
        if op isa Differential
            if any(isequal(op.x), all_ivs(v))
                nonlinlapterm = nonlinlap_check(arguments(term)[1], op)

                if nonlinlapterm !== nothing
                    badterm, shouldexpand = check_deriv_arg(nonlinlapterm, v)
                else
                    badterm,
                    shouldexpand = filter_differentials(
                        arguments(term)[1], op, v, 1)
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
                euler = isequal(
                    op.domain.domain.left, v.intervals[op.domain.variables][1]) &&
                        isequal(op.domain.domain.right, Num(op.domain.variables))
                whole = isequal(
                    op.domain.domain.left, v.intervals[op.domain.variables][1]) &&
                        isequal(op.domain.domain.right, v.intervals[op.domain.variables][2])
                if any([euler, whole])
                    u = arguments(term)[1]
                    out = check_deriv_arg(u, v)
                    @assert out==(nothing, false) "Integral $term must be purely of a variable, got $u. Try wrapping the integral argument with an auxiliary variable."
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
    if iscall(term)
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
            @assert length(interfaces)==0 "Interface BCs like $(interfaces[1].eq) are not yet supported in conjunction with system transformation, please transform manually if needed and set `should_transform=false` in the discretization. If you need this feature, please open an issue on GitHub."

            boundaries = boundarymap[dv][iv]
            length(bcs) == 0 && continue

            generate_aux_bcs!(newbcs, newvar, term, filter(isupper, boundaries),
                v, rulesforeachboundary(iv, true))
            generate_aux_bcs!(newbcs, newvar, term, filter(!isupper, boundaries),
                v, rulesforeachboundary(iv, false))
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
    bcs = reverse(sort(bcs, by = bc -> bc.order))
    map(bcs) do bc
        deriv = bc.order == 0 ? identity : (Differential(bc.x)^bc.order)
        bcrule_lhs = deriv(operation(bc.u)(v.args[operation(bc.u)]...))
        bcterm = deriv(bc.u)
        rhs = symbolic_linear_solve(bc.eq, bcterm)
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
    # ! catch failures to expand and throw a better error message
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
