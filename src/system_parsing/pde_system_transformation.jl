"""
Replace the PDESystem with an equivalent PDESystem which is compatible with MethodOfLines, mutates boundarymap and v

Modified copilot explanation:

"""
function PDEBase.transform_pde_system!(
        v::PDEBase.VariableMap, boundarymap, sys::PDESystem, disc::MOLDiscretization
    )
    eqs = copy(get_eqs(sys))
    bcs = copy(get_bcs(sys))

    # When HermiteScheme is active, automatically introduce auxiliary derivative
    # variables (v = Dx(u), etc.) so the user does not need to reformulate the PDE.
    if disc isa MOLFiniteDifference && disc.advection_scheme isa HermiteScheme
        hermite_transform!(eqs, bcs, boundarymap, v, disc)
    end

    done = false
    # Replace bad terms with a greedy strategy until the system comes up clean.
    while !done
        done = true
        for eq in eqs
            term, badterm, shouldexpand = descend_to_incompatible(eq.lhs, v)
            # Expand derivatives where possible
            if shouldexpand
                expanded = expand_derivatives(term)
                if isequal(expanded, term)
                    throw(
                        ArgumentError(
                            "Could not expand derivatives in $term. If $badterm is a PDE unknown, add it to the PDESystem dependent-variable list."
                        )
                    )
                end
                @warn "Expanding derivatives in term $term."
                rule = term => expanded
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

    sys = PDESystem(
        eqs, bcs, sys.domain, sys.ivs, Num.(v.ū),
        sys.ps; name = sys.name, initial_conditions = sys.initial_conditions
    )
    return sys
end

function PDEBase.should_transform(pdesys::PDESystem, disc::MOLFiniteDifference, boundarymap)
    if !disc.should_transform
        return false
    end
    if has_interfaces(boundarymap) && !(disc.advection_scheme isa HermiteScheme)
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
            return filter_differentials(SU.arguments(term)[1], differential, v, depth + op.order)
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
                        arguments(term)[1], op, v, 1
                    )
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
                    op.domain.domain.left, v.intervals[op.domain.variables][1]
                ) &&
                    isequal(op.domain.domain.right, Num(op.domain.variables))
                whole = isequal(
                    op.domain.domain.left, v.intervals[op.domain.variables][1]
                ) &&
                    isequal(op.domain.domain.right, v.intervals[op.domain.variables][2])
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
    hermite_transform!(eqs, bcs, boundarymap, v, disc)

Automatically introduce auxiliary derivative variables for `HermiteScheme`.

For each dependent variable `u` and each spatial independent variable `x`:
  1. Create a new variable `v` representing `Dx(u)` via `diff2term`.
  2. Replace every occurrence of `Dx(u)` in the existing equations and BCs with `v`.
  3. Add a new time-evolution equation for `v` obtained by applying `Dx` to the
     original PDE for `u` and expanding.
  4. Derive initial conditions and spatial boundary conditions for `v`. Explicit
     first-derivative conditions are solved directly. Dirichlet traces provide
     tangential derivatives by differentiation and normal derivatives when the
     transformed evolution equation can be solved for them.

The mapping is recorded in `disc.advection_scheme.deriv_var`.  If the user already
supplied explicit mappings, this step is skipped.
"""
function hermite_transform!(eqs, bcs, boundarymap, v, disc)
    scheme = disc.advection_scheme
    scheme.automatic || return
    empty!(scheme.deriv_var)

    t = v.time
    original_depvar_ops = copy(v.depvar_ops)

    for uop in original_depvar_ops
        uargs = v.args[uop]
        u_applied = uop(uargs...)
        spatial_ivs = filter(!isequal(t), uargs)

        for x in spatial_ivs
            Dx = Differential(x)
            dx_u = Dx(u_applied)

            # Create the new symbolic variable vˍx(t, x, ...) from Dx(u(t, x, ...)).
            newvar = diff2term(Symbolics.value(dx_u))
            newop = operation(newvar)

            # Register the new variable in the VariableMap.
            update_varmap!(v, newvar)

            # Find the original PDE for u: the equation whose LHS contains Dt(u).
            Dt = Differential(t)
            u_eq_idx = findfirst(eqs) do eq
                dv = get_depvars(eq.lhs, v.depvar_ops)
                any(d -> isequal(operation(d), uop), dv)
            end
            u_eq_idx === nothing && error(
                "HermiteScheme: could not find a time-evolution equation for $uop."
            )
            # PDEBase cardinalizes equations before this transformation, so the
            # equation is `lhs - rhs ~ 0` and its stored RHS is zero. Recover
            # the actual evolution RHS by solving for the time derivative.
            dt_u = Dt(u_applied)
            original_rhs = symbolic_linear_solve(eqs[u_eq_idx], dt_u)

            # Replacement rule: Dx(u(t,x,...)) => v(t,x,...) throughout the system.
            rule = Symbolics.value(dx_u) => newvar
            subs_alleqs!(eqs, bcs, rule)

            # Keep the primary equation in explicit evolution form. Besides
            # making the coupled system clear, this RHS is used below to infer
            # the derivative partner's Dirichlet boundary values.
            transformed_rhs = substitute(original_rhs, rule)
            eqs[u_eq_idx] = dt_u ~ transformed_rhs

            # New evolution equation for v: v_t = Dx(original_rhs), expanded.
            v_rhs = expand_derivatives(Dx(original_rhs))
            v_rhs = pde_substitute(v_rhs, Dict(rule))
            push!(eqs, Dt(newvar) ~ v_rhs)

            # Derive ICs and spatial BCs for the derivative partner. Initial
            # conditions and tangential Dirichlet data are differentiated;
            # normal Dirichlet data are inferred from the evolution equation.
            bcivmap = reduce(
                (d1, d2) -> mergewith(vcat, d1, d2), collect(values(boundarymap))
            )

            function rulesforeachboundary(iv, upper)
                cond(b) = upper ? b isa UpperBoundary : b isa LowerBoundary
                return generate_bc_rules(filter(cond, get(bcivmap, iv, [])), v)
            end

            newbcs = []
            for iv in all_ivs(v)
                boundaries = get(get(boundarymap, uop, Dict()), iv, [])
                isempty(boundaries) && continue

                if isequal(iv, t)
                    # The time boundary is the initial condition. Existing
                    # auxiliary-BC machinery differentiates it with respect
                    # to the spatial variable correctly.
                    generate_aux_bcs!(
                        newbcs, newvar, Symbolics.value(dx_u),
                        filter(isupper, boundaries), v,
                        rulesforeachboundary(iv, true)
                    )
                    generate_aux_bcs!(
                        newbcs, newvar, Symbolics.value(dx_u),
                        filter(!isupper, boundaries), v,
                        rulesforeachboundary(iv, false)
                    )
                else
                    for bc in boundaries
                        xval = isupper(bc) ? v.intervals[iv][2] : v.intervals[iv][1]
                        newargs_sub = substitute.(arguments(newvar), (iv => xval,))
                        bcdv = newop(newargs_sub...)

                        if bc isa InterfaceBoundary
                            # Periodic: if u(t,a) ~ u(t,b), register both
                            # orientations of the corresponding partner seam.
                            other_bcdv = newop(arguments(bc.u2)...)
                            eq = bcdv ~ other_bcdv
                            push!(
                                newbcs,
                                typeof(bc)(bcdv, other_bcdv, bc.x, bc.x2, eq)
                            )
                            continue
                        end

                        # Boundary-evaluated derivatives do not match the
                        # global Dx(u(t,x)) rule. Classify Neumann/Robin data by
                        # whether the original equation can be solved for the
                        # boundary-normal first derivative.
                        v_at_boundary = bcdv
                        bclhs = bcdv
                        new_order = 0

                        normal_derivative = Differential(iv)(bc.u)
                        normal_rhs = symbolic_linear_solve(bc.eq, normal_derivative)
                        constrains_normal_derivative =
                            !isequal(Symbolics.value(normal_rhs), NaN)

                        if constrains_normal_derivative
                            if isequal(iv, x)
                                # Normal partner: Neumann and Robin conditions
                                # directly determine the normal derivative.
                                rhs = normal_rhs
                            else
                                # Tangential partner: commute derivatives and
                                # differentiate the solved normal relation along
                                # this partner's direction.
                                bclhs = Differential(iv)(bcdv)
                                tangential_derivative = Dx(bc.u)
                                rhs = expand_derivatives(Dx(normal_rhs))
                                rhs = pde_substitute(
                                    rhs, Dict(tangential_derivative => bcdv)
                                )
                                new_order = 1
                            end
                        else
                            # Dirichlet: tangential derivatives are obtained by
                            # differentiating the known boundary trace directly.
                            # For the normal derivative, use the transformed PDE
                            # and solve after substituting the trace beneath any
                            # remaining differential operators.
                            bc_val = symbolic_linear_solve(bc.eq, bc.u)
                            if !isequal(iv, x)
                                rhs = expand_derivatives(Dx(bc_val))
                            else
                                Dt_local = Differential(t)
                                u_t_boundary = expand_derivatives(Dt_local(bc_val))

                                directional_orders = d_orders(
                                    x, [dt_u ~ original_rhs]
                                )
                                isempty(directional_orders) && throw(ArgumentError(
                                    "HermiteScheme cannot infer a normal derivative " *
                                        "condition for $newvar from $(bc.eq): the " *
                                        "evolution equation contains no $x derivative of $u_applied."
                                ))
                                partner_term(order) = order == 1 ? newvar :
                                    (Dx^(order - 1))(newvar)
                                boundary_partner_term(order) = order == 1 ?
                                    v_at_boundary : (Dx^(order - 1))(v_at_boundary)
                                partner_rules = Dict(
                                    (Dx^order)(u_applied) => partner_term(order)
                                        for order in directional_orders
                                )
                                rhs_at_boundary = pde_substitute(
                                    original_rhs, partner_rules
                                )
                                rhs_at_boundary = expand_derivatives(pde_substitute(
                                    rhs_at_boundary, Dict(u_applied => bc_val)
                                ))
                                boundary_partner_rules = Dict(
                                    partner_term(order) => boundary_partner_term(order)
                                        for order in directional_orders
                                )
                                rhs_at_boundary = pde_substitute(
                                    rhs_at_boundary, boundary_partner_rules
                                )
                                rhs_at_boundary = substitute(
                                    rhs_at_boundary, iv => xval
                                )
                                boundary_eq = u_t_boundary ~ rhs_at_boundary

                                solved = false
                                for order in reverse(0:(maximum(directional_orders) - 1))
                                    target = order == 0 ? v_at_boundary :
                                        (Dx^order)(v_at_boundary)
                                    candidate = symbolic_linear_solve(boundary_eq, target)
                                    isequal(Symbolics.value(candidate), NaN) && continue
                                    bclhs = target
                                    rhs = candidate
                                    new_order = order
                                    solved = true
                                    break
                                end
                                solved || throw(ArgumentError(
                                    "HermiteScheme could not infer a normal derivative " *
                                        "condition for $newvar from $(bc.eq) and " *
                                        "evolution equation $(eqs[u_eq_idx])."
                                ))
                            end
                        end

                        eq = bclhs ~ rhs
                        newbc = isupper(bc) ?
                            UpperBoundary(bcdv, t, iv, new_order, eq, v) :
                            LowerBoundary(bcdv, t, iv, new_order, eq, v)
                        push!(newbcs, newbc)
                    end
                end
            end
            newbcs = unique(newbcs)
            independent_bcs = filter(newbcs) do bc
                !(bc isa InterfaceBoundary) || !isupper(bc)
            end
            append!(bcs, map(bc -> bc.eq, independent_bcs))
            update_boundarymap!(boundarymap, newbcs, newop, v)

            # Record the mapping for the Hermite discretization path.
            if length(spatial_ivs) == 1
                scheme.deriv_var[uop] = newop
            else
                scheme.deriv_var[(uop, x)] = newop
            end
        end
    end
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
            @assert length(interfaces) == 0 "Interface BCs like $(interfaces[1].eq) are not yet supported in conjunction with system transformation, please transform manually if needed and set `should_transform=false` in the discretization. If you need this feature, please open an issue on GitHub."

            boundaries = boundarymap[dv][iv]
            length(bcs) == 0 && continue

            generate_aux_bcs!(
                newbcs, newvar, term, filter(isupper, boundaries),
                v, rulesforeachboundary(iv, true)
            )
            generate_aux_bcs!(
                newbcs, newvar, term, filter(!isupper, boundaries),
                v, rulesforeachboundary(iv, false)
            )
        end
    end
    newbcs = unique(newbcs)
    # add the new bc equations
    append!(bcs, map(bc -> bc.eq, newbcs))
    # Add the new boundary conditions and initial conditions to the boundarymap
    return update_boundarymap!(boundarymap, newbcs, newop, v)
    # update pmap
end

function generate_bc_rules(bcs, v)
    bcs = reverse(sort(bcs, by = bc -> bc.order))
    return map(bcs) do bc
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
    return
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
    bcrhs = expand_derivatives(pde_substitute(deriv(term), Dict(rules)))
    eq = bclhs ~ bcrhs

    newbc = if isupper(bc)
        UpperBoundary(bcdv, t, x, bc.order, eq, v)
    else
        LowerBoundary(bcdv, t, x, bc.order, eq, v)
    end
    return push!(newbcs, newbc)
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
    return
end
