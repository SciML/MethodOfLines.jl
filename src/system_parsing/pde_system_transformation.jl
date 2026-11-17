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
        # The automatic HermiteScheme finds its derivative partners during the
        # transformation; without it every derivative would silently fall back to
        # centered differences.
        scheme = disc.advection_scheme
        scheme isa HermiteScheme && scheme.automatic && throw(
            ArgumentError(
                "`HermiteScheme()` introduces its derivative partners by transforming " *
                    "the PDE system, which `should_transform = false` disables. Set " *
                    "`should_transform = true`, or give the partners explicitly with " *
                    "`HermiteScheme(deriv_var = u => v)`."
            )
        )
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
    _x_derivative_orders!(acc, ex, x)

Push into `acc` the total order in `x` of every derivative chain in `ex`; `Dx(Dy(Dx(u)))`
counts 2, and a chain without an `x` derivative adds nothing.
"""
function _x_derivative_orders!(acc, ex, x)
    ex = Symbolics.value(ex)
    (ex isa SymbolicUtils.BasicSymbolic && iscall(ex)) || return acc
    if operation(ex) isa Differential
        n = 0
        e = ex
        while e isa SymbolicUtils.BasicSymbolic && iscall(e) && operation(e) isa Differential
            op = operation(e)
            isequal(op.x, x) && (n += op.order)
            e = arguments(e)[1]
        end
        n > 0 && push!(acc, n)
        return _x_derivative_orders!(acc, e, x)
    end
    foreach(a -> _x_derivative_orders!(acc, a, x), arguments(ex))
    return acc
end

# Dependent-variable terms of `ex`: subexpression => (op, order in x, other differentials
# applied, outermost first), or `nothing` for a term this cannot evaluate on a wall.
function _hermite_dependent_terms!(acc, ex, x, ops)
    ex = Symbolics.value(ex)
    (ex isa SymbolicUtils.BasicSymbolic && iscall(ex)) || return acc
    op = operation(ex)
    if op isa Differential
        n = 0
        others = Differential[]
        e = ex
        while e isa SymbolicUtils.BasicSymbolic && iscall(e) && operation(e) isa Differential
            d = operation(e)
            isequal(d.x, x) ? (n += d.order) : push!(others, d)
            e = arguments(e)[1]
        end
        if e isa SymbolicUtils.BasicSymbolic && iscall(e) && any(o -> isequal(operation(e), o), ops)
            acc[ex] = (operation(e), n, others)
            return acc
        end
        acc[ex] = nothing                  # derivative of a compound expression
        return acc
    end
    if any(o -> isequal(op, o), ops)
        acc[ex] = (op, 0, Differential[])
        return acc
    end
    foreach(a -> _hermite_dependent_terms!(acc, a, x, ops), arguments(ex))
    return acc
end

# Dirichlet data of `q` on the wall (`upper` end of `x`) that contain no dependent
# variable, or `nothing`.
function _hermite_wall_dirichlet_data(q, x, upper, boundarymap, ops)
    for b in get(get(boundarymap, q, Dict()), x, [])
        b isa InterfaceBoundary && continue
        (upper ? b isa UpperBoundary : b isa LowerBoundary) || continue
        b.order == 0 || continue
        g = symbolic_linear_solve(b.eq, b.u)
        isequal(Symbolics.value(g), NaN) && continue
        isempty(_hermite_dependent_terms!(Dict{Any, Any}(), g, x, ops)) || continue
        return g
    end
    return nothing
end

"""
    _hermite_wall_normal_derivative(q, k, t, x, xval, upper, R, boundarymap, depth)

`Dx^k(q)` on the wall `x = xval`, as an expression in the remaining variables, or `nothing`.

It uses the Dirichlet data on that wall and the evolution equations `Dt(r) ~ R[r]` as given.
On the wall `Dt(Dx^j(r)) = Dx^j(R[r])`; where `Dx^j(r)` is known there, this is an equation
for the one unknown wall derivative it contains, once every other term is known in turn
(recursively, at most `depth` levels; derivatives along the wall and in time of a known
wall value are taken directly).

For the wave equation as a system, `Dt(u) ~ p` and `Dt(p) ~ Dxx(u)`, with `u` given on the
wall: `p_wall = Dt(u_wall)`, then `u_xx = Dt(p_wall)` from `p`'s equation. `u_x` and `p_x`
are not determined (the wave reflects), so they are evolved.
"""
function _hermite_wall_normal_derivative(q, k, t, x, xval, upper, R, boundarymap, depth)
    ops = collect(keys(R))
    if k == 0
        g = _hermite_wall_dirichlet_data(q, x, upper, boundarymap, ops)
        g === nothing || return g
    end
    depth > 0 || return nothing
    for (r, Rr) in R, j in 0:k
        (isequal(r, q) && j == k) && continue
        hr = _hermite_wall_normal_derivative(r, j, t, x, xval, upper, R, boundarymap, depth - 1)
        hr === nothing && continue
        ex = j == 0 ? Symbolics.value(Rr) : expand_derivatives((Differential(x)^j)(Rr))
        terms = _hermite_dependent_terms!(Dict{Any, Any}(), ex, x, ops)
        any(isnothing, values(terms)) && continue
        targets = [te for (te, (op, n, ds)) in terms if isequal(op, q) && n == k && isempty(ds)]
        length(targets) == 1 || continue
        target = only(targets)
        z = Symbolics.unwrap(only(Symbolics.@variables __hermite_wall_unknown))
        rules = Dict{Any, Any}(target => z)
        ok = true
        for (te, (op, n, ds)) in terms
            isequal(te, target) && continue
            h = _hermite_wall_normal_derivative(op, n, t, x, xval, upper, R, boundarymap, depth - 1)
            h === nothing && (ok = false; break)
            for d in reverse(ds)
                h = expand_derivatives(d(h))
            end
            rules[te] = h
        end
        ok || continue
        rhs = substitute(substitute(ex, rules), Dict(x => xval))
        lhs = expand_derivatives(Differential(t)(hr))
        sol = symbolic_linear_solve(lhs ~ rhs, z)
        isequal(Symbolics.value(sol), NaN) && continue
        return sol
    end
    return nothing
end

"""
    _hermite_wall_partner_condition(uop, t, x, xval, upper, v_at_boundary, Dx, R, boundarymap)

Condition for the normal partner `v = Dx(u)` on a Dirichlet wall when `u`'s own equation
gives none: `v ~ u_x` if the wall determines `u_x`, else `Dx(v) ~ u_xx`, both from
[`_hermite_wall_normal_derivative`](@ref). Returns `(lhs, rhs, order)` or `nothing`.
"""
function _hermite_wall_partner_condition(uop, t, x, xval, upper, v_at_boundary, Dx, R, boundarymap)
    for k in 1:2
        h = _hermite_wall_normal_derivative(uop, k, t, x, xval, upper, R, boundarymap, 3)
        h === nothing && continue
        return k == 1 ? (v_at_boundary, h, 0) : ((Dx^1)(v_at_boundary), h, 1)
    end
    return nothing
end

"""
    _hermite_reduce_second_order_in_time!(eqs, bcs, boundarymap, v)

Rewrite every `Dt(Dt(u)) ~ R` as the first-order system `Dt(u) ~ uˍt`, `Dt(uˍt) ~ R` (with
`Dt(u)` in `R` replaced by `uˍt`), so that [`hermite_transform!`](@ref) treats it as a
system written out by the user.

`uˍt` gets:
  - the initial condition from `Dt(u(t0, x)) ~ g`, which is removed from `u`'s conditions;
  - on each spatial boundary, the time derivative of `u`'s condition when that condition is
    data (Dirichlet `u ~ g` gives `uˍt ~ Dt(g)`, Neumann `Dx(u) ~ h` gives `Dx(uˍt) ~ Dt(h)`);
  - the periodic conditions of `u`.

A missing initial condition for `Dt(u)`, or another kind of boundary condition (Robin, or
one with a time derivative), throws an `ArgumentError` that asks for the first-order system.
"""
function _hermite_reduce_second_order_in_time!(eqs, bcs, boundarymap, v)
    t = v.time
    Dt = Differential(t)
    for uop in copy(v.depvar_ops)
        u = uop(v.args[uop]...)
        dt2 = Symbolics.value(Dt(Dt(u)))
        dt1 = Symbolics.value(Dt(u))
        has(ex, term) = Symbolics.hasnode(term, Symbolics.value(ex.lhs - ex.rhs))
        idx2 = findall(eq -> has(eq, dt2), eqs)
        isempty(idx2) && continue
        any(eq -> has(eq, dt1), eqs) && continue    # mixed orders: left to the checks below
        length(idx2) == 1 || throw(
            ArgumentError(
                "HermiteScheme needs exactly one equation with $(Dt(Dt(u))), found " *
                    "$(length(idx2)): $(eqs[idx2])."
            )
        )
        # the initial condition for Dt(u)
        tbcs = get(get(boundarymap, uop, Dict()), t, [])
        ic1 = filter(b -> !(b isa InterfaceBoundary) && b.order == 1, tbcs)
        length(ic1) == 1 || throw(
            ArgumentError(
                "HermiteScheme needs an initial condition for $(Dt(u)) to rewrite " *
                    "$(eqs[only(idx2)]) as a first-order system in time."
            )
        )
        ic1 = only(ic1)
        g = symbolic_linear_solve(ic1.eq, Dt(ic1.u))
        isequal(Symbolics.value(g), NaN) && throw(
            ArgumentError("HermiteScheme could not solve $(ic1.eq) for $(Dt(ic1.u)).")
        )

        newvar = diff2term(dt1)
        newop = operation(newvar)
        update_varmap!(v, newvar)
        subs_alleqs!(eqs, bcs, dt2 => Symbolics.value(Dt(newvar)))
        subs_alleqs!(eqs, bcs, dt1 => newvar)
        push!(eqs, Dt(u) ~ newvar)

        filter!(b -> b !== ic1, boundarymap[uop][t])
        filter!(eq -> !isequal(eq, ic1.eq), bcs)

        newbcs = []
        t0 = v.intervals[t][1]
        bcdv = newop(substitute.(arguments(newvar), (t => t0,))...)
        push!(newbcs, LowerBoundary(bcdv, t, t, 0, bcdv ~ g, v))
        for x in filter(!isequal(t), v.args[uop])
            for bc in get(boundarymap[uop], x, [])
                xval = isupper(bc) ? v.intervals[x][2] : v.intervals[x][1]
                bcdv = newop(substitute.(arguments(newvar), (x => xval,))...)
                if bc isa InterfaceBoundary
                    other = newop(arguments(bc.u2)...)
                    push!(newbcs, typeof(bc)(bcdv, other, bc.x, bc.x2, bcdv ~ other))
                    continue
                end
                deriv = bc.order == 0 ? identity : (Differential(x)^bc.order)
                h = symbolic_linear_solve(bc.eq, deriv(bc.u))
                isdata = !isequal(Symbolics.value(h), NaN) &&
                    isempty(_hermite_dependent_terms!(Dict{Any, Any}(), h, x, v.depvar_ops))
                isdata || throw(
                    ArgumentError(
                        "HermiteScheme can rewrite $(Dt(Dt(u))) as a first-order system " *
                            "only with Dirichlet, Neumann or periodic conditions on $u; " *
                            "got $(bc.eq). Write the system out instead, e.g. " *
                            "`Dt(u) ~ p` and `Dt(p) ~ ...`, with conditions on `p`."
                    )
                )
                eq = deriv(bcdv) ~ expand_derivatives(Dt(h))
                push!(
                    newbcs, isupper(bc) ? UpperBoundary(bcdv, t, x, bc.order, eq, v) :
                        LowerBoundary(bcdv, t, x, bc.order, eq, v)
                )
            end
        end
        append!(
            bcs, [b.eq for b in newbcs if !(b isa InterfaceBoundary) || !isupper(b)]
        )
        update_boundarymap!(boundarymap, newbcs, newop, v)
    end
    return
end

"""
    hermite_transform!(eqs, bcs, boundarymap, v, disc)

Introduce the derivative partners of an automatic `HermiteScheme`: for each dependent
variable `u` and spatial variable `x`, a variable `uˍx` standing for `Dx(u)`, with its own
evolution equation, initial condition and boundary conditions.

The steps, in order:
  1. Rewrite `Dt(Dt(u)) ~ R` as a first-order system in time
     ([`_hermite_reduce_second_order_in_time!`](@ref)).
  2. Record the directions in which a variable's PDE is diffusive (only even-order `x`
     derivatives), whose boundary stencils are wider (`scheme.diffusive`).
  3. For each `u` and `x`: replace `Dx(u)` by `uˍx` in the equations and conditions, and add
     `Dt(uˍx) ~ Dx(R)` from `u`'s equation `Dt(u) ~ R`.
  4. Give `uˍx` the initial condition `Dx` of `u`'s, and boundary conditions from `u`'s:
     at a boundary of `x`, Neumann or Robin data on `u` become a condition on `uˍx`, which
     replaces the one on `u`; Dirichlet data give `Dx(uˍx) ~ u_xx` from `u`'s equation
     solved on the wall, or, where that equation has no `x` derivative, from the other
     equations ([`_hermite_wall_partner_condition`](@ref)). At a boundary of another
     direction `y`, Dirichlet data give `uˍx` by differentiation along the boundary; with
     Neumann or Robin data `uˍx` has no condition and is evolved by its own equation.
     Periodic conditions carry over.

The mapping `u => uˍx` is recorded in `disc.advection_scheme.deriv_var`. Nothing is done
for a scheme with explicit partners (`HermiteScheme(deriv_var = ...)`).
"""
function hermite_transform!(eqs, bcs, boundarymap, v, disc)
    scheme = disc.advection_scheme
    scheme.automatic || return
    empty!(scheme.deriv_var)
    empty!(scheme.diffusive)

    t = v.time
    # Dt(Dt(u)) ~ R becomes Dt(u) ~ uˍt, Dt(uˍt) ~ R (before anything reads the equations)
    _hermite_reduce_second_order_in_time!(eqs, bcs, boundarymap, v)
    original_depvar_ops = copy(v.depvar_ops)

    # Directions in which a variable's own PDE is diffusive: only even-order derivatives in
    # x appear (heat, not advection or advection-diffusion). Decided on the equations as
    # given, before the partners rewrite them (advection's partner equation reads u_xx).
    for uop in original_depvar_ops
        uargs = v.args[uop]
        dt_u = Symbolics.value(Differential(t)(uop(uargs...)))
        idxs = findall(eq -> Symbolics.hasnode(dt_u, Symbolics.value(eq.lhs - eq.rhs)), eqs)
        length(idxs) == 1 || continue
        eq = eqs[only(idxs)]
        for x in filter(!isequal(t), uargs)
            orders = _x_derivative_orders!(Set{Int}(), eq.lhs - eq.rhs, x)
            !isempty(orders) && all(iseven, orders) && push!(scheme.diffusive, (uop, x))
        end
    end
    # Evolution right-hand sides as given, for wall conditions that a variable's own
    # equation cannot provide (`_hermite_wall_partner_condition`).
    evol_rhs = Dict{Any, Any}()
    for uop in original_depvar_ops
        dt_u = Symbolics.value(Differential(t)(uop(v.args[uop]...)))
        idxs = findall(eq -> Symbolics.hasnode(dt_u, Symbolics.value(eq.lhs - eq.rhs)), eqs)
        length(idxs) == 1 || continue
        r = symbolic_linear_solve(eqs[only(idxs)], dt_u)
        isequal(Symbolics.value(r), NaN) || (evol_rhs[uop] = r)
    end
    # Neumann/Robin conditions on u that a normal partner condition replaces.
    replaced_bcs = Pair[]

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

            # Find the original PDE for u: the one equation containing Dt(u). Other
            # equations may mention u (coupled systems), so containing u is not enough.
            Dt = Differential(t)
            dt_u = Dt(u_applied)
            u_eq_idxs = findall(eq -> Symbolics.hasnode(Symbolics.value(dt_u), Symbolics.value(eq.lhs - eq.rhs)), eqs)
            isempty(u_eq_idxs) && throw(
                ArgumentError(
                    "HermiteScheme needs an equation with the first time derivative " *
                        "$dt_u to transform $u_applied; none of the equations contains it. " *
                        "Time derivatives of third or higher order must be written as a " *
                        "system with first or second time derivatives, e.g. `Dt(u) ~ p` and " *
                        "`Dt(Dt(p)) ~ ...` for `Dt(Dt(Dt(u))) ~ ...`."
                )
            )
            length(u_eq_idxs) == 1 || throw(
                ArgumentError(
                    "HermiteScheme needs exactly one equation with $dt_u to transform " *
                        "$u_applied, found $(length(u_eq_idxs)): $(eqs[u_eq_idxs])."
                )
            )
            u_eq_idx = only(u_eq_idxs)
            # PDEBase cardinalizes equations before this transformation, so the
            # equation is `lhs - rhs ~ 0` and its stored RHS is zero. Recover
            # the actual evolution RHS by solving for the time derivative.
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
                                # directly determine the normal derivative. The
                                # partner condition holds the data exactly, so the
                                # condition on u is dropped below and the PDE
                                # evolves u at this boundary with Hermite weights.
                                rhs = normal_rhs
                                push!(replaced_bcs, uop => bc)
                            else
                                # Tangential partner (e.g. uˍy at an x boundary): no
                                # condition. A derived one, Dx(uˍy) ~ Dy(normal data),
                                # would use MOL's second-order one-sided stencil and
                                # limit the order of the scheme. The partner's own
                                # equation evolves it here instead, as u is evolved above.
                                continue
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
                                rhs = nothing
                                solved = false
                                if !isempty(directional_orders)
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
                                    rhs_at_boundary = expand_derivatives(
                                        pde_substitute(
                                            rhs_at_boundary, Dict(u_applied => bc_val)
                                        )
                                    )
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
                                end
                                if !solved
                                    # u's own equation gives no condition (e.g. Dt(u) ~ p
                                    # of the wave equation as a system): use the other
                                    # equations and the wall data.
                                    fb = _hermite_wall_partner_condition(
                                        uop, t, x, xval, isupper(bc), v_at_boundary, Dx,
                                        evol_rhs, boundarymap
                                    )
                                    fb === nothing && throw(
                                        ArgumentError(
                                            isempty(directional_orders) ?
                                                "HermiteScheme cannot infer a normal derivative " *
                                                "condition for $newvar from $(bc.eq): the " *
                                                "evolution equation contains no $x derivative of " *
                                                "$u_applied, and the other equations do not " *
                                                "determine one on this boundary." :
                                                "HermiteScheme could not infer a normal derivative " *
                                                "condition for $newvar from $(bc.eq) and " *
                                                "evolution equation $(eqs[u_eq_idx])."
                                        )
                                    )
                                    bclhs, rhs, new_order = fb
                                end
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

    # Remove only after every direction is done: tangential partners in other
    # directions are derived from these conditions.
    for (uop, bc) in unique(replaced_bcs)
        filter!(b -> b !== bc, boundarymap[uop][bc.x])
        filter!(eq -> !isequal(eq, bc.eq), bcs)
    end
    return
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
