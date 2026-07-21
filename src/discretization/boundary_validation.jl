using ModelingToolkit: operation, arguments, iscall, Differential
using Symbolics: unwrap

"""
    validate_system_wellposedness(pdes, bmap, s, _disc)

Validate the well-posedness of the PDE system before discretization. For every
dependent variable and spatial dimension, the highest spatial derivative order
sets the required number of boundary conditions.

For order ≥ 2, boundary conditions are counted by effective constraint weight:
each single-point equation contributes 1; a coupled or periodic equation that
evaluates `u` at two endpoints contributes 2. Several independent conditions
may share the same spatial point (e.g. clamped ends `u = 0` and `u_x = 0` at
`x = 0`).

A first-order spatial derivative whose coefficient is not a compile-time numeric
constant (e.g. Burgers' `-u * u_x`, parameterized `v * u_x`, or spatial `sin(x) * u_x`)
is treated as needing data at *both* ends of the domain, since the upwind direction
cannot be fixed before the solve.
"""
function validate_system_wellposedness(pdes, bmap, s, _disc)
    for u_call in s.ū
        u_op = operation(unwrap(u_call))
        spatial_ivs = s.x̄

        for x in spatial_ivs
            max_order = 0
            has_dynamic_advection = false

            for eq in pdes
                max_order = max(
                    max_order,
                    get_max_derivative_order(eq.lhs, u_op, x),
                    get_max_derivative_order(eq.rhs, u_op, x)
                )
            end

            if max_order == 1
                for eq in pdes
                    if is_dynamic_advection(eq.lhs, u_op, x) ||
                            is_dynamic_advection(eq.rhs, u_op, x)
                        has_dynamic_advection = true
                        break
                    end
                end
            end

            if max_order > 0
                u_bcs = get(bmap, u_op, nothing)
                u_bcs_x = u_bcs !== nothing ? get(u_bcs, x, nothing) : nothing
                provided_bcs = u_bcs_x !== nothing ? length(u_bcs_x) : 0

                if max_order >= 2
                    effective_bcs = count_effective_boundary_conditions(
                        u_bcs_x, u_call, u_op, x
                    )
                    if effective_bcs < max_order
                        throw(
                            ArgumentError(
                                "Ill-posed PDE: $(u_op) in $(x) has order-$(max_order) " *
                                    "spatial derivatives but only $(effective_bcs) effective " *
                                    "boundary constraint(s) provided; need $(max_order)."
                            )
                        )
                    end
                elseif max_order == 1 && has_dynamic_advection
                    provided_locations = count_boundary_locations(
                        u_bcs_x, u_call, u_op, x
                    )
                    if provided_locations < 2
                        throw(
                            ArgumentError(
                                "Ill-posed PDE: $(u_op) in $(x) has a non-constant advection " *
                                    "coefficient, so the upwind direction is not fixed and boundary " *
                                    "data is required at BOTH ends; only $(provided_locations) provided."
                            )
                        )
                    end
                elseif max_order == 1 && !has_dynamic_advection && provided_bcs < 1
                    throw(
                        ArgumentError(
                            "Ill-posed PDE: $(u_op) in $(x) has a 1st-order spatial " *
                                "derivative but no boundary condition was provided."
                        )
                    )
                end
            end
        end
    end
    return
end

"""
    _get_equation_locations(u_bcs_x, u_call, u_op, x)

Return a vector of pairs mapping each boundary condition equation to a `Set` of
its unique spatial locations for the dimension `x`. Internal helper function.
"""
function _get_equation_locations(u_bcs_x, u_call, u_op, x)
    u_bcs_x === nothing && return ()
    x_index = findfirst(isequal(x), arguments(unwrap(u_call)))
    x_index === nothing && return ()

    eq_locs = Pair{Any, Set{Any}}[]
    for bc_obj in u_bcs_x
        bc_eq = boundary_condition_equation(bc_obj)
        unique_bounds = Set()
        if hasproperty(bc_eq, :lhs) && hasproperty(bc_eq, :rhs)
            extract_boundary_locations!(unique_bounds, bc_eq.lhs, u_op, x_index, x)
            extract_boundary_locations!(unique_bounds, bc_eq.rhs, u_op, x_index, x)
        end
        push!(eq_locs, bc_eq => unique_bounds)
    end
    return eq_locs
end

"""
    count_boundary_locations(u_bcs_x, u_call, u_op, x)

Return the number of unique spatial locations at which `u_op` appears in the
boundary conditions for dimension `x`. Numeric locations are normalised to
`Float64` so that e.g. `0` and `0.0` hash to the same key.
"""
function count_boundary_locations(u_bcs_x, u_call, u_op, x)
    eq_locs = _get_equation_locations(u_bcs_x, u_call, u_op, x)
    eq_locs === () && return 0

    unique_bounds = Set()
    for (_, locs) in eq_locs
        union!(unique_bounds, locs)
    end
    return length(unique_bounds)
end

"""
    boundary_condition_equation(bc_obj)

Return the underlying equation for a boundary condition object.
"""
boundary_condition_equation(bc_obj) =
    hasproperty(bc_obj, :eq) ? bc_obj.eq : bc_obj

"""
    count_effective_boundary_conditions(u_bcs_x, u_call, u_op, x)

Return the effective number of boundary constraints for dimension `x`. Each
single-point equation contributes 1; a coupled or periodic equation that
evaluates `u_op` at two or more unique spatial locations contributes 2.

Periodic conditions appear twice in the parsed boundary map (once per endpoint);
they are deduplicated by structural equality of the underlying equation before
counting. Every counted equation contributes at least 1 constraint.
"""
function count_effective_boundary_conditions(u_bcs_x, u_call, u_op, x)
    eq_locs = _get_equation_locations(u_bcs_x, u_call, u_op, x)
    eq_locs === () && return 0

    function structurally_equal(a, b)
        if hasproperty(a, :lhs) && hasproperty(a, :rhs) &&
                hasproperty(b, :lhs) && hasproperty(b, :rhs)
            return isequal(a.lhs, b.lhs) && isequal(a.rhs, b.rhs)
        end
        return isequal(a, b)
    end

    seen_eqs = Any[]
    total = 0
    for (bc_eq, unique_bounds) in eq_locs
        if any(seen -> structurally_equal(seen, bc_eq), seen_eqs)
            continue
        end
        push!(seen_eqs, bc_eq)
        total += length(unique_bounds) >= 2 ? 2 : 1
    end
    return total
end

"""
    extract_boundary_locations!(unique_bounds, expr, u_op, x_index, x)

Walk the AST of a boundary condition and push the spatial location of every
`u_op(..., x_loc, ...)` occurrence into `unique_bounds`.
"""
function extract_boundary_locations!(unique_bounds::Set, expr, u_op, x_index, x)
    expr = unwrap(expr)
    if iscall(expr)
        op = operation(expr)

        if isequal(op, u_op)
            args = arguments(expr)
            if x_index <= length(args)
                raw_val = unwrap(args[x_index])
                normalized_val = raw_val isa Number ? Float64(raw_val) : raw_val
                push!(unique_bounds, normalized_val)
            else
                throw(
                    ArgumentError(
                        "Malformed boundary condition: $(u_op) needs at least $(x_index) " *
                            "arguments for dimension $(x), got $(length(args))."
                    )
                )
            end
            return
        end

        for arg in arguments(expr)
            extract_boundary_locations!(unique_bounds, arg, u_op, x_index, x)
        end
    end
    return
end

"""
    get_max_derivative_order(expr, u_op, x)

Return the highest order of a spatial derivative of `u_op` with respect to `x`
appearing anywhere in `expr`, or `-1` if `u_op` does not appear at all.
"""
function get_max_derivative_order(expr, u_op, x)
    expr = unwrap(expr)
    if iscall(expr)
        op = operation(expr)

        if isequal(op, u_op)
            return 0
        end

        inner_max = -1
        for arg in arguments(expr)
            m = get_max_derivative_order(arg, u_op, x)
            if m > inner_max
                inner_max = m
            end
        end

        if op isa Differential && isequal(op.x, x) && inner_max >= 0
            return op.order + inner_max
        end

        return inner_max
    end
    return isequal(expr, u_op) ? 0 : -1
end

"""
    is_static_numeric_constant(expr)

Return `true` when `expr` is a numeric literal, or a product/quotient of numeric
literals only.
"""
function is_static_numeric_constant(expr)
    expr = unwrap(expr)
    if expr isa Number
        return true
    end
    val = Symbolics.value(expr)
    if val isa Number
        return true
    end
    if iscall(expr)
        op = operation(expr)
        if isequal(op, *) || isequal(op, /)
            return all(is_static_numeric_constant, arguments(expr))
        end
    end
    return false
end

"""
    is_dynamic_advection(expr, u_op, x)

Return `true` when `expr` contains a 1st-order spatial derivative of `u_op`
multiplied (or divided) by a coefficient that is not a compile-time numeric
constant. Parameters, states, and independent variables all make the upwind
direction unknown at discretization time.
"""
function is_dynamic_advection(expr, u_op, x)
    expr = unwrap(expr)
    if iscall(expr)
        op = operation(expr)

        if isequal(op, *) || isequal(op, /)
            args = arguments(expr)
            has_deriv = false
            for arg in args
                arg_unwrapped = unwrap(arg)
                arg_op = iscall(arg_unwrapped) ? operation(arg_unwrapped) : nothing
                if arg_op isa Differential && isequal(arg_op.x, x)
                    if get_max_derivative_order(arg_unwrapped, u_op, arg_op.x) == 1
                        has_deriv = true
                        break
                    end
                end
            end

            if has_deriv
                for arg in args
                    arg_unwrapped = unwrap(arg)
                    arg_op = iscall(arg_unwrapped) ? operation(arg_unwrapped) : nothing
                    is_deriv = arg_op isa Differential && isequal(arg_op.x, x) &&
                        get_max_derivative_order(arg_unwrapped, u_op, arg_op.x) == 1
                    if !is_deriv && !is_static_numeric_constant(arg_unwrapped)
                        return true
                    end
                end
            end
        end

        for arg in arguments(expr)
            if is_dynamic_advection(arg, u_op, x)
                return true
            end
        end
    end
    return false
end
