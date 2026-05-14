import PDEBase: check_boundarymap
using PDEBase: VariableMap, depvars, get_time, get_eqs, d_orders
using ModelingToolkit: operation, arguments, iscall, Differential
using Symbolics: unwrap

"""
    validate_system_wellposedness(pdes, bmap, s, disc)

Validates the well-posedness of the PDE system. 
Dynamically checks the maximum spatial derivative order for each variable in a given dimension, 
and ensures that at least that many boundary conditions are provided.
"""
function validate_system_wellposedness(pdes, bmap, s, disc)
    for u_call in s.ū
        u_op = operation(u_call)
        spatial_ivs = s.x̄ 

        for x in spatial_ivs
            max_order = 0
            for eq in pdes
                order_lhs = get_max_derivative_order(eq.lhs, u_op, x)
                order_rhs = get_max_derivative_order(eq.rhs, u_op, x)

                max_order = max(max_order, order_lhs, order_rhs)
            end

            if max_order > 0
                u_bcs = get(bmap, u_op, nothing)
                u_bcs_x = u_bcs !== nothing ? get(u_bcs, x, nothing) : nothing
                bc_count = u_bcs_x === nothing ? 0 : length(u_bcs_x)

                if bc_count < max_order
                    throw(ArgumentError(
                        "Missing boundary conditions for variable $(u_op) in dimension $(x). " *
                        "The system is ill-posed. The highest spatial derivative order is $(max_order), " *
                        "but only $(bc_count) boundary condition(s) were provided."
                    ))
                end
            end
        end
    end
end

"""
    get_max_derivative_order(expr, u_op, x)

Recursively scans the expression tree in a single pass to find the maximum 
derivative order of `u_op` with respect to `x`. 
Accounts for higher-order symbolic derivatives natively.
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

        if op isa Differential && isequal(op.x, x)
            if inner_max >= 0
                diff_order = hasproperty(op, :order) ? op.order : 1
                return diff_order + inner_max
            end
        end

        return inner_max
    end

    return isequal(expr, u_op) ? 0 : -1
end