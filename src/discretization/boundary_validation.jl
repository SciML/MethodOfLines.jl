import PDEBase: check_boundarymap
using PDEBase: VariableMap, depvars, get_time, get_eqs, d_orders
using ModelingToolkit: operation, arguments, iscall, Differential
using Symbolics: unwrap

"""
    validate_system_wellposedness(pdes, bmap, s, disc)

Validates the well-posedness of the PDE system. 
If a variable has a spatial derivative with respect to a given dimension, 
this function ensures that boundary conditions are provided for both ends of that domain.
"""
function validate_system_wellposedness(pdes, bmap, s, disc)
    for u_call in s.ū
        u_op = operation(u_call)
        spatial_ivs = s.x̄ 

        for x in spatial_ivs
            has_deriv = false
            for eq in pdes
                if occursin_derivative_of(eq.lhs, u_op, x) || occursin_derivative_of(eq.rhs, u_op, x)
                    has_deriv = true
                    break
                end
            end

            if has_deriv
                u_bcs = get(bmap, u_op, nothing)
                u_bcs_x = u_bcs !== nothing ? get(u_bcs, x, nothing) : nothing

                if u_bcs_x === nothing || length(u_bcs_x) < 2
                    throw(ArgumentError(
                        "Missing boundary condition for variable $(u_op) in dimension $(x). " *
                        "The system is ill-posed. Since $(u_op) has spatial derivatives with respect to $(x), " *
                        "you must provide boundary conditions for both ends of the domain."
                    ))
                end
            end
        end
    end
end

function occursin_derivative_of(expr, u_op, x)
    expr = unwrap(expr)
    if iscall(expr)
        op = operation(expr)
        args = arguments(expr)

        if op isa Differential && isequal(op.x, x)
            if any(arg -> occursin_variable(arg, u_op), args)
                return true
            end
        end
        return any(arg -> occursin_derivative_of(arg, u_op, x), args)
    end
    return false
end

function occursin_variable(expr, u_op)
    expr = unwrap(expr)
    if iscall(expr)
        if isequal(operation(expr), u_op)
            return true
        end
        return any(arg -> occursin_variable(arg, u_op), arguments(expr))
    end
    return isequal(expr, u_op)
end