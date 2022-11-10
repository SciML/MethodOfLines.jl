broadcast_substitute(expr, dict) = _sub(expr, dict isa Dict ? dict : Dict(dict))

function _sub(expr, dict)
    if haskey(dict, expr)
        return dict[expr]
    elseif istree(expr)
        op = operation(expr)
        args = _sub.(arguments(expr), (dict,))
        if any(arg -> symtype(arg) <: AbstractArray, args)
            try
                return unwrap(op(map(wrap, args)...))
            catch e
                throw(ArgumentError("Cannot broadcast operation $op over arguments $args"))
            end
        else
            return op(args...)
        end
    else
        return expr
    end
end
