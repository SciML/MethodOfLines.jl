broadcast_substitute(expr, pairs) = _sub(expr, pairs, false)
broadcast_substitute(expr, pairs, verbose) = _sub(expr, pairs, verbose)


function _sub(expr, pairs, verbose)
    ipair = findfirst(p -> isequal(p.first, expr), pairs)
    if ipair !== nothing
        verbose && @warn "Replacing $expr with $(pairs[ipair])"
        return pairs[ipair].second
    elseif istree(expr)
        op = operation(expr)
        args = _sub.(arguments(expr), (pairs,), (verbose,))
        if any(arg -> symtype(arg) <: AbstractArray, args)
            try
                return broadcast(op, args...)
                #return unwrap(op(map(wrap, args)...))
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
