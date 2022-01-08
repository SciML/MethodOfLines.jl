"""
Counts the Differential operators for given variable x. This is used to determine
the order of a PDE.
"""
function count_differentials(term, x::Symbolics.Symbolic)
    S = Symbolics
    SU = SymbolicUtils
    if !S.istree(term)
        return 0
    else
        op = SU.operation(term)
        count_children = sum(map(arg -> count_differentials(arg, x), SU.arguments(term)))
        if op isa Differential && op.x === x
            return 1 + count_children
        end
        return count_children
    end
end

"""
return list of differential orders in the equation
"""
function differential_order(eq, x::Symbolics.Symbolic)
    S = Symbolics
    SU = SymbolicUtils
    orders = Set{Int}()
    if S.istree(eq)
        op = SU.operation(eq)
        if op isa Differential
            push!(orders, count_differentials(eq, x))
        else
            for o in map(ch -> differential_order(ch, x), SU.arguments(eq))
                union!(orders, o)
            end
        end
    end
    return filter(!iszero, orders)
end

"""
find all the dependent variables given by depvar_ops in an expression
"""
function get_depvars(eq,depvar_ops)
    S = Symbolics
    SU = SymbolicUtils
    depvars = Set()
    if eq isa Num
       eq = eq.val
    end
    if S.istree(eq)
        if eq isa Term && any(u->isequal(operation(eq),u),depvar_ops)
              push!(depvars, eq)
        else
            for o in map(x->get_depvars(x,depvar_ops), SU.arguments(eq))
                union!(depvars, o)
            end
        end
    end
    return depvars
end

half_range(x) = -div(x,2):div(x,2)
