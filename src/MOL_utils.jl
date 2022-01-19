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

"""
A function that creates a tuple of CartesianIndices of unit length and `N` dimensions, one pointing along each dimension.
"""
function unitindices(N::Int) #create unit CartesianIndex for each dimension
    out = Vector{CartesianIndex{N}}(undef, N)
    null = zeros(Int, N)
    for i in 1:N
        unit_i = copy(null)
        unit_i[i] = 1
        out[i] = CartesianIndex(Tuple(unit_i))
    end
    Tuple(out)
end

function split_additive_terms(eq)
    rhs_arg = istree(eq.rhs) && (SymbolicUtils.operation(eq.rhs) == +) ? SymbolicUtils.arguments(eq.rhs) : [eq.rhs]
                lhs_arg = istree(eq.lhs) && (SymbolicUtils.operation(eq.lhs) == +) ? SymbolicUtils.arguments(eq.lhs) : [eq.lhs]

    return vcat(lhs_arg,rhs_arg)
end

half_range(x) = -div(x,2):div(x,2)


