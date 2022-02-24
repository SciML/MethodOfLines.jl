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

@inline function get_all_depvars(pdesys, depvar_ops)
    pdeeqs = pdesys.eqs isa Vector ? pdesys.eqs : [pdesys.eqs]
    return collect(mapreduce(x->get_depvars(x.lhs,depvar_ops), union, pdeeqs) ∪ mapreduce(x->get_depvars(x.rhs,depvar_ops), union, pdeeqs))
end

"""
A function that creates a tuple of CartesianIndices of unit length and `N` dimensions, one pointing along each dimension.
"""
function unitindices(N::Int) #create unit CartesianIndex for each dimension
    null = zeros(Int, N)
    if N == 0
        return CartesianIndex()
    else
        return map(1:N) do i
            unit_i = copy(null)
            unit_i[i] = 1
            CartesianIndex(Tuple(unit_i))
        end
    end
end

@inline function unitindex(N, j)
    N == 0  && return CartesianIndex()
    unitindices(N)[j]
end

function _split_terms(term)
    S = Symbolics
    SU = SymbolicUtils
    # TODO: Update this to be exclusive of derivatives and depvars rather than inclusive of +-/*
    if S.istree(term) && ((operation(term) == +) | (operation(term) == -) | (operation(term) == *) | (operation(term) == /))
        return mapreduce(_split_terms, vcat, SU.arguments(term))
    else
        return [term]
    end
end

function split_terms(eq::Equation)
    lhs = _split_terms(eq.lhs)
    rhs = _split_terms(eq.rhs)
    return vcat(lhs,rhs)
end

function split_additive_terms(eq)
    # Calling the methods from symbolicutils matches the expressions
    rhs_arg = istree(eq.rhs) && (SymbolicUtils.operation(eq.rhs) == +) ? SymbolicUtils.arguments(eq.rhs) : [eq.rhs]
    lhs_arg = istree(eq.lhs) && (SymbolicUtils.operation(eq.lhs) == +) ? SymbolicUtils.arguments(eq.lhs) : [eq.lhs]

    return vcat(lhs_arg,rhs_arg)
end

@inline clip(II::CartesianIndex{M}, j, N) where M = II[j] > N ? II - unitindices(M)[j] : II

subsmatch(expr, rule) = isequal(substitute(expr, rule), expr) ? false : true

#substitute(eq::Equation, rules) = substitute(eq.lhs, rules) ~ substitute(eq.rhs, rules)

remove(args, t) = filter(x -> t === nothing || !isequal(x, t.val), args)

half_range(x) = -div(x,2):div(x,2)


