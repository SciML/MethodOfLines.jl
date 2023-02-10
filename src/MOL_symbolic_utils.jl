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
        if op isa Differential && isequal(op.x, x)
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
Determine whether a term has a derivative anywhere in it.
"""
function has_derivatives(term)
    if istree(term)
        op = operation(term)
        if op isa Differential
            return true
        else
            return any(has_derivatives, arguments(term))
        end
    else
        return false
    end
end

"""
Finds the derivative or depvar within a term
"""
function find_derivative(term, depvar_op)
    S = Symbolics
    SU = SymbolicUtils
    orders = Set{Int}()
    if S.istree(eq)
        op = SU.operation(term)
        if (op isa Differential) | isequal(op, depvar_op)
            return term
        else
            for arg in SU.arguments(term)
                res = find_derivative(arg, depvar_op)
                if res !== nothing
                    return res
                end
            end
        end
    end
    return nothing
end

"""
Substitute rules in all equations and bcs inplace
"""
function subs_alleqs!(eqs, bcs, rules)
    subs_alleqs!(eqs, rules)
    subs_alleqs!(bcs, rules)
end

subs_alleqs!(eqs, rules) = map!(eq -> substitute(eq.lhs, rules) ~ substitute(eq.rhs, rules), eqs, eqs)

"""
find all the dependent variables given by depvar_ops in an expression
"""
function get_depvars(eq, depvar_ops)
    depvars = Set()
    eq = safe_unwrap(eq)
    if istree(eq)
        if any(u -> isequal(operation(eq), u), depvar_ops)
            push!(depvars, eq)
        else
            for o in map(x -> get_depvars(x, depvar_ops), arguments(eq))
                union!(depvars, o)
            end
        end
    end
    return depvars
end

@inline function get_all_depvars(pdeeqs, depvar_ops)
    return collect(mapreduce(x -> get_depvars(x.lhs, depvar_ops), union, pdeeqs) ∪ mapreduce(x -> get_depvars(x.rhs, depvar_ops), union, pdeeqs))
end

@inline function get_all_depvars(pdesys::PDESystem, depvar_ops)
    pdeeqs = pdesys.eqs
    return collect(mapreduce(x -> get_depvars(x.lhs, depvar_ops), union, pdeeqs) ∪ mapreduce(x -> get_depvars(x.rhs, depvar_ops), union, pdeeqs))
end

get_ops(depvars) = map(u -> operation(safe_unwrap(u)), depvars)

function split_terms(eq::Equation)
    lhs = _split_terms(eq.lhs)
    rhs = _split_terms(eq.rhs)
    return vcat(lhs, rhs)
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

# Additional handling to get around limitations in rules
# Splits out derivatives from containing math expressions for ingestion by the rules
function _split_terms(term, x̄)
    S = Symbolics
    SU = SymbolicUtils
    st(t) = _split_terms(t, x̄)
    # TODO: Update this to handle more ops e.g. exp sin tanh etc.
    # TODO: Handle cases where two nonlinear laplacians are multiplied together
    if S.istree(term)
        # Additional handling for upwinding
        if (operation(term) == *)
            args = SU.arguments(term)
            for (i, arg) in enumerate(args)
                # Incase of upwinding, we need to keep the original term
                if S.istree(arg) && operation(arg) isa Differential
                    # Flatten the arguments of the differential to make nonlinear laplacian work in more cases
                    try
                        args[i] = operation(arg)(flatten_division.(SU.arguments(arg))...)
                    catch e
                        println("Argument to derivative in $term is not a dependant variable, is trivially differentiable or is otherwise not differentiable.")
                        throw(e)
                    end
                    return [*(flatten_division.(args)...)]
                end
            end
            return mapreduce(st, vcat, SU.arguments(term))
        elseif (operation(term) == /)
            args = SU.arguments(term)
            # Incase of upwinding or spherical, we need to keep the original term
            if S.istree(args[1])
                if args[1] isa Differential
                    try
                        args[1] = operation(arg)(flatten_division.(SU.arguments(arg))...)
                    catch e
                        println("Argument to derivative in $term is not a dependant variable, is trivially differentiable or is otherwise not differentiable.")
                        throw(e)
                    end
                    return [/(flatten_division.(args)...)]
                    # Handle with care so that spherical still works
                elseif operation(args[1]) == *
                    subargs = SU.arguments(args[1])
                    # look for a differential in the arguments
                    for (i, arg) in enumerate(subargs)
                        if S.istree(arg) && operation(arg) isa Differential
                            # Flatten the arguments of the differential to make nonlinear laplacian/spherical work in more cases
                            try
                                subargs[i] = operation(arg)(flatten_division.(SU.arguments(arg))...)
                                args[1] = operation(args[1])(flatten_division.(subargs)...)
                            catch e
                                println("Argument to derivative in $term is not a dependant variable, is trivially differentiable or is otherwise not differentiable.")
                                throw(e)
                            end
                            return [/(flatten_division.(args)...)]
                        end
                    end
                end
            end
            # Basecase for division
            return vcat(st(args[1]), st(args[2]))
        elseif (operation(term) == +) | (operation(term) == -)
            return mapreduce(st, vcat, SU.arguments(term))
        elseif (operation(term) isa Differential)
            return [operation(term)(flatten_division.(SU.arguments(term))...)]
        else
            return [term]
        end
    else
        return [term]
    end
end

function split_terms(eq::Equation, x̄)
    lhs = _split_terms(eq.lhs, x̄)
    rhs = _split_terms(eq.rhs, x̄)
    return filter(term -> !isequal(term, Num(0)), flatten_division.(vcat(lhs, rhs)))
end

function split_additive_terms(eq)
    # Calling the methods from symbolicutils matches the expressions
    rhs_arg = istree(eq.rhs) && (SymbolicUtils.operation(eq.rhs) == +) ? SymbolicUtils.arguments(eq.rhs) : [eq.rhs]
    lhs_arg = istree(eq.lhs) && (SymbolicUtils.operation(eq.lhs) == +) ? SymbolicUtils.arguments(eq.lhs) : [eq.lhs]

    return vcat(lhs_arg, rhs_arg)
end

# Filthy hack to get around limitations in rules and avoid simplification to a dividing expression
@inline function flatten_division(term)
    #=rules = [@rule(/(~a, ~b) => *(~a, b^(-1.0))),
             @rule(/(*(~~a), ~b) => *(~a..., b^(-1.0))),
             @rule(/(~a, *(~~b)) => *(~a, *(~b...)^(-1.0))),
             @rule(/(*(~~a), *(~~b)) => *(~a..., *(~b...)^(-1.0)))]
    for r in rules
        if r(term) !== nothing
            return r(term)
        end
    end=#
    return term
end

subsmatch(expr, rule) = isequal(substitute(expr, rule), expr) ? false : true
subsmatch(eq::Equation, rule) = subsmatch(eq.lhs, rule) | subsmatch(eq.rhs, rule)
#substitute(eq::Equation, rules) = substitute(eq.lhs, rules) ~ substitute(eq.rhs, rules)

"""
    ex2term(x::Term) -> Symbolic
    ex2term(x) -> x

Convert a Term to a variable `Term`. Note that it only takes a `Term`
not a `Num`.
```
"""
function ex2term(term, v)
    istree(term) || return term
    termdvs = collect(get_depvars(term, v.depvar_ops))
    symdvs = filter(u -> all(x -> !(safe_unwrap(x) isa Number), arguments(u)), termdvs)
    exdv = last(sort(symdvs, by=u -> length(arguments(u))))
    name = Symbol("⟦" * string(term) * "⟧")
    return setname(similarterm(exdv, rename(operation(exdv), name), arguments(exdv)), name)
end

safe_unwrap(x) = x isa Num ? unwrap(x) : x

function recursive_unwrap(ex)
    if !istree(ex)
        return safe_unwrap(ex)
    end

    op = operation(ex)
    args = arguments(ex)
    return safe_unwrap(op(recursive_unwrap.(args)))
end
