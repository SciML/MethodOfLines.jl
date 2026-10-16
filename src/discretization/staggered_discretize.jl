function SciMLBase.discretize(
        pdesys::PDESystem,
        discretization::MOLFiniteDifference{G},
        analytic = nothing, kwargs...
    ) where {G <: StaggeredGrid}
    sys, tspan = SciMLBase.symbolic_discretize(pdesys, discretization)
    return try
        simpsys = mtkcompile(sys)
        if tspan === nothing
            add_metadata!(getmetadata(sys, ModelingToolkit.ProblemTypeCtx, nothing), sys)
            unkns = get_unknowns(simpsys)
            return prob = NonlinearProblem(
                simpsys, unkns .=> ones(length(unkns));
                discretization.kwargs..., kwargs...
            )
        else
            mol_metadata = getmetadata(simpsys, ModelingToolkit.ProblemTypeCtx, nothing)
            add_metadata!(mol_metadata, sys)
            prob = ODEProblem(
                simpsys, nothing, tspan;
                discretization.kwargs...,
                kwargs...
            )
            return staggered_split(prob, simpsys, edge_aligned_dims(discretization))
        end
    catch e
        error_analysis(sys, e)
    end
end

# `f1` and `f2` hold the time derivatives of two groups that 2-colour the coupling graph
# of the variables, as `staggered_groups` does, so that every coupled pair straddles the
# two halves. Of each connected component, `f1` takes the colour of the variables edge
# aligned in an odd number of directions, so a pure Yee system splits into E and H as its
# staggering sets them. Constant fields go in `f2`. Each half generates its own rows of
# `sys` on their own and zeroes the rest, so `f1 + f2` is exactly `f`, with time,
# parameters and boundary values live.
function staggered_split(prob, sys, edge_dims)
    nedge = Dict(nameof(op) => length(dims) for (op, dims) in edge_dims)
    isodd_edge(v) = isodd(get(nedge, nameof(operation(v)), 0))
    # `mtkcompile` leaves one `D(u[i]) ~ rhs` per unknown, in the order of the unknowns
    eqs = get_eqs(sys)
    owner = map(eq -> _parent_array(only(arguments(eq.lhs))), eqs)
    isconstant = map(eq -> isequal(unwrap_const(eq.rhs), 0), eqs)
    colour = _coupling_colours(eqs[.!isconstant], owner[.!isconstant]; first = isodd_edge)
    isf1 = [get(colour, v, 2) == 1 for v in owner]
    rows1, rows2 = findall(isf1), findall(!, isf1)
    rhs(rows) = ModelingToolkit.generate_rhs(
        ModelingToolkitBase.@set(sys.eqs = eqs[rows]);
        expression = Val{false}, wrap_gfw = Val{true}
    )
    g1, g2 = rhs(rows1), rhs(rows2)
    f1(du, u, p, t) = (g1(view(du, rows1), u, p, t); du[rows2] .= 0; nothing)
    f2(du, u, p, t) = (g2(view(du, rows2), u, p, t); du[rows1] .= 0; nothing)
    return SplitODEProblem(f1, f2, prob.u0, prob.tspan, prob.p)
end

# Colour 1 or 2 for each of `owner`, the variables whose time derivatives `derivs` give, so
# that no variable shares a colour with one in its time derivative's right hand side. The
# first variable of each connected component, taking the variables that satisfy `first`
# before the others and each in order, gets colour 1.
function _coupling_colours(derivs, owner; first = Returns(false))
    vars = unique(owner)
    vars = [filter(first, vars); filter(!first, vars)]
    coupled = Dict(v => Set{Any}() for v in vars)
    for (eq, v) in zip(derivs, owner), x in Symbolics.get_variables(eq.rhs)
        w = _parent_array(x)
        if haskey(coupled, w) && !isequal(w, v)
            push!(coupled[v], w)
            push!(coupled[w], v)
        end
    end
    colour = Dict{Any, Int}()
    for root in vars
        haskey(colour, root) && continue
        colour[root] = 1
        queue = [root]
        while !isempty(queue)
            v = popfirst!(queue)
            for w in coupled[v]
                if !haskey(colour, w)
                    colour[w] = 3 - colour[v]
                    push!(queue, w)
                elseif colour[w] == colour[v]
                    throw(
                        ArgumentError(
                            "The coupling graph of the variables is not bipartite: $v and $w are coupled through a cycle of odd length, so no two groups separate every coupled pair."
                        )
                    )
                end
            end
        end
    end
    return colour
end

"""
    staggered_groups(prob::DAEProblem)

Split the differential equations of `prob`, a `DAEProblem(pdesys, disc)` on a staggered
grid, into the two groups a leapfrog alternates between, and generate each group's right
hand side on its own, so that each half step evaluates only its own equations.

Returns `(; groups, constant, algebraic)`:

  - `groups` holds two `(; f, states)`. `f(dv, u, p, t)` writes the time derivatives of
    the states `states` at the full state `u`, in that order: `dv[k]` is `du[states[k]]`.
    Taking `v = u[states]` as a partitioned state therefore needs no gather. `f` reads all
    of `u`, so both halves of a partitioned state are scattered into one full state first.
  - `constant` holds the states whose time derivative is zero (`Dt(c) ~ 0` coefficient
    fields). Neither group writes them.
  - `algebraic` holds the states `prob.differential_vars` marks algebraic, such as PEC walls
    and other Dirichlet points. Neither group writes them, so a leapfrog holds them at their
    value, which is right for time-independent Dirichlet values only.

A row's time derivative is solved for from its residual, so for any `u`, setting
`du[states] = dv` for both groups and zero elsewhere makes `prob.f` vanish on every
differential row.

The groups 2-colour the coupling graph of the variables: `w` is coupled to `u` when `w`
appears in `Dt(u)`'s right hand side. Each coupled pair then straddles the two groups, as
the leapfrog needs; grouping by staggering instead would put a current that shares `E`'s
nodes in `E`'s group. For Maxwell's equations with a Drude current `J` and its integral
`G` the groups are `(E, G)` and `(H, J)`. Taking the variables in the order of their
equations, which follows the PDEs, the first of each connected component goes in the first
group. A coupling graph that is not bipartite throws an `ArgumentError`.
"""
function staggered_groups(prob::DAEProblem)
    sys = prob.f.sys
    derivs = filter(!isnothing, map(_solve_time_derivative, get_eqs(sys)))
    owner = map(eq -> _parent_array(only(arguments(eq.lhs))), derivs)
    isconstant = map(eq -> eq.rhs === 0, derivs)
    states(eqs) = reduce(
        vcat, map(eq -> _state_indices(sys, only(arguments(eq.lhs))), eqs); init = Int[]
    )

    colour = _coupling_colours(derivs[.!isconstant], owner[.!isconstant])

    groups = map((1, 2)) do c
        eqs = [
            eq.lhs ~ eq.rhs for (eq, k, v) in zip(derivs, isconstant, owner)
                if !k && colour[v] == c
        ]
        gsys = Setfield.@set sys.eqs = eqs
        f = ModelingToolkit.generate_rhs(gsys; expression = Val{false}, wrap_gfw = Val{true})
        (; f, states = states(eqs))
    end
    return (;
        groups, constant = states(derivs[isconstant]),
        algebraic = findall(!, prob.differential_vars),
    )
end

# `(lhs = D(u[r]), rhs = f)`, with `f` the literal `0` for a constant, for an equation linear
# in its one time derivative `D(u[r])`, which MOL writes as sums of scalar multiples,
# broadcast over slices or, for a pointwise equation, of scalars; `nothing` for an
# algebraic equation. Not an `Equation`, which needs `f` to have
# the shape of `D(u[r])`.
function _solve_time_derivative(eq)
    coefs = Dict{Any, Any}()
    lhs = _split_time_derivative!(coefs, safe_unwrap(eq.lhs), 1)
    rhs = _split_time_derivative!(coefs, safe_unwrap(eq.rhs), -1)
    isempty(coefs) && return nothing
    length(coefs) == 1 || throw(
        ArgumentError(
            "$eq holds more than one time derivative, so it cannot be solved for one of them."
        )
    )
    (d, a), = coefs
    lhs = lhs === nothing || _iszero_broadcast(lhs) ? 0 : lhs
    rhs = rhs === nothing || _iszero_broadcast(rhs) ? 0 : rhs
    # the residual is `rhs - lhs - a * d`
    f = rhs === 0 ? (lhs === 0 ? 0 : _broadcast(*, -1, lhs)) :
        lhs === 0 ? rhs : _broadcast(-, rhs, lhs)
    return (lhs = d, rhs = isequal(a, 1) || f === 0 ? f : _broadcast(/, f, a))
end

# Return the part of `ex` free of time derivatives (`nothing` if there is none), and add `c`
# times the coefficient of each time derivative in `ex` to `coefs`.
function _split_time_derivative!(coefs, ex, c)
    _has_differential(ex) || return ex
    _iszero_broadcast(ex) && return nothing
    if operation(ex) isa Differential
        _iszero_broadcast(c) ||
            (coefs[ex] = haskey(coefs, ex) ? _broadcast(+, coefs[ex], c) : c)
        return nothing
    elseif (args = _operands(ex, +)) !== nothing
        kept = [_split_time_derivative!(coefs, x, c) for x in args]
        filter!(!isnothing, kept)
        return isempty(kept) ? nothing : length(kept) == 1 ? only(kept) : _broadcast(+, kept...)
    elseif (args = _operands(ex, *)) !== nothing
        i = findall(_has_differential, args)
        if length(i) == 1
            factors = args[setdiff(eachindex(args), i)]
            rest = _split_time_derivative!(coefs, args[only(i)], _broadcast(*, c, factors...))
            return rest === nothing ? nothing : _broadcast(*, factors..., rest)
        end
    end
    throw(ArgumentError("$ex is not linear in its time derivative."))
end

# The operands of `ex` if it applies `f`, elementwise or to scalars, else `nothing`.
function _operands(ex, f)
    _isbroadcast(ex, f) && return arguments(ex)[2:end]
    return iscall(ex) && operation(ex) === f ? arguments(ex) : nothing
end

function _isbroadcast(ex, f)
    return iscall(ex) && operation(ex) in (broadcast, mol_broadcast) &&
        unwrap_const(arguments(ex)[1]) === f
end

# MOL spells a zero array as a literal, as `0 .* x` or as `0 .+ 0 .* x`.
function _iszero_broadcast(ex)
    ex = unwrap_const(ex)
    ex isa Union{Number, AbstractArray{<:Number}} && return all(iszero, ex)
    _isbroadcast(ex, +) && return all(_iszero_broadcast, arguments(ex)[2:end])
    _isbroadcast(ex, *) && return any(_iszero_broadcast, arguments(ex)[2:end])
    return false
end

# `f.(xs...)`, spelled as `mol_broadcast`, when any of `xs` is an array, else `f(xs...)`
function _broadcast(f, xs...)
    any(is_array_valued, xs) || return f(unwrap_const.(xs)...)
    return array_noinline_broadcasts(broadcast(f, xs...))
end

_parent_array(x) = iscall(x) && operation(x) === getindex ? first(arguments(x)) : x

# Positions in the state vector of `x`, an unknown or a slice of one, in column-major order.
function _state_indices(sys, x)
    idx = SymbolicIndexingInterface.variable_index(sys, _parent_array(x))
    x === _parent_array(x) && return vec(collect(idx))
    return vec(collect(idx[unwrap_const.(arguments(x)[2:end])...]))
end
