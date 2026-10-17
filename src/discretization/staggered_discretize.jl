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

# `f1` holds the time derivatives of the variables edge aligned in an odd number of
# directions and `f2` those of every other variable. A staggered derivative moves its
# variable half a step in one direction, so it always couples the two groups, while
# pointwise terms stay within one: for Maxwell's equations the groups are E and H.
# Each half evaluates the full right hand side and zeroes the other group's rows, so
# `f1 + f2` is exactly `f` for any number of variables, with time, parameters and
# boundary values live, at the price of a second evaluation of `f` per step.
function staggered_split(prob, sys, edge_dims)
    get_var_from_state(state) = operation(arguments(state)[1])
    nedge = Dict(nameof(op) => length(dims) for (op, dims) in edge_dims)
    isedge = [
        isodd(get(nedge, nameof(get_var_from_state(u)), 0)) for u in get_unknowns(sys)
    ]
    rows1, rows2 = findall(isedge), findall(!, isedge)
    f1(du, u, p, t) = (prob.f(du, u, p, t); du[rows2] .= 0; nothing)
    f2(du, u, p, t) = (prob.f(du, u, p, t); du[rows1] .= 0; nothing)
    return SplitODEProblem(f1, f2, prob.u0, prob.tspan, prob.p)
end

"""
    staggered_groups(prob::DAEProblem; lossy = false, fuse_broadcasts = true, kwargs...)

Split the differential equations of `prob`, a `DAEProblem(pdesys, disc)` on a staggered
grid, into the two groups a leapfrog alternates between, and generate each group's right
hand side on its own, so that each half step evaluates only its own equations.

Returns `(; groups, constant, algebraic)`:

  - `groups` holds two `(; f, update!, states)`. `f(dv, u, p, t)` writes the time
    derivatives of the states `states` at the full state `u`, in that order: `dv[k]` is
    `du[states[k]]`. Taking `v = u[states]` as a partitioned state therefore needs no
    gather. `f` reads all of `u`, so both halves of a partitioned state are scattered into
    one full state first.
  - `update!(u, p, t, dt)` steps the group's states in place, `u[states] .+= dt .* dv`,
    without `dv`: each variable's regions are written straight into its block of `u`. With
    `lossy = true` it is `update!(u, p, t, dt, a, b)`, `u[states] .= a[states] .* u[states]
    .+ b[states] .* dv`, for per-state coefficient vectors `a` and `b` laid out like `u`.
    `fuse_broadcasts` and `kwargs` are passed to the code generation of `update!`.
    An equation that reads a state its group writes, other than its own, would race with
    the in-place write: fused code generation then copies the group's inputs on entry, and
    without it `update!` throws.
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
function staggered_groups(
        prob::DAEProblem; lossy = false, fuse_broadcasts = true, kwargs...
    )
    sys = prob.f.sys
    derivs = filter(!isnothing, map(_solve_time_derivative, get_eqs(sys)))
    owner = map(eq -> _parent_array(only(arguments(eq.lhs))), derivs)
    isconstant = map(eq -> eq.rhs === 0, derivs)
    states(eqs) = reduce(
        vcat, map(eq -> _state_indices(sys, only(arguments(eq.lhs))), eqs); init = Int[]
    )

    vars = unique(owner[.!isconstant])
    coupled = Dict(v => Set{Any}() for v in vars)
    for (eq, v) in zip(derivs[.!isconstant], owner[.!isconstant]),
            x in Symbolics.get_variables(eq.rhs)
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

    groups = map((1, 2)) do c
        eqs = [
            eq.lhs ~ eq.rhs for (eq, k, v) in zip(derivs, isconstant, owner)
                if !k && colour[v] == c
        ]
        gsys = Setfield.@set sys.eqs = eqs
        f = ModelingToolkit.generate_rhs(gsys; expression = Val{false}, wrap_gfw = Val{true})
        update! = _group_update(sys, eqs; lossy, fuse_broadcasts, kwargs...)
        (; f, update!, states = states(eqs))
    end
    return (;
        groups, constant = states(derivs[isconstant]),
        algebraic = findall(!, prob.differential_vars),
    )
end

# `update!` for the group of equations `D(u[r]) ~ f` in `eqs`: per variable one in-place
# function over the variable's grid, compiled from an `ArrayMaker` whose regions are the
# equations' `r` and whose values are `u[r] + dt * f` (or `a[r] * u[r] + b[r] * f`), called
# on the variable's block of `u`. The `ArrayMaker` writes only its regions, so constant and
# algebraic states keep their values.
function _group_update(sys, eqs; lossy, fuse_broadcasts, codegen_kwargs...)
    owner(eq) = _parent_array(only(arguments(eq.lhs)))
    race = _group_race(eqs)
    # Without a race every region reads the elements of `u` it writes only at their own
    # index, so fused code may read `u` while writing it, with no copies. With one, fused
    # code copies its inputs on entry; unfused code writes region by region and cannot.
    fused = fuse_broadcasts && isdefined(SymbolicUtils.Code, :FUSE_BROADCASTS_KEY)
    race === nothing || fused || return function (args...)
        throw(
            ArgumentError(
                "$(race[1]) reads $(race[2]), a state of its own group other than its own, so an in-place update would race on it."
            )
        )
    end
    dvs = ModelingToolkitBase.flat_unknowns(sys)
    ps = ModelingToolkitBase.reorder_parameters(sys)
    T = SymbolicUtils.SymReal
    dt = SymbolicUtils.Sym{T}(:mol_dt; type = Real)
    updates = Tuple(
        map(unique(map(owner, eqs))) do v
            idx = SymbolicIndexingInterface.variable_index(sys, v)
            block = first(idx):last(idx)
            vec(collect(idx)) == block || throw(
                ArgumentError("The states of $v are not contiguous in the state vector.")
            )
            sh = SymbolicUtils.shape(v)
            coef(name) = SymbolicUtils.Sym{T}(name; type = Array{Real, length(sh)}, shape = sh)
            ab = lossy ? (coef(:mol_a), coef(:mol_b)) : ()
            # An `ArrayMaker` this small is generated element by element, uncovered elements
            # included, so it is given the variable itself everywhere first.
            regions = length(v) <= 16 ? [collect(UnitRange{Int}, sh)] : Vector{UnitRange{Int}}[]
            values = length(v) <= 16 ? Any[v] : Any[]
            for eq in eqs
                isequal(owner(eq), v) || continue
                x = only(arguments(eq.lhs))
                r = x === v ? collect(UnitRange{Int}, sh) : map(unwrap_const.(arguments(x)[2:end])) do i
                        UnitRange{Int}(i isa Integer ? (i:i) : i)
                end
                slice(c) = Symbolics.unwrap(Symbolics.wrap(c)[r...])
                ar = map(slice, ab)
                # a pointwise equation writes `u[i]`, a scalar: its region is `u[i:i]`
                xr = is_array_valued(x) ? x : slice(v)
                value = lossy ?
                    _broadcast(+, _broadcast(*, ar[1], xr), _broadcast(*, ar[2], eq.rhs)) :
                    _broadcast(+, xr, _broadcast(*, dt, eq.rhs))
                push!(regions, r)
                push!(values, value)
            end
            maker = SymbolicUtils.ArrayMaker{T}(regions, values; shape = sh)
            inplace = race === nothing ? (; fuse_unalias = false) : (;)
            _, iip = ModelingToolkitBase.build_function_wrapper(
                sys, maker, dvs, ps..., ModelingToolkit.get_iv(sys), dt, ab...; p_start = 2,
                p_end = 1 + length(ps), u_arg = 1, n_param_buffers = length(ps),
                fuse_broadcasts, inplace..., codegen_kwargs...
            )
            f! = ModelingToolkitBase.eval_or_rgf(iip; eval_module = @__MODULE__)
            (f!, block, size(idx))
        end
    )
    blockview(x, block, dims) = reshape(view(x, block), dims)
    return function update!(u, p, t, dt, a = nothing, b = nothing)
        foreach(updates) do (f!, block, dims)
            out = blockview(u, block, dims)
            lossy ? f!(out, u, p, t, dt, blockview(a, block, dims), blockview(b, block, dims)) :
                f!(out, u, p, t, dt)
        end
        return nothing
    end
end

# The first `(eq, x)` of the equations `eqs` of one group such that `eq` reads `x`, a slice
# of a variable the group writes other than the slice `eq` writes, compared by value; or
# `nothing`. An in-place update without such reads reads each state it writes only in the
# kernel that writes it, at the same element.
function _group_race(eqs)
    written = Set(map(eq -> _parent_array(only(arguments(eq.lhs))), eqs))
    function slice(x)
        v = _parent_array(x)
        return (v, x === v ? nothing : unwrap_const.(arguments(x)[2:end]))
    end
    for eq in eqs, x in Symbolics.get_variables(eq.rhs)
        _parent_array(x) in written || continue
        xs, ys = slice(x), slice(only(arguments(eq.lhs)))
        isequal(xs[1], ys[1]) && xs[2] == ys[2] || return (eq, x)
    end
    return nothing
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
