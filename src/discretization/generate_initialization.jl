# Initialization of the time-dependent system from the array-valued initial conditions of
# `generate_ic_defaults`. An initial condition fixes the part of its variable that the
# equations differentiate to a higher order in time than the condition's own; the rest of
# it is only a guess for the consistent-initialization solve. MethodOfLines differentiates
# whole slices, `Dt(u[2:n-1, ...])`, so the initialization has one equation per such slice
# rather than one per grid point.

"""
    generate_system(disc_state, s::DiscreteSpace, u0, tspan, metadata, disc::MOLDiscretization)

PDEBase's `generate_system`, with initialization equations and guesses built from the
array-valued initial conditions `u0` by [`discrete_initialization`](@ref).
"""
function PDEBase.generate_system(
        disc_state::PDEBase.EquationState, s::DiscreteSpace, u0, tspan, metadata,
        disc::MOLDiscretization; checks = true
    )
    t = get_time(disc)
    if t === nothing || isempty(u0)
        return invoke(
            PDEBase.generate_system,
            Tuple{
                PDEBase.EquationState, Any, Any, Any, Any,
                AbstractEquationSystemDiscretization,
            },
            disc_state, s, u0, tspan, metadata, disc; checks
        )
    end
    pdesys = metadata.pdesys
    alleqs = vcat(disc_state.eqs, unique(disc_state.bceqs))
    init_eqs, guesses = discrete_initialization(alleqs, safe_unwrap(t), u0)
    defaults = Dict{Any, Any}(pdesys.initial_conditions)
    ps = ModelingToolkitBase.get_ps(pdesys)
    ps = ps === nothing || ps === SciMLBase.NullParameters() ? Num[] : ps
    # parameters given as `p => value` carry their defaults
    if !isempty(ps) && first(ps) isa Pair
        merge!(defaults, Dict{Any, Any}(ps))
        ps = Num[first(p) for p in ps]
    end
    sys = ModelingToolkitBase.System(
        alleqs, t, get_system_unknowns(s), ps;
        initial_conditions = defaults, initialization_eqs = init_eqs, guesses, tspan,
        name = getfield(pdesys, :name),
        metadata = [ModelingToolkitBase.ProblemTypeCtx => metadata], checks
    )
    return sys, tspan
end

"""
    discrete_initialization(eqs, t, u0)

The initialization equations and guesses for the discretized equations `eqs` from the
initial conditions `u0`, pairs from an unknown (or a time derivative of one) to its
values. Each initial condition is a guess for its whole variable, and an equation for
each part of the variable that `eqs` differentiate to a higher order in time.
"""
function discrete_initialization(eqs, t, u0)
    parts = differentiated_parts(eqs, t)
    init_eqs = Equation[]
    guesses = Dict{Any, Any}()
    for (key, vals) in u0
        guesses[key] = vals
        unknown, order = _time_derivative_of(key, t)
        D = order == 0 ? identity : Differential(t)^order
        covered = vals isa AbstractArray ? falses(size(vals)) : falses()
        for (part, idxs, part_order) in get(parts, unknown, ())
            part_order > order || break
            region = idxs === nothing ? CartesianIndices(covered) : _region(idxs)
            if !any(view(covered, region))
                push!(init_eqs, D(part) ~ (idxs === nothing ? vals : vals[idxs...]))
                covered[region] .= true
                continue
            end
            # overlapping slices: the remaining points one by one, so that no point is
            # constrained twice
            for I in region
                covered[I] && continue
                push!(init_eqs, D(unknown[Tuple(I)...]) ~ vals[I])
                covered[I] = true
            end
        end
    end
    return init_eqs, guesses
end

# For each unknown, the parts of it the equations differentiate in time: tuples of the
# differentiated expression, the indices it covers (`nothing` for the whole unknown) and
# its highest order, highest order first.
function differentiated_parts(eqs, t)
    parts = Dict{Any, Vector{Tuple{Any, Any, Int}}}()
    for (part, order) in time_differentiated(eqs, t)
        unknown, idxs = _parent_and_indices(part)
        push!(get!(parts, unknown, Tuple{Any, Any, Int}[]), (part, idxs, order))
    end
    for v in values(parts)
        sort!(v; by = p -> (-p[3], string(p[1])))
    end
    return parts
end

# Each expression differentiated in time in `eqs`, with the highest order it is
# differentiated to.
function time_differentiated(eqs, t)
    out = Dict{Any, Int}()
    for eq in eqs
        _time_differentiated!(out, eq.lhs, t)
        _time_differentiated!(out, eq.rhs, t)
    end
    return out
end

function _time_differentiated!(out, ex, t)
    ex = safe_unwrap(ex)
    _issym(ex) || return out
    if _is_time_derivative(ex, t)
        part, order = _time_derivative_of(ex, t)
        out[part] = max(order, get(out, part, 0))
        return out
    end
    for arg in arguments(ex)
        _time_differentiated!(out, arg, t)
    end
    return out
end

_is_time_derivative(ex, t) = _issym(ex) && operation(ex) isa Differential &&
    isequal(operation(ex).x, t)

# `Dt^k(x)` as `(x, k)`.
function _time_derivative_of(ex, t)
    ex = safe_unwrap(ex)
    order = 0
    while _is_time_derivative(ex, t)
        order += operation(ex).order
        ex = safe_unwrap(only(arguments(ex)))
    end
    return ex, order
end

# The array an element or slice `u[i, 2:n-1]` indexes and its indices, or `x` and
# `nothing` for anything else.
function _parent_and_indices(x)
    iscall(x) && operation(x) === getindex || return x, nothing
    args = arguments(x)
    return safe_unwrap(args[1]), Tuple(unwrap_const(safe_unwrap(i)) for i in args[2:end])
end

_region(idxs) = CartesianIndices(map(i -> i isa Integer ? (i:i) : i, idxs))
