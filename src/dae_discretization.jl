# DAEProblem construction, preserving array (slice-form) equations.
#
# `discretize` runs `mtkcompile`, which scalarizes array equations before codegen: an
# `ODEProblem` needs `D(x) = f(x)`, and isolating the derivative is structural
# simplification. MethodOfLines already emits residuals `D(u) - f ~ 0`, which is exactly
# the implicit-DAE form `DAEProblem` consumes, so this path skips `mtkcompile` and the
# array equations survive into the generated code.

"""
    BrownFullBasicInitUnsafeError(offenders)

Raised by `DAEProblem(::PDESystem, ::MOLFiniteDifference)` when the discretized system
carries initialization equations that `BrownFullBasicInit()` would silently discard, so no
default initialization algorithm can be chosen safely. `offenders` pairs each such
equation with the reason it is not honored.
"""
struct BrownFullBasicInitUnsafeError <: Exception
    offenders::Vector{Pair{Equation, String}}
end

function Base.showerror(io::IO, e::BrownFullBasicInitUnsafeError)
    print(
        io,
        """
        MethodOfLines cannot build a `DAEProblem` for this system without `mtkcompile`.

        `BrownFullBasicInit()` is the only initialization algorithm that reproduces the
        `discretize` (`ODEProblem` + `mtkcompile`) result on the array-equation path. It
        takes the differential variables' initial values as given and solves for the
        algebraic variables and the derivatives, so it would silently ignore these
        initialization equations:
        """
    )
    for (eq, reason) in e.offenders
        print(io, "\n  ", eq, "\n    ", reason)
    end
    print(
        io,
        """


        Use `discretize(pdesys, disc)` instead. That path runs `mtkcompile`, which honors
        these constraints, at the cost of scalarizing the array equations: the slice form
        cannot currently be preserved for this system.
        """
    )
    return
end

_issym(ex) = ex isa SymbolicUtils.BasicSymbolic && iscall(ex)

function _has_differential(ex)
    ex = safe_unwrap(ex)
    _issym(ex) || return false
    operation(ex) isa Differential && return true
    return any(_has_differential, arguments(ex))
end

# The order of the highest derivative with respect to `t` appearing in `ex`.
function _time_deriv_order(ex, t)
    ex = safe_unwrap(ex)
    _issym(ex) || return 0
    op = operation(ex)
    inner = maximum(a -> _time_deriv_order(a, t), arguments(ex); init = 0)
    return op isa Differential && isequal(op.x, t) ? inner + op.order : inner
end

# A derivative of a slice, `D(u[2:n-1])`, names the slice; the unknowns are its elements.
function _scalar_elements(ex)
    ex = safe_unwrap(ex)
    is_array_valued(ex) || return (ex,)
    return safe_unwrap.(vec(collect(Symbolics.scalarize(Symbolics.wrap(ex)))))
end

function _collect_differentiated!(out, ex, unks)
    ex = safe_unwrap(ex)
    _issym(ex) || return out
    if operation(ex) isa Differential
        for arg in arguments(ex), v in _scalar_elements(arg)
            v in unks && push!(out, v)
        end
    end
    for arg in arguments(ex)
        _collect_differentiated!(out, arg, unks)
    end
    return out
end

# The unknowns of `sys` that appear under a differential; the rest are algebraic.
function differential_unknowns(sys, unks)
    out = Set{Any}()
    for eq in get_eqs(sys)
        _collect_differentiated!(out, eq.lhs, unks)
        _collect_differentiated!(out, eq.rhs, unks)
    end
    return out
end

# `unknowns` of a system built from array equations are the scalar elements, but an
# expression may mention the parent array or a slice of it, so check both.
function _mentions_unknown(ex, unks, arrays)
    for v in Symbolics.get_variables(ex)
        v = safe_unwrap(v)
        (v in unks || v in arrays) && return true
        if iscall(v) && operation(v) === getindex &&
                safe_unwrap(first(arguments(v))) in arrays
            return true
        end
    end
    return false
end

"""
    brown_init_offenders(sys)

Return the initialization equations of `sys` that `BrownFullBasicInit()` would not honor,
each paired with the reason, and an empty vector when the algorithm is safe for `sys`.

`BrownFullBasicInit()` holds the differential variables at their given values and solves
for everything else, so an initialization equation survives it only when it fixes a single
differential unknown to a value involving no other unknown. Everything else is reported,
including equations this predicate cannot classify.
"""
function brown_init_offenders(sys)
    ieqs = initialization_equations(sys)
    offenders = Pair{Equation, String}[]
    isempty(ieqs) && return offenders
    unks = Set{Any}(safe_unwrap.(get_unknowns(sys)))
    diffvars = differential_unknowns(sys, unks)
    arrays = Set{Any}(
        safe_unwrap(first(arguments(u)))
            for u in unks if iscall(u) && operation(u) === getindex
    )
    for eq in ieqs
        reason = _brown_init_offence(eq, diffvars, unks, arrays)
        reason === nothing || push!(offenders, eq => reason)
    end
    return offenders
end

function _brown_init_offence(eq, diffvars, unks, arrays)
    if _has_differential(eq.lhs) || _has_differential(eq.rhs)
        return "constrains a time derivative, which BrownFullBasicInit solves for."
    end
    lhs = safe_unwrap(eq.lhs)
    if !(lhs in unks)
        return "does not fix a single unknown of the discretized system, so it cannot be checked against what BrownFullBasicInit holds fixed."
    end
    if !(lhs in diffvars)
        return "constrains $lhs, which is algebraic rather than differential; BrownFullBasicInit solves for algebraic variables instead of honoring a given value."
    end
    if _mentions_unknown(eq.rhs, unks, arrays)
        return "relates $lhs to other unknowns; BrownFullBasicInit fixes each differential unknown independently and does not enforce the relation."
    end
    return nothing
end

# `varmap_to_vars` needs a value for every unknown and every derivative of an unknown. The
# discretized initial conditions cover the grid points of every variable given one; the
# rest are quantities the initialization solves for, for which this is only a guess.
function _dae_operating_point(sys, u0, t)
    D = Differential(t)
    op = Dict{Any, Any}()
    for x in get_unknowns(sys)
        x = safe_unwrap(x)
        op[x] = 0.0
        op[D(x)] = 0.0
    end
    for (k, v) in u0
        op[safe_unwrap(k)] = v
    end
    return op
end

"""
    DAEProblem(pdesys::PDESystem, discretization::MOLFiniteDifference; kwargs...)

Discretize `pdesys` and build a `DAEProblem` from the residuals MethodOfLines emits,
without running `mtkcompile`. The array equations reach the generated code intact,
which `mtkcompile` would undo.

`initializealg` defaults to `BrownFullBasicInit()`, which is the only algorithm that
reproduces the [`discretize`](@ref) result here, and is chosen only when the discretized
system's initialization equations are ones it honors; otherwise a
[`BrownFullBasicInitUnsafeError`](@ref) is raised naming the offending equations. Passing
`initializealg` explicitly overrides both the default and that check.

The solution is a `PDETimeSeriesSolution`, the same wrapper [`discretize`](@ref)
produces, so it is indexed and interpolated by the `PDESystem`'s own variables:
`sol[u(t, x)]`, `sol(t, x)`.
"""
function SciMLBase.DAEProblem(
        pdesys::PDESystem, discretization::MOLFiniteDifference;
        initializealg = nothing, build_initializeprob = false, kwargs...
    )
    sys, tspan = SciMLBase.symbolic_discretize(pdesys, discretization)
    if tspan === nothing
        throw(
            ArgumentError(
                "`DAEProblem` requires a time variable; pass one to `MOLFiniteDifference`, or use `discretize` for the `NonlinearProblem` a time-independent system discretizes to."
            )
        )
    end
    return _dae_problem(
        sys, tspan, discretization; initializealg, build_initializeprob, kwargs...
    )
end

"""
    _dae_problem(sys, tspan, discretization; kwargs...)

Build the `DAEProblem` from an already discretized system. Shared by
`DAEProblem(::PDESystem, ::MOLFiniteDifference)` and [`discretize`](@ref) so that the
system is discretized once.
"""
function _dae_problem(
        sys, tspan, discretization::MOLFiniteDifference;
        initializealg = nothing, build_initializeprob = false, kwargs...
    )
    sys = complete(sys)
    if initializealg === nothing
        offenders = brown_init_offenders(sys)
        isempty(offenders) || throw(BrownFullBasicInitUnsafeError(offenders))
        initializealg = DiffEqBase.BrownFullBasicInit()
    end
    t = safe_unwrap(get_time(discretization))
    maxorder = maximum(
        eq -> max(_time_deriv_order(eq.lhs, t), _time_deriv_order(eq.rhs, t)),
        get_eqs(sys); init = 0
    )
    if maxorder > 1
        throw(
            ArgumentError(
                "The discretized system is of order $maxorder in $t. A `DAEProblem` is first order, and the order reduction that would fix this is part of `mtkcompile`, which scalarizes the array equations. Use `discretize(pdesys, discretization)`."
            )
        )
    end
    metadata = getmetadata(sys, ModelingToolkit.ProblemTypeCtx, nothing)
    op = _dae_operating_point(sys, metadata.u0, t)
    return DAEProblem(
        sys, op, tspan; initializealg, build_initializeprob,
        discretization.kwargs..., kwargs...
    )
end
