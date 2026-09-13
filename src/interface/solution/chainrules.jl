# Reverse-mode rules for the time series wrapper. `solve` builds the wrapper from the solver's
# solution inside the differentiated call, and the wrapper is indexed and interpolated by
# symbols; traced as they are, these go through symbolic code, and the interpolants carry no
# rule for their values. The rules below route every cotangent back to the solver's solution,
# whose rules SciMLSensitivity provides.

# Cotangent of `A.original_sol` for a cotangent `Δ` of the array `A[dv]`, the reverse of the
# reconstruction in `PDETimeSeriesSolution(sol, metadata)`: a field that the system observes
# as one array goes through that observed function, once per saved time; otherwise each grid
# point is an unknown, gathered from the state vector, or observed on its own.
function field_cotangent(config, A::SciMLBase.PDETimeSeriesSolution, dv, Δ)
    sol = A.original_sol
    discretespace = A.disc_data.discretespace
    discu = get(discretespace.discvars, dv, nothing)
    discu === nothing && throw(not_implemented(dv))
    time_first = isequal(arguments(safe_unwrap(dv))[1], discretespace.time)
    at_time(k) = time_first ? selectdim(Δ, 1, k) : selectdim(Δ, ndims(Δ), k)
    Δu = [zero(u) for u in sol.u]
    Δp = Ref{Any}(NoTangent())
    function observed_pullback!(f, k, δ)
        pullback = rrule_via_ad(config, f, sol.u[k], sol.prob.p, sol.t[k])[2]
        _, du, dp, _ = pullback(δ)
        du = unthunk(du)
        du isa AbstractZero || (Δu[k] .+= du)
        Δp[] = Δp[] + unthunk(dp)
        return nothing
    end
    lhs = array_observed_lhs(discu, ModelingToolkitBase.observed(sol.prob.f.sys))
    if lhs !== nothing
        f = SymbolicIndexingInterface.observed(sol, lhs)
        for k in eachindex(sol.u)
            δ = at_time(k)
            all(iszero, δ) && continue
            observed_pullback!(f, k, collect(δ))
        end
    else
        for I in CartesianIndices(discu)
            i = SymbolicIndexingInterface.variable_index(sol, discu[I])
            if i !== nothing
                for k in eachindex(sol.u)
                    Δu[k][i] += at_time(k)[I]
                end
            else
                f = SymbolicIndexingInterface.observed(sol, safe_unwrap(discu[I]))
                for k in eachindex(sol.u)
                    δ = at_time(k)[I]
                    iszero(δ) && continue
                    observed_pullback!(f, k, δ)
                end
            end
        end
    end
    prob_tangent = Δp[] isa AbstractZero ? NoTangent() : Tangent{typeof(sol.prob)}(; p = Δp[])
    return Tangent{typeof(sol)}(; u = Δu, prob = prob_tangent)
end

# The dependent variable behind an index: a dependent variable of the system or the name a
# complex-valued one was replaced by.
function indexed_dependent_variable(A::SciMLBase.PDETimeSeriesSolution, sym)
    idv = sym_to_index(sym, A.dvs)
    idv === nothing || return A.dvs[idv]
    for (u, replaced) in A.disc_data.discretespace.vars.replaced_vars
        isequal(safe_unwrap(sym), safe_unwrap(replaced)) && return Num(u)
    end
    return nothing
end

independent_variable(A::SciMLBase.PDETimeSeriesSolution, sym) = sym_to_index(sym, A.ivs) !== nothing

function not_implemented(sym)
    return ArgumentError("Reverse-mode differentiation of the solution is not implemented for $sym.")
end

# Cotangent of `A.original_sol` for a cotangent `Δ` of the array `A[sym]`. Complex-valued
# variables, reconstructed from their real and imaginary parts, are not covered.
function solution_cotangent(config, A::SciMLBase.PDETimeSeriesSolution, sym, Δ)
    dv = indexed_dependent_variable(A, sym)
    dv === nothing && throw(not_implemented(sym))
    return field_cotangent(config, A, dv, Δ)
end

function ChainRulesCore.rrule(
        config::ChainRulesCore.RuleConfig{>:ChainRulesCore.HasReverseMode},
        ::typeof(getindex), A::SciMLBase.PDETimeSeriesSolution{T, N, S, D},
        sym::Union{Num, Symbol}
    ) where {T, N, S, D <: MOLMetadata}
    y = A[sym]
    function getindex_pullback(Δ)
        Δ = unthunk(Δ)
        (Δ isa AbstractZero || independent_variable(A, sym)) &&
            return (NoTangent(), NoTangent(), NoTangent())
        Δsol = solution_cotangent(config, A, sym, Δ)
        return (NoTangent(), Tangent{typeof(A)}(; original_sol = Δsol), NoTangent())
    end
    return y, getindex_pullback
end

function ChainRulesCore.rrule(
        config::ChainRulesCore.RuleConfig{>:ChainRulesCore.HasReverseMode},
        ::typeof(getindex), A::SciMLBase.PDETimeSeriesSolution{T, N, S, D},
        sym::Union{Num, Symbol}, args...
    ) where {T, N, S, D <: MOLMetadata}
    y = A[sym, args...]
    Δargs = map(_ -> NoTangent(), args)
    function getindex_pullback(Δ)
        Δ = unthunk(Δ)
        (Δ isa AbstractZero || independent_variable(A, sym)) &&
            return (NoTangent(), NoTangent(), NoTangent(), Δargs...)
        Δfield = zero(A[sym])
        Δfield[args...] = Δ
        Δsol = solution_cotangent(config, A, sym, Δfield)
        return (NoTangent(), Tangent{typeof(A)}(; original_sol = Δsol), NoTangent(), Δargs...)
    end
    return y, getindex_pullback
end

# The cell of the linear interpolation along one axis that contains `c`, as the index of its
# lower knot, and the position of `c` in it; the same choice Interpolations makes.
function linear_cell(knots, c)
    i = clamp(searchsortedfirst(knots, c) - 1, firstindex(knots), lastindex(knots) - 1)
    return i, (c - knots[i]) / (knots[i + 1] - knots[i])
end

# Cotangent of `A.original_sol` for a cotangent `Δ` of the interpolation of `dv` at `args`,
# numbers or vectors in the order of the arguments of `dv`. The interpolation is linear along
# each axis, so the cotangent of a value goes to the corners of its cell with the weights of
# the evaluation, and from there back as for `A[dv]`.
function interpolation_cotangent(config, A::SciMLBase.PDETimeSeriesSolution, dv, args, Δ)
    itp = A.interp[dv]
    hasproperty(itp, :it) && itp.it isa Gridded{<:Linear} || throw(
        ArgumentError(
            "Reverse-mode differentiation of the interpolation is implemented for `Gridded(Linear())` only."
        )
    )
    knots = itp.knots
    points = map(a -> a isa Number ? (a,) : a, args)
    n = length(points)
    Δfield = zero(A.u[dv])
    for (k, J) in enumerate(Iterators.product(map(eachindex, points)...))
        δ = Δ[k]
        (δ === nothing || iszero(δ)) && continue
        cells = ntuple(d -> linear_cell(knots[d], points[d][J[d]]), n)
        for corner in Iterators.product(ntuple(_ -> (false, true), n)...)
            w = prod(d -> corner[d] ? cells[d][2] : 1 - cells[d][2], 1:n)
            Δfield[ntuple(d -> cells[d][1] + corner[d], n)...] += δ * w
        end
    end
    return field_cotangent(config, A, dv, Δfield)
end

function ChainRulesCore.rrule(
        config::ChainRulesCore.RuleConfig{>:ChainRulesCore.HasReverseMode},
        A::SciMLBase.PDETimeSeriesSolution{T, N, S, D},
        args::Vararg{Union{Num, Number, AbstractArray, Colon}}; dv = nothing
    ) where {T, N, S, D <: MOLMetadata}
    y = A(args...; dv)
    Δargs = map(_ -> NoTangent(), args)
    function call_pullback(Δ)
        Δ = unthunk(Δ)
        Δ isa AbstractZero && return (NoTangent(), Δargs...)
        grid_args = _grid_args(A, args, dv)
        Δsol = NoTangent()
        if dv === nothing
            for (dvi, Δi) in zip(A.dvs, Δ)
                (Δi === nothing || Δi isa AbstractZero) && continue
                Δsol = Δsol + interpolation_cotangent(
                    config, A, dvi, _dv_args(A, dvi, grid_args), Δi
                )
            end
        else
            cdv = indexed_dependent_variable(A, dv)
            cdv === nothing && throw(not_implemented(dv))
            Δsol = interpolation_cotangent(config, A, cdv, grid_args, Δ)
        end
        return (Tangent{typeof(A)}(; original_sol = Δsol), Δargs...)
    end
    return y, call_pullback
end

function ChainRulesCore.rrule(
        ::Type{SciMLBase.PDETimeSeriesSolution}, sol::SciMLBase.AbstractODESolution,
        metadata::MOLMetadata
    )
    A = SciMLBase.PDETimeSeriesSolution(sol, metadata)
    function wrap_pullback(ȳ)
        ȳ = unthunk(ȳ)
        ȳ isa AbstractZero && return (NoTangent(), NoTangent(), NoTangent())
        ȳ.u isa AbstractZero && ȳ.interp isa AbstractZero && ȳ.prob isa AbstractZero || throw(
            ArgumentError(
                "Reverse-mode differentiation reaches the PDE solution through indexing, `sol[u(t, x)]`, and interpolation, `sol(t, x)`; the other fields of the solution have no rule."
            )
        )
        return (NoTangent(), ȳ.original_sol, NoTangent())
    end
    return A, wrap_pullback
end
