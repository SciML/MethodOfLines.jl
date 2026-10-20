# Reverse-mode hooks behind the rules SciMLBase provides for `sol[sym]`, `sol[sym, inds...]`
# and `sol(args...; dv)`. `solve` builds the wrapper from the solver's solution inside the
# differentiated call, and the wrapper is indexed and interpolated by symbols; the hooks map a
# cotangent of a field or of an interpolation back to the solver's solution, the reverse of
# `PDETimeSeriesSolution(sol, metadata)` and `PDENoTimeSolution(sol, metadata)`.

# Cotangent of `A.original_sol` for a cotangent `Δ` of the array `A[dv]`. A grid point that is
# an unknown is gathered from the state vector by its index; the others, such as eliminated
# boundary points, go through one observed function of all of them.
function field_cotangent(A::SciMLBase.AbstractPDESolution, dv, Δ)
    sol = A.original_sol
    odesys = sol.prob.f.sys
    discu = get(A.disc_data.discretespace.discvars, dv, nothing)
    discu === nothing && throw(not_implemented(dv))
    idxs = grid_indices(odesys, discu)
    obs = findall(isnothing, idxs)
    f = isempty(obs) ? nothing : grid_observed_function(sol, discu, obs)
    return state_cotangent(A, dv, Δ, idxs, obs, f)
end

# The observed function of the elements `obs` of `discu`, valued in floats: a constant point,
# such as a homogeneous Dirichlet boundary, evaluates to an integer, which has no tangent.
function grid_observed_function(sol, discu, obs)
    g = SymbolicIndexingInterface.observed(sol, [safe_unwrap(discu[I]) for I in obs])
    return (args...) -> float.(g(args...))
end

zero_states(A::SciMLBase.PDETimeSeriesSolution) = [zero(u) for u in A.original_sol.u]
zero_states(A::SciMLBase.PDENoTimeSolution) = zero(A.original_sol.u)

function scatter_states!(Δu, Δ, idxs)
    for I in CartesianIndices(idxs)
        i = idxs[I]
        i === nothing || (Δu[i] += Δ[I])
    end
    return Δu
end

# One state cotangent and one observed term `(f, k, δ)` per saved time, with time the first or
# the last argument of `dv`.
function state_cotangent(A::SciMLBase.PDETimeSeriesSolution, dv, Δ, idxs, obs, f)
    time_first = isequal(arguments(safe_unwrap(dv))[1], A.disc_data.discretespace.time)
    Δu = zero_states(A)
    observed = Tuple{Any, Int, Any}[]
    for k in eachindex(Δu)
        Δk = time_first ? selectdim(Δ, 1, k) : selectdim(Δ, ndims(Δ), k)
        scatter_states!(Δu[k], Δk, idxs)
        δ = [Δk[I] for I in obs]
        any(!iszero, δ) && push!(observed, (f, k, δ))
    end
    return (; u = Δu, observed)
end

function state_cotangent(A::SciMLBase.PDENoTimeSolution, dv, Δ, idxs, obs, f)
    Δu = scatter_states!(zero_states(A), Δ, idxs)
    observed = Tuple{Any, Any}[]
    δ = [Δ[I] for I in obs]
    any(!iszero, δ) && push!(observed, (f, δ))
    return (; u = Δu, observed)
end

# The dependent variable behind an index: a dependent variable of the system, or the flattened
# name of an array-indexed one, `u[1](t, x)`. A complex-valued field, typed or not, is stored as
# two real ones and has no rule.
function indexed_dependent_variable(A::SciMLBase.AbstractPDESolution, sym)
    sym isa Complex && return nothing
    idv = sym_to_index(sym, A.dvs)
    idv === nothing || return A.dvs[idv]
    for (u, replaced) in A.disc_data.discretespace.vars.replaced_vars
        isequal(safe_unwrap(sym), safe_unwrap(replaced)) && return Num(u)
    end
    return nothing
end

independent_variable(A::SciMLBase.AbstractPDESolution, sym) = sym_to_index(sym, A.ivs) !== nothing

function not_implemented(sym)
    return ArgumentError("Reverse-mode differentiation of the solution is not implemented for $sym.")
end

# `sol[sym]` and `sol[sym, inds...]`: the grid has no cotangent, a field goes back to the
# solver's solution. Complex-valued variables, reconstructed from their real and imaginary
# parts, are not covered.
function SciMLBase.pde_index_cotangent(
        A::SciMLBase.AbstractPDESolution{T, N, S, D}, sym, Δ
    ) where {T, N, S, D <: MOLMetadata}
    sym isa Union{Num, Symbol} || throw(not_implemented(sym))
    independent_variable(A, sym) && return nothing
    dv = indexed_dependent_variable(A, sym)
    dv === nothing && throw(not_implemented(sym))
    return field_cotangent(A, dv, Δ)
end

# The cell of the linear interpolation along one axis that contains `c`, as the index of its
# lower knot, and the position of `c` in it; the same choice Interpolations makes.
function linear_cell(knots, c)
    i = clamp(searchsortedfirst(knots, c) - 1, firstindex(knots), lastindex(knots) - 1)
    return i, (c - knots[i]) / (knots[i + 1] - knots[i])
end

# Cotangents of `A.original_sol` and of the evaluation points for a cotangent `Δ` of the
# interpolation of `dv` at `points`, numbers or vectors in the order of the arguments of `dv`.
# The interpolation is linear along each axis, so the cotangent of a value goes to the corners
# of its cell with the weights of the evaluation, and from there back as for `A[dv]`; the
# cotangent of a coordinate is the slope of the interpolation there.
function interpolation_cotangent(A::SciMLBase.AbstractPDESolution, dv, points, Δ)
    itp = A.interp[dv]
    hasproperty(itp, :it) && itp.it isa Gridded{<:Linear} || throw(
        ArgumentError(
            "Reverse-mode differentiation of the interpolation is implemented for `Gridded(Linear())` only."
        )
    )
    knots = itp.knots
    points = map(p -> p isa Number ? (p,) : p, points)
    n = length(points)
    Δfield = zero(A.u[dv])
    Δpoints = map(p -> zeros(eltype(Δfield), length(p)), points)
    for (k, J) in enumerate(Iterators.product(map(eachindex, points)...))
        δ = Δ[k]
        δ isa Number && !iszero(δ) || continue
        point = ntuple(d -> points[d][J[d]], n)
        cells = ntuple(d -> linear_cell(knots[d], point[d]), n)
        for corner in Iterators.product(ntuple(_ -> (false, true), n)...)
            w = prod(d -> corner[d] ? cells[d][2] : 1 - cells[d][2], 1:n)
            Δfield[ntuple(d -> cells[d][1] + corner[d], n)...] += δ * w
        end
        slope = Interpolations.gradient(itp, point...)
        for d in 1:n
            Δpoints[d][J[d]] += δ * slope[d]
        end
    end
    return field_cotangent(A, dv, Δfield), Δpoints
end

_add_cotangent(::Nothing, Δ) = Δ
_add_cotangent(a, Δ) = a + Δ

# `sol(args...; dv)`: numbers, vectors and colons in the order of the arguments of `dv`, or of
# `sol.ivs` when `dv` is not given and every dependent variable is interpolated.
function SciMLBase.pde_call_cotangent(
        A::SciMLBase.AbstractPDESolution{T, N, S, D}, args, dv, Δ
    ) where {T, N, S, D <: MOLMetadata}
    Δargs = Any[nothing for _ in args]
    # The cotangents of the arguments of a dependent variable, back at their positions.
    function scatter!(Δpoints, positions)
        for (d, i) in enumerate(positions)
            args[i] isa Colon && continue
            Δargs[i] = _add_cotangent(
                Δargs[i], args[i] isa Number ? Δpoints[d][1] : Δpoints[d]
            )
        end
        return nothing
    end
    if dv === nothing
        grid_args = _grid_args(A, args)
        Δu = zero_states(A)
        observed = Any[]
        for (dvi, Δi) in zip(A.dvs, Δ)
            # An entry without a cotangent.
            Δi isa Union{Number, AbstractArray} || continue
            c, Δpoints = interpolation_cotangent(
                A, dvi, _dv_args(A, dvi, grid_args), Δi
            )
            Δu .+= c.u
            append!(observed, c.observed)
            scatter!(Δpoints, _iv_indices(A, dvi))
        end
        return (; u = Δu, observed), Tuple(Δargs)
    end
    cdv = indexed_dependent_variable(A, dv)
    cdv === nothing && throw(not_implemented(dv))
    c, Δpoints = interpolation_cotangent(A, cdv, _grid_args(A, args, cdv), Δ)
    scatter!(Δpoints, eachindex(args))
    return c, Tuple(Δargs)
end
