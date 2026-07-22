# Work-precision comparisons. spatial_workprecision: L2 error vs minimum solve time over
# a node-count sweep per grid class. temporal_workprecision: DiffEqDevTools tolerance
# sweeps on a fixed grid, verifying identical temporal behavior across code paths.

using BenchmarkTools
using DiffEqDevTools
using OrdinaryDiffEq
using OrdinaryDiffEqSSPRK
using SciMLBase
using Statistics

const WP_ABSTOL = 1.0e-10
const WP_RELTOL = 1.0e-10

function _accuracy_solve(prob, t_end)
    return solve(prob, Tsit5(); abstol = WP_ABSTOL, reltol = WP_RELTOL, saveat = [t_end])
end

"""
    spatial_workprecision(sys, mkdisc; ns, kinds, samples, seconds)

Per (kind, n): quadrature-weighted L2 error at `t_end` and minimum solve time over
`samples` runs, for `sys` discretized via `mkdisc(sys, kind, n)`. The zero-argument
method runs the periodic-advection sweep.
"""
function spatial_workprecision(; ns = (41, 81, 161), kinds = GRID_KINDS, kwargs...)
    return spatial_workprecision(
        advection_system(), advection_discretization; ns, kinds, kwargs...
    )
end

function spatial_workprecision(
        sys, mkdisc; ns = (41, 81, 161), kinds = GRID_KINDS, samples = 5, seconds = 60
    )
    rows = NamedTuple[]
    for kind in kinds, n in ns
        prob = discretize(sys.pdesys, mkdisc(sys, kind, n))

        sol = _accuracy_solve(prob, sys.t_end)
        @assert SciMLBase.successful_retcode(sol) "solve failed for $kind N=$n"
        xg = sol[sys.xvar]
        usol = sol[sys.uvar(sys.tvar, sys.xvar)][end, :]
        @assert all(isfinite, usol) "non-finite solution for $kind N=$n"
        err = l2_error(usol, sys.exact.(xg, sys.t_end), xg)

        bench = @benchmarkable _accuracy_solve($prob, $(sys.t_end)) evals = 1
        trial = run(bench; samples, seconds)
        tmin = minimum(trial)
        gc_frac = mean(trial.gctimes ./ trial.times)

        push!(
            rows, (;
                kind, n, l2 = err, time_ns = tmin.time,
                allocs = tmin.allocs, memory = tmin.memory, gc_frac,
            )
        )
    end
    return rows
end

"""
    eoc_table(rows)

Observed convergence orders per grid class; assumes interval counts double between
consecutive sweep levels.
"""
function eoc_table(rows)
    table = NamedTuple[]
    for kind in unique(r.kind for r in rows)
        sub = sort([r for r in rows if r.kind == kind]; by = r -> r.n)
        for k in 1:(length(sub) - 1)
            push!(
                table, (;
                    kind, n_coarse = sub[k].n, n_fine = sub[k + 1].n,
                    order = log2(sub[k].l2 / sub[k + 1].l2),
                )
            )
        end
    end
    return table
end

"""
    equal_error_speedup(rows; base, contender, npts = 7)

Log-log interpolation of solve time vs L2 error over the common error range of two grid
classes; `speedup = t_base / t_contender` at each sampled level.
"""
function equal_error_speedup(rows; base = :uniform, contender = :front_adapted, npts = 7)
    function curve(kind)
        sub = sort([r for r in rows if r.kind == kind]; by = r -> r.l2)
        return (log10.([r.l2 for r in sub]), log10.([Float64(r.time_ns) for r in sub]))
    end
    function interp(xs, ys, x)
        j = clamp(searchsortedlast(xs, x), 1, length(xs) - 1)
        θ = (x - xs[j]) / (xs[j + 1] - xs[j])
        return ys[j] + θ * (ys[j + 1] - ys[j])
    end

    xb, yb = curve(base)
    xc, yc = curve(contender)
    (length(xb) < 2 || length(xc) < 2) && return NamedTuple[]
    lo = max(minimum(xb), minimum(xc))
    hi = min(maximum(xb), maximum(xc))
    lo < hi || return NamedTuple[]

    out = NamedTuple[]
    for ℓ in range(lo, hi; length = npts)
        tb = 10.0^interp(xb, yb, ℓ)
        tc = 10.0^interp(xc, yc, ℓ)
        push!(out, (; l2 = 10.0^ℓ, t_base_ns = tb, t_contender_ns = tc, speedup = tb / tc))
    end
    return out
end

"""
    temporal_workprecision(; n, kinds)

`WorkPrecisionSet` per grid class with identical solver/tolerance sweeps; reference is a
tight-tolerance Vern9 solve in a `TestSolution`.
"""
function temporal_workprecision(;
        n = 161, kinds = (:uniform, :uniform_vector, :stretched), numruns = 5
    )
    sys = advection_system()
    abstols = 1.0 ./ 10.0 .^ (5:2:11)
    reltols = 1.0 ./ 10.0 .^ (2:2:8)
    setups = [Dict(:alg => Tsit5()), Dict(:alg => SSPRK43())]
    names = ["Tsit5", "SSPRK43"]

    wps = Pair{Symbol, WorkPrecisionSet}[]
    for kind in kinds
        prob = discretize(sys.pdesys, advection_discretization(sys, kind, n))
        # wrap = Val(false): appxtrue rejects MOL's PDETimeSeriesSolution wrapper.
        ref = solve(prob, Vern9(); abstol = 1.0e-13, reltol = 1.0e-13, wrap = Val(false))
        @assert SciMLBase.successful_retcode(ref)
        wp = WorkPrecisionSet(
            prob, abstols, reltols, setups;
            appxsol = TestSolution(ref), save_everystep = false, wrap = Val(false),
            error_estimate = :final, numruns, names,
        )
        push!(wps, kind => wp)
    end
    return wps
end
