# MMS advection convergence for WENO-5 on non-uniform grids.
# Kernel-level tests: test/Components/weno_nonuniform_{core,boundary}.jl.
# EOC vs nominal_h = (b-a)/(N-1); fixed-β refinement is self-similar.

using ModelingToolkit, MethodOfLines, Test, DomainSets
using OrdinaryDiffEqSSPRK: SSPRK33
using SciMLBase
using ModelingToolkit: Differential

const L = 2π
const α = 0.15
const tf = 0.05
const CFL = 0.01

const REFINEMENT_NS = [81, 161]
const SINH_STRETCH = 4.0
# ~22% per-cell wall stretching.
const TANH_STRETCH = 2.0

uniform_grid(a, b, n::Integer) = collect(range(a, b, length = n))

# Center-clustered: x'(ξ) ∝ cosh(βξ).
function sinh_grid(a, b, n::Integer; β = SINH_STRETCH)
    ξ = range(-1, 1, length = n)
    x = collect(a .+ (b - a) .* (sinh.(β .* ξ) ./ sinh(β) .+ 1) ./ 2)
    x[1] = a
    x[end] = b
    return x
end

# Wall-clustered: x'(ξ) ∝ sech²(βξ).
function tanh_grid(a, b, n::Integer; β = TANH_STRETCH)
    ξ = range(-1, 1, length = n)
    x = collect(a .+ (b - a) .* (tanh.(β .* ξ) ./ tanh(β) .+ 1) ./ 2)
    x[1] = a
    x[end] = b
    return x
end

# Cell-width weighted relative L2.
function rel_l2(u::AbstractVector, uref::AbstractVector, x::AbstractVector)
    w = [diff(x); x[end] - x[end - 1]]
    err = sqrt(sum(w .* abs2.(u .- uref)))
    ref = sqrt(sum(w .* abs2.(uref)))
    return err / max(ref, eps(eltype(u)))
end

nominal_h(a, b, n::Integer) = (b - a) / (n - 1)

min_cell_width(x::AbstractVector) = minimum(diff(x))

pairwise_eoc(errs, hs) =
    [log(errs[k] / errs[k + 1]) / log(hs[k] / hs[k + 1]) for k in 1:(length(errs) - 1)]

mms_u(x, t, v) = sin(2π * (x - v * t) / L) + α * sin(4π * (x - v * t) / L)

@parameters t x
@variables u(..)

# Discretize once; dt varies at solve time.
function build_advection_prob(xgrid; v)
    x0, xL = xgrid[1], xgrid[end]
    Dt = Differential(t)
    Dx = Differential(x)
    eq = Dt(u(t, x)) ~ -v * Dx(u(t, x))
    bcs = [
        u(0.0, x) ~ mms_u(x, 0.0, v),
        u(t, x0) ~ mms_u(x0, t, v),
        u(t, xL) ~ mms_u(xL, t, v),
    ]
    domains = [t ∈ Interval(0.0, tf), x ∈ Interval(x0, xL)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
    disc = MOLFiniteDifference([x => xgrid], t; advection_scheme = WENOScheme())
    return discretize(pdesys, disc)
end

function solve_error(prob, xgrid; v, c = CFL)
    dt = c * min_cell_width(xgrid) / abs(v)
    sol = solve(prob, SSPRK33(); dt, saveat = [tf], adaptive = false)
    @test SciMLBase.successful_retcode(sol)
    xs = sol[x]
    return rel_l2(sol[u(t, x)][end, :], mms_u.(xs, tf, v), xs)
end

@testset "WENO Pipeline Spatial Convergence (MMS)" begin
    a, b = 0.0, L
    v = 1.0
    hs_nom = nominal_h.(a, b, REFINEMENT_NS)

    @testset "Uniform grid baseline (WENOScheme)" begin
        errs = map(REFINEMENT_NS) do N
            xgrid = uniform_grid(a, b, N)
            solve_error(build_advection_prob(xgrid; v), xgrid; v)
        end
        # Calibration: EOC ≈ 3.85.
        @test pairwise_eoc(errs, hs_nom)[1] > 3.75
    end

    sinh_grids = [sinh_grid(a, b, N) for N in REFINEMENT_NS]
    sinh_probs = [build_advection_prob(g; v) for g in sinh_grids]
    sinh_errs = [solve_error(p, g; v) for (p, g) in zip(sinh_probs, sinh_grids)]

    @testset "Non-uniform sinh clustering (center-clustered)" begin
        # Calibration: EOC ≈ 2.45, pre-asymptotic at N ≤ 161.
        @test pairwise_eoc(sinh_errs, hs_nom)[1] > 2.2
        @test pairwise_eoc(sinh_errs, min_cell_width.(sinh_grids))[1] > 2.2
    end

    @testset "Temporal error isolation (shared discretization)" begin
        # Calibration: spread ≈ 7e-11.
        errs_t = [
            sinh_errs[end],
            solve_error(sinh_probs[end], sinh_grids[end]; v, c = CFL / 4),
            solve_error(sinh_probs[end], sinh_grids[end]; v, c = CFL / 16),
        ]
        spread = (maximum(errs_t) - minimum(errs_t)) / (sum(errs_t) / length(errs_t))
        @test spread < 0.02
    end

    @testset "Reversed wind (v = -1) on sinh grid" begin
        # Scheme is sign-agnostic. Calibration: err_neg / err_pos ≈ 1.002.
        xg = sinh_grids[1]
        err_neg = solve_error(build_advection_prob(xg; v = -1.0), xg; v = -1.0)
        @test err_neg < 1.5 * sinh_errs[1]
    end

    @testset "Wall-clustered tanh grid (boundary stencils under stretching)" begin
        # Error-band regression, not EOC. Calibration: err ≈ 9.4e-6.
        N = REFINEMENT_NS[1]
        xg = tanh_grid(a, b, N)
        err = solve_error(build_advection_prob(xg; v), xg; v)
        @test err < 2.0e-5
    end
end

# Mixed WENO + centered diffusion holds a viscous shock on a clustered NU grid.
@testset "Mixed WENO advection + centered diffusion holds a viscous shock layer" begin
    ν = 2.0e-3
    t_end = 1.0
    u_inf(ξ) = -tanh(ξ / (2 * ν))

    ρ(ξ) = 1 + 30 * exp(-ξ^2 / (2 * 0.02^2))
    xsamp = range(-1.0, 1.0, length = 5001)
    cdf = cumsum(ρ.(xsamp))
    cdf = (cdf .- cdf[1]) ./ (cdf[end] - cdf[1])
    xg = map(range(0, 1, length = 129)) do l
        k = searchsortedfirst(cdf, l)
        k <= 1 && return float(xsamp[1])
        θ = (l - cdf[k - 1]) / (cdf[k] - cdf[k - 1])
        return xsamp[k - 1] + θ * (xsamp[k] - xsamp[k - 1])
    end
    xg[1], xg[end] = -1.0, 1.0
    @assert all(diff(xg) .> 0)

    Dt = Differential(t)
    Dx = Differential(x)
    eq = Dt(u(t, x)) ~ -u(t, x) * Dx(u(t, x)) + ν * Dx(Dx(u(t, x)))
    bcs = [
        u(0.0, x) ~ u_inf(x),
        u(t, -1.0) ~ 1.0,
        u(t, 1.0) ~ -1.0,
    ]
    domains = [t ∈ Interval(0.0, t_end), x ∈ Interval(-1.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
    disc = MOLFiniteDifference([x => xg], t; advection_scheme = WENOScheme())
    prob = discretize(pdesys, disc)

    dxmin = min_cell_width(xg)
    dt = 0.2 * min(dxmin, dxmin^2 / (2 * ν))
    sol = solve(prob, SSPRK33(); dt, adaptive = false, saveat = [t_end])
    @test SciMLBase.successful_retcode(sol)
    xs = sol[x]
    uend = sol[u(t, x)][end, :]
    @test all(isfinite, uend)

    @test rel_l2(uend, u_inf.(xs), xs) < 3.0e-4
    @test maximum(abs, uend) <= 1 + 1.0e-3
    i0 = findfirst(k -> uend[k] * uend[k + 1] < 0, 1:(length(uend) - 1))
    @test i0 !== nothing
    x0 = xs[i0] - uend[i0] * (xs[i0 + 1] - xs[i0]) / (uend[i0 + 1] - uend[i0])
    @test abs(x0) < dxmin
end
