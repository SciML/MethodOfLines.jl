using Test
using LinearAlgebra
using ForwardDiff
using Symbolics
using MethodOfLines

const Dual = ForwardDiff.Dual

const weno_f_nonuniform = MethodOfLines.weno_f_nonuniform
const weno_f_uniform = MethodOfLines.weno_f_uniform
const compute_beta_nonuniform = MethodOfLines.compute_beta_nonuniform
const ideal_weno_linear_weights = MethodOfLines.ideal_weno_linear_weights
const ideal_weights_lagrange = MethodOfLines.ideal_weights_lagrange
const shi_hu_shu_weights = MethodOfLines.shi_hu_shu_weights
const _stencil_taylor_moment3 = MethodOfLines._stencil_taylor_moment3
const _stencil_taylor_moment4 = MethodOfLines._stencil_taylor_moment4

const WENO_EPS = 1.0e-14
const WENO_PARAMS = [WENO_EPS]

function uniform_stencil(h::Real)
    return collect(0.0:h:(4h))
end

function stretched_stencil(x0::Real, h::Real, α::Real)
    return [x0 - h * (1 + α), x0 - h * α, x0, x0 + h * α, x0 + h * (1 + α)]
end

function weno_derivative(f, xv; ε = WENO_EPS)
    u = f.(xv)
    return weno_f_nonuniform(u, [ε], 0.0, xv, diff(xv))
end

function jiang_shu_beta(u1, u2, u3, stencil::Int)
    if stencil == 1
        return 13 * (u1 - 2 * u2 + u3)^2 / 12 + (3 * u1 - 4 * u2 + u3)^2 / 4
    elseif stencil == 2
        return 13 * (u1 - 2 * u2 + u3)^2 / 12 + (u1 - u3)^2 / 4
    else
        return 13 * (u1 - 2 * u2 + u3)^2 / 12 + (u1 - 4 * u2 + 3 * u3)^2 / 4
    end
end

function face_geometry(xv)
    x_face_l = (xv[2] + xv[3]) / 2
    x_face_r = (xv[3] + xv[4]) / 2
    return x_face_l, x_face_r
end

function left_face_linear_weights(xv)
    x_face_l, _ = face_geometry(xv)
    return ideal_weno_linear_weights(
        xv[3], xv[4], xv[5], xv[2], xv[3], xv[4], xv[1], xv[2], xv[3], x_face_l,
    )
end

function left_face_betas(u, xv)
    x_face_l, x_face_r = face_geometry(xv)
    return (
        compute_beta_nonuniform(u[3], u[4], u[5], xv[3], xv[4], xv[5], x_face_l, x_face_r),
        compute_beta_nonuniform(u[2], u[3], u[4], xv[2], xv[3], xv[4], x_face_l, x_face_r),
        compute_beta_nonuniform(u[1], u[2], u[3], xv[1], xv[2], xv[3], x_face_l, x_face_r),
    )
end

function left_face_nonlinear_weights(u, xv; ε = WENO_EPS)
    d = left_face_linear_weights(xv)
    β = left_face_betas(u, xv)
    return shi_hu_shu_weights(d..., β..., ε)
end

function halving_order(errors)
    rates = [log(errors[i - 1] / errors[i]) / log(2) for i in 2:length(errors)]
    return minimum(rates), sum(rates) / length(rates)
end

function promote_dual_stencil(u, xv)
    du = Dual{1}.(u, ones(length(u)))
    dx = Dual{1}.(xv, zeros(length(xv)))
    xL, xR = face_geometry(xv)
    return du, dx, Dual{1}(xL, 0.0), Dual{1}(xR, 0.0)
end

function evaluate_symbolic(expr, subs)
    s = Symbolics.substitute(expr, subs)
    s = Symbolics.expand(s)
    s = Symbolics.simplify(s)
    return Symbolics.value(s)
end

@testset "Non-uniform WENO — type stability (@inferred)" begin
    h = 0.1
    xv = uniform_stencil(h)
    u = xv .^ 3
    x_face_l, x_face_r = face_geometry(xv)

    @test @inferred(
        compute_beta_nonuniform(
            u[3], u[4], u[5], xv[3], xv[4], xv[5], x_face_l, x_face_r,
        )
    ) ≈ jiang_shu_beta(u[3], u[4], u[5], 1)

    w_lag = @inferred ideal_weights_lagrange(xv[3], xv[4], xv[5], x_face_l)
    @test sum(w_lag) ≈ 1.0

    d = @inferred ideal_weno_linear_weights(
        xv[3], xv[4], xv[5], xv[2], xv[3], xv[4], xv[1], xv[2], xv[3], x_face_l,
    )
    @test sum(d) ≈ 1.0

    β = left_face_betas(u, xv)
    ω = @inferred shi_hu_shu_weights(d..., β..., WENO_EPS)
    @test sum(ω) ≈ 1.0

    @test @inferred(weno_f_nonuniform(u, WENO_PARAMS, 0.0, xv, diff(xv))) isa Float64
end

@testset "Non-uniform WENO — zero allocation hot paths" begin
    h = 0.1
    xv = uniform_stencil(h)
    u = sin.(xv)
    dx_vec = diff(xv)
    p = WENO_PARAMS
    t = 0.0
    x_face_l, x_face_r = face_geometry(xv)

    weno_f_nonuniform(u, p, t, xv, dx_vec)
    compute_beta_nonuniform(u[3], u[4], u[5], xv[3], xv[4], xv[5], x_face_l, x_face_r)
    ideal_weno_linear_weights(
        xv[3], xv[4], xv[5], xv[2], xv[3], xv[4], xv[1], xv[2], xv[3], x_face_l,
    )
    shi_hu_shu_weights(0.1, 0.6, 0.3, 1.0e-8, 1.0e-8, 1.0e-8, WENO_EPS)

    @test @allocated(weno_f_nonuniform(u, p, t, xv, dx_vec)) == 0
    @test @allocated(
        compute_beta_nonuniform(
            u[3], u[4], u[5], xv[3], xv[4], xv[5], x_face_l, x_face_r,
        )
    ) == 0
    @test @allocated(
        ideal_weno_linear_weights(
            xv[3], xv[4], xv[5], xv[2], xv[3], xv[4], xv[1], xv[2], xv[3], x_face_l,
        )
    ) == 0
    @test @allocated(shi_hu_shu_weights(0.1, 0.6, 0.3, 1.0e-8, 1.0e-8, 1.0e-8, WENO_EPS)) == 0

    x_view = @view xv[1:5]
    weno_f_nonuniform(u, p, t, x_view, dx_vec)
    @test @allocated(weno_f_nonuniform(u, p, t, x_view, dx_vec)) == 0
end

@testset "Non-uniform WENO — Lagrange intra-stencil weights" begin
    grids = (
        uniform_stencil(0.1),
        stretched_stencil(1.0, 0.08, 0.35),
        [0.0, 1.0, 1.0 + 1.0e-6, 1.0 + 2.0e-6, 100.0],
    )
    for xv in grids
        for x_eval in face_geometry(xv)
            for (x1, x2, x3) in ((xv[3], xv[4], xv[5]), (xv[2], xv[3], xv[4]), (xv[1], xv[2], xv[3]))
                w = ideal_weights_lagrange(x1, x2, x3, x_eval)
                @test sum(w) ≈ 1.0
                @test w[1] * x1 + w[2] * x2 + w[3] * x3 ≈ x_eval atol = 1.0e-12
                for (j, xj) in enumerate((x1, x2, x3))
                    wj = ideal_weights_lagrange(x1, x2, x3, xj)
                    for k in 1:3
                        @test wj[k] ≈ (j == k ? 1.0 : 0.0) atol = 1.0e-12
                    end
                end
            end
        end
    end
end

@testset "Non-uniform WENO — ideal linear weights Taylor constraints" begin
    moderate_grids = (
        uniform_stencil(0.1),
        stretched_stencil(0.75, 0.05, 0.4),
        [0.0, 1.0, 1.0 + 1.0e-3, 1.0 + 2.0e-3, 10.0],
    )
    for xv in moderate_grids
        for x_f in face_geometry(xv)
            d = ideal_weno_linear_weights(
                xv[3], xv[4], xv[5], xv[2], xv[3], xv[4], xv[1], xv[2], xv[3], x_f,
            )
            c3 = (
                _stencil_taylor_moment3(xv[3], xv[4], xv[5], x_f),
                _stencil_taylor_moment3(xv[2], xv[3], xv[4], x_f),
                _stencil_taylor_moment3(xv[1], xv[2], xv[3], x_f),
            )
            c4 = (
                _stencil_taylor_moment4(xv[3], xv[4], xv[5], x_f),
                _stencil_taylor_moment4(xv[2], xv[3], xv[4], x_f),
                _stencil_taylor_moment4(xv[1], xv[2], xv[3], x_f),
            )
            @test sum(d) ≈ 1.0
            @test dot(d, c3) ≈ 0.0 atol = 1.0e-12
            @test dot(d, c4) ≈ 0.0 atol = 1.0e-12
        end
    end

    h = 0.1
    xv = uniform_stencil(h)
    x_face_l, x_face_r = face_geometry(xv)
    dm = ideal_weno_linear_weights(
        xv[3], xv[4], xv[5], xv[2], xv[3], xv[4], xv[1], xv[2], xv[3], x_face_l,
    )
    dp = ideal_weno_linear_weights(
        xv[3], xv[4], xv[5], xv[2], xv[3], xv[4], xv[1], xv[2], xv[3], x_face_r,
    )
    @test dm[1] ≈ 1 / 16 atol = 1.0e-12
    @test dm[2] ≈ 5 / 8 atol = 1.0e-12
    @test dm[3] ≈ 5 / 16 atol = 1.0e-12
    @test dp[1] ≈ 5 / 16 atol = 1.0e-12
    @test dp[2] ≈ 5 / 8 atol = 1.0e-12
    @test dp[3] ≈ 1 / 16 atol = 1.0e-12
end

@testset "Non-uniform WENO — extreme grid Taylor conditioning (known limitation)" begin
    xv_neg = [0.0, 2.0, 2.000001, 1.000002000001e6, 1.000002000101e6]
    for x_f in face_geometry(xv_neg)
        d = ideal_weno_linear_weights(
            xv_neg[3], xv_neg[4], xv_neg[5], xv_neg[2], xv_neg[3], xv_neg[4],
            xv_neg[1], xv_neg[2], xv_neg[3], x_f,
        )
        c3 = (
            _stencil_taylor_moment3(xv_neg[3], xv_neg[4], xv_neg[5], x_f),
            _stencil_taylor_moment3(xv_neg[2], xv_neg[3], xv_neg[4], x_f),
            _stencil_taylor_moment3(xv_neg[1], xv_neg[2], xv_neg[3], x_f),
        )
        c4 = (
            _stencil_taylor_moment4(xv_neg[3], xv_neg[4], xv_neg[5], x_f),
            _stencil_taylor_moment4(xv_neg[2], xv_neg[3], xv_neg[4], x_f),
            _stencil_taylor_moment4(xv_neg[1], xv_neg[2], xv_neg[3], x_f),
        )
        @test sum(d) ≈ 1.0
        @test all(isfinite, d)
        scale = max(1.0, maximum(abs.(c4)))
        @test abs(dot(d, c3)) / scale < 1.0e-6
        @test abs(dot(d, c4)) / scale < 1.0e-6
    end
end

@testset "Non-uniform WENO — smoothness indicator β" begin
    h = 0.1
    xv = uniform_stencil(h)
    x_face_l, x_face_r = face_geometry(xv)
    u = [1.0, 2.0, 3.0, 4.0, 5.0]

    β1 = compute_beta_nonuniform(u[3], u[4], u[5], xv[3], xv[4], xv[5], x_face_l, x_face_r)
    β2 = compute_beta_nonuniform(u[2], u[3], u[4], xv[2], xv[3], xv[4], x_face_l, x_face_r)
    β3 = compute_beta_nonuniform(u[1], u[2], u[3], xv[1], xv[2], xv[3], x_face_l, x_face_r)

    @test β1 ≈ jiang_shu_beta(u[3], u[4], u[5], 1) atol = 1.0e-10
    @test β2 ≈ jiang_shu_beta(u[2], u[3], u[4], 2) atol = 1.0e-10
    @test β3 ≈ jiang_shu_beta(u[1], u[2], u[3], 3) atol = 1.0e-10

    u_const = fill(3.7, 5)
    β_const = left_face_betas(u_const, xv)
    @test all(x -> abs(x) < 1.0e-20, β_const)

    xv_nu = stretched_stencil(1.0, 0.08, 0.35)
    x_face_l_nu, x_face_r_nu = face_geometry(xv_nu)
    u_nu = sin.(xv_nu)
    β_nu = compute_beta_nonuniform(
        u_nu[3], u_nu[4], u_nu[5], xv_nu[3], xv_nu[4], xv_nu[5], x_face_l_nu, x_face_r_nu,
    )
    @test β_nu ≈ 0.0002295868746123596 atol = 1.0e-12
    @test β_nu >= 0.0

    u_jump = [0.0, 0.0, 0.0, 1.0, 1.0]
    β_jump = left_face_betas(u_jump, xv)
    β_smooth = left_face_betas(sin.(xv), xv)
    @test all(>=(0.0), β_jump)
    @test maximum(β_jump) > 100 * maximum(β_smooth)
end

@testset "Non-uniform WENO — Shi-Hu-Shu splitting" begin
    smooth_β = (1.0e-10, 1.0e-10, 1.0e-10)
    shock_β = (1.0e-10, 1.0, 1.0e-10)

    for d in ((0.1, 0.6, 0.3), (1 / 16, 5 / 8, 5 / 16), (-0.2, 0.9, 0.3))
        ω_s = shi_hu_shu_weights(d..., smooth_β..., WENO_EPS)
        ω_h = shi_hu_shu_weights(d..., shock_β..., WENO_EPS)
        @test sum(ω_s) ≈ 1.0 atol = 1.0e-12
        @test sum(ω_h) ≈ 1.0 atol = 1.0e-12
        @test all(isfinite, ω_s)
        @test all(isfinite, ω_h)
    end

    d_extreme = (-9.22, 9.22, 0.9999)
    ω_ext = shi_hu_shu_weights(d_extreme..., smooth_β..., WENO_EPS)
    @test sum(ω_ext) ≈ 1.0 atol = 1.0e-3
    @test all(isfinite, ω_ext)

    d_neg = (-0.2, 0.9, 0.3)
    ω_lin = shi_hu_shu_weights(d_neg..., smooth_β..., WENO_EPS)
    @test ω_lin[1] ≈ d_neg[1] atol = 1.0e-10
    @test ω_lin[2] ≈ d_neg[2] atol = 1.0e-10
    @test ω_lin[3] ≈ d_neg[3] atol = 1.0e-10

    xv_neg = [0.0, 2.0, 2.000001, 1.000002000001e6, 1.000002000101e6]
    x_face_r = (xv_neg[3] + xv_neg[4]) / 2
    dr = ideal_weno_linear_weights(
        xv_neg[3], xv_neg[4], xv_neg[5], xv_neg[2], xv_neg[3], xv_neg[4],
        xv_neg[1], xv_neg[2], xv_neg[3], x_face_r,
    )
    @test minimum(dr) < 0.0
    β = left_face_betas(ones(5), xv_neg)
    ω = shi_hu_shu_weights(dr..., β..., 1.0e-6)
    @test sum(ω) ≈ 1.0 atol = 1.0e-3
    @test all(isfinite, ω)

    flux = weno_f_nonuniform(ones(5), [1.0e-6], 0.0, xv_neg, diff(xv_neg))
    @test isfinite(flux)
    @test abs(flux) < 1.0
end

@testset "Non-uniform WENO — uniform grid 5th-order convergence" begin
    hs = [0.2, 0.1, 0.05, 0.025, 0.0125]
    f = x -> x^5
    errors = [abs(weno_derivative(f, uniform_stencil(h)) - 5 * (2h)^4) for h in hs]
    min_rate, mean_rate = halving_order(errors)
    @test min_rate > 3.8
    @test mean_rate > 4.0
    @test all(errors[i] > errors[i + 1] for i in 1:(length(errors) - 1))

    for (g, fp) in (
            (sin, h -> cos(2h)),
            (exp, h -> exp(2h)),
        )
        errors = [abs(weno_derivative(g, uniform_stencil(h)) - fp(h)) for h in hs]
        @test all(errors[i] > errors[i + 1] for i in 1:(length(errors) - 1))
        min_rate, _ = halving_order(errors)
        @test min_rate > 1.5
    end

    h = 0.1
    xv = uniform_stencil(h)
    u_linear = [1.0, 2.0, 3.0, 4.0, 5.0]
    flux_nu = weno_f_nonuniform(u_linear, [1.0e-6], 0.0, xv, diff(xv))
    flux_uni = weno_f_uniform(u_linear, [1.0e-6], 0.0, xv, h)
    @test flux_nu ≈ flux_uni rtol = 1.0e-10
end

@testset "Non-uniform WENO — stretched grid convergence" begin
    x0 = 1.0
    α = 0.35
    hs = [0.2, 0.1, 0.05, 0.025, 0.0125]
    f = x -> x^5
    fp = 5x0^4

    errors = [abs(weno_derivative(f, stretched_stencil(x0, h, α)) - fp) for h in hs]
    @test all(errors[i] > errors[i + 1] for i in 1:(length(errors) - 1))
    min_rate, _ = halving_order(errors)
    @test min_rate > 1.5
end

@testset "Non-uniform WENO — oscillatory smooth fields" begin
    h = 0.05
    xv = uniform_stencil(h)
    for ω in (5.0, 20.0, 50.0)
        f = x -> sin(ω * x)
        exact = ω * cos(ω * 2h)
        approx = weno_derivative(f, xv)
        @test isfinite(approx)
        @test abs(approx - exact) < 0.5 * ω
    end

    xv_stretch = stretched_stencil(0.5, 0.04, 0.5)
    f = x -> cos(12x) * exp(-x^2)
    exact = -12 * sin(12 * 0.5) * exp(-0.25) - 2 * 0.5 * cos(12 * 0.5) * exp(-0.25)
    approx = weno_derivative(f, xv_stretch)
    @test isfinite(approx)
    @test abs(approx - exact) < 5.0
end

@testset "Non-uniform WENO — shock capturing (ω vs d_k)" begin
    h = 0.1
    xv = uniform_stencil(h)
    u_smooth = sin.(xv)
    u_shock = [0.0, 0.0, 0.0, 1.0, 1.0]
    u_asymmetric = [0.0, 0.0, 0.25, 0.75, 1.0]
    u_osc = [0.0, 1.0, 0.0, 1.0, 0.0]

    d = left_face_linear_weights(xv)
    ω_smooth = left_face_nonlinear_weights(u_smooth, xv)
    ω_shock = left_face_nonlinear_weights(u_shock, xv)
    ω_osc = left_face_nonlinear_weights(u_osc, xv)

    β_smooth = left_face_betas(u_smooth, xv)
    β_shock = left_face_betas(u_shock, xv)
    @test maximum(β_shock) > 100 * maximum(β_smooth)
    @test sum(abs, ω_smooth .- d) < sum(abs, ω_shock .- d)
    @test sum(abs, ω_smooth .- d) < sum(abs, ω_osc .- d)
    @test argmax(ω_shock) == argmin(β_shock)
    @test ω_shock[argmin(β_shock)] > ω_shock[argmax(β_shock)]
    @test sum(abs, ω_shock .- d) > 0.05

    flux_asymmetric = weno_f_nonuniform(u_asymmetric, WENO_PARAMS, 0.0, xv, diff(xv))
    @test flux_asymmetric ≈ 3.8291328236980453 rtol = 1.0e-12
    @test abs(flux_asymmetric) > 0.1
end

@testset "Non-uniform WENO — Heaviside and extreme stretch" begin
    h = 0.01
    xv = uniform_stencil(h)
    u_step = [0.0, 0.0, 0.0, 1.0, 1.0]
    u_ramp = [0.0, 0.25, 0.5, 0.75, 1.0]

    for u in (u_step, u_ramp, [0.0, 0.0, 1.0, 1.0, 1.0])
        flux = weno_f_nonuniform(u, WENO_PARAMS, 0.0, xv, diff(xv))
        @test isfinite(flux)
        @test !isnan(flux)
    end

    xv_stretch = [0.0, 1.0, 1.0 + 1.0e-6, 1.0 + 2.0e-6, 100.0]
    flux_exp = weno_derivative(x -> exp(x), xv_stretch)
    @test isfinite(flux_exp)
    @test abs(flux_exp - exp(1.0 + 1.0e-6)) < 50.0

    ratio = 1.0e6
    xv_ratio = [0.0, 1.0, 1.0 + 1 / ratio, 1.0 + 2 / ratio, 1.0 + 3 / ratio]
    d = left_face_linear_weights(xv_ratio)
    @test all(d .> 0.0)
    @test sum(d) ≈ 1.0
    flux_ratio = weno_f_nonuniform(ones(5), WENO_PARAMS, 0.0, xv_ratio, diff(xv_ratio))
    @test isfinite(flux_ratio)
end

@testset "Non-uniform WENO — Float32 and promoted types" begin
    h = 0.1f0
    xv = collect(0.0f0:h:(4h))
    u = sin.(xv)
    dx_vec = diff(xv)
    p32 = [1.0f-6]
    t = 0.0f0

    flux = weno_f_nonuniform(u, p32, t, xv, dx_vec)
    @test flux isa Float32
    @test isfinite(flux)

    x_face_l, x_face_r = face_geometry(xv)
    β = compute_beta_nonuniform(u[3], u[4], u[5], xv[3], xv[4], xv[5], x_face_l, x_face_r)
    @test β isa Float32

    weno_f_nonuniform(u, p32, t, xv, dx_vec)
    compute_beta_nonuniform(u[3], u[4], u[5], xv[3], xv[4], xv[5], x_face_l, x_face_r)
    ideal_weno_linear_weights(
        xv[3], xv[4], xv[5], xv[2], xv[3], xv[4], xv[1], xv[2], xv[3], x_face_l,
    )
    @test @allocated(weno_f_nonuniform(u, p32, t, xv, dx_vec)) == 0
    @test @allocated(
        compute_beta_nonuniform(
            u[3], u[4], u[5], xv[3], xv[4], xv[5], x_face_l, x_face_r,
        )
    ) == 0
    @test @allocated(
        ideal_weno_linear_weights(
            xv[3], xv[4], xv[5], xv[2], xv[3], xv[4], xv[1], xv[2], xv[3], x_face_l,
        )
    ) == 0
end

@testset "Non-uniform WENO — low-degree polynomial exactness" begin
    h = 0.1
    xv = uniform_stencil(h)
    for p in 0:1
        f = x -> x^p
        exact = p == 0 ? 0.0 : 1.0
        approx = weno_derivative(f, xv)
        @test abs(approx - exact) < 1.0e-12
    end
end

@testset "Non-uniform WENO — ForwardDiff.Dual compatibility" begin
    h = 0.1
    xv = uniform_stencil(h)
    u = sin.(xv)
    du, dx, xL, xR = promote_dual_stencil(u, xv)

    β_dual = compute_beta_nonuniform(
        du[3], du[4], du[5], dx[3], dx[4], dx[5], xL, xR,
    )
    @test β_dual isa Dual{1, Float64, 1}
    @test @inferred(
        compute_beta_nonuniform(
            du[3], du[4], du[5], dx[3], dx[4], dx[5], xL, xR,
        )
    ) isa Dual{1, Float64, 1}
    @test isfinite(ForwardDiff.value(β_dual))
    @test isfinite(ForwardDiff.partials(β_dual, 1))

    d_dual = ideal_weno_linear_weights(
        dx[3], dx[4], dx[5], dx[2], dx[3], dx[4], dx[1], dx[2], dx[3], (dx[2] + dx[3]) / 2,
    )
    @test all(x -> x isa Dual{1, Float64, 1}, d_dual)
    @test sum(ForwardDiff.value.(d_dual)) ≈ 1.0

    flux_dual = weno_f_nonuniform(du, [1.0e-14], 0.0, dx, diff(xv))
    @test flux_dual isa Dual{1, Float64, 1}
    @test @inferred(weno_f_nonuniform(du, [1.0e-14], 0.0, dx, diff(xv))) isa Dual{1, Float64, 1}
    @test isfinite(ForwardDiff.value(flux_dual))

    seeds_u3 = [0.0, 0.0, 1.0, 0.0, 0.0]
    du3 = Dual{1}.(u, seeds_u3)
    flux_u3 = weno_f_nonuniform(du3, [1.0e-14], 0.0, dx, diff(xv))
    ε_fd = 1.0e-7
    u_p = copy(u); u_p[3] += ε_fd
    u_m = copy(u); u_m[3] -= ε_fd
    fd_u3 = (
        weno_f_nonuniform(u_p, [1.0e-14], 0.0, xv, diff(xv)) -
            weno_f_nonuniform(u_m, [1.0e-14], 0.0, xv, diff(xv))
    ) / (2ε_fd)
    @test ForwardDiff.partials(flux_u3, 1) ≈ fd_u3 rtol = 1.0e-5

    for j in 1:5
        seeds = zeros(5)
        seeds[j] = 1.0
        du_j = Dual{1}.(u, seeds)
        flux_j = weno_f_nonuniform(du_j, [1.0e-14], 0.0, dx, diff(xv))
        @test flux_j isa Dual{1, Float64, 1}
        @test isfinite(ForwardDiff.partials(flux_j, 1))
    end
end

@testset "Non-uniform WENO — Symbolics.Num compatibility" begin
    @variables u1 u2 u3 u4 u5 x1 x2 x3 x4 x5
    u_sym = [u1, u2, u3, u4, u5]
    x_sym = [x1, x2, x3, x4, x5]
    dx_sym = [x2 - x1, x3 - x2, x4 - x3, x5 - x4]
    x_face_l = (x2 + x3) / 2
    x_face_r = (x3 + x4) / 2

    d_sym = ideal_weno_linear_weights(
        x3, x4, x5, x2, x3, x4, x1, x2, x3, x_face_l,
    )
    @test all(x -> x isa Num, d_sym)
    @test occursin("x3", string(d_sym[1]))

    β_sym = compute_beta_nonuniform(u3, u4, u5, x3, x4, x5, x_face_l, x_face_r)
    @test β_sym isa Num
    @test occursin("u3", string(β_sym))

    @variables d1 d2 d3 b1 b2 b3
    ω_sym = shi_hu_shu_weights(d1, d2, d3, b1, b2, b3, 1.0e-6)
    @test all(x -> x isa Num, ω_sym)
    @test occursin("b1", string(ω_sym[1]))

    subs_smooth = Dict(
        d1 => 0.1, d2 => 0.6, d3 => 0.3,
        b1 => 1.0e-10, b2 => 1.0e-10, b3 => 1.0e-10,
    )
    ω_eval = evaluate_symbolic.(ω_sym, Ref(subs_smooth))
    @test sum(ω_eval) ≈ 1.0 atol = 1.0e-12
    @test all(isfinite, ω_eval)

    subs_ifelse = Dict(
        d1 => -0.2, d2 => 0.9, d3 => 0.3,
        b1 => 1.0e-10, b2 => 1.0e-10, b3 => 1.0e-10,
    )
    ω_neg = evaluate_symbolic.(ω_sym, Ref(subs_ifelse))
    @test sum(ω_neg) ≈ 1.0 atol = 1.0e-10
    @test all(isfinite, ω_neg)

    flux_sym = weno_f_nonuniform(u_sym, [1.0e-6], 0.0, x_sym, dx_sym)
    @test flux_sym isa Num
    @test occursin("u3", string(flux_sym))
    @test !isequal(flux_sym, 0.0)

    D_sym = Symbolics.derivative(flux_sym, u3)
    @test D_sym isa Num
    @test occursin("u", string(D_sym))

    subs = Dict(
        u1 => 1.0, u2 => 2.0, u3 => 3.0, u4 => 4.0, u5 => 5.0,
        x1 => 0.0, x2 => 0.1, x3 => 0.2, x4 => 0.3, x5 => 0.4,
    )
    xv = uniform_stencil(0.1)
    flux_num = evaluate_symbolic(flux_sym, subs)
    flux_ref = weno_f_nonuniform([1.0, 2.0, 3.0, 4.0, 5.0], [1.0e-6], 0.0, xv, diff(xv))
    @test flux_num ≈ flux_ref rtol = 1.0e-10

    d_num = ideal_weno_linear_weights(
        subs[x3], subs[x4], subs[x5], subs[x2], subs[x3], subs[x4],
        subs[x1], subs[x2], subs[x3], (subs[x2] + subs[x3]) / 2,
    )
    d_sub = evaluate_symbolic.(d_sym, Ref(subs))
    @test d_sub[1] ≈ d_num[1] rtol = 1.0e-10
    @test d_sub[2] ≈ d_num[2] rtol = 1.0e-10
    @test d_sub[3] ≈ d_num[3] rtol = 1.0e-10
end

@testset "Non-uniform WENO — mixed-type grid contract (MOL pipeline)" begin
    h = 0.1
    xv = uniform_stencil(h)
    u = sin.(xv)
    dual_u = Dual{1}.(u, ones(5))

    @test weno_f_nonuniform(dual_u, [1.0e-14], 0.0, xv, diff(xv)) isa Dual{1, Float64, 1}

    @variables u1 u2 u3 u4 u5
    u_sym = [u1, u2, u3, u4, u5]
    xL, xR = face_geometry(xv)
    @test compute_beta_nonuniform(u3, u4, u5, xv[3], xv[4], xv[5], xL, xR) isa Num
    @test weno_f_nonuniform(u_sym, [1.0e-6], 0.0, xv, diff(xv)) isa Num
end
