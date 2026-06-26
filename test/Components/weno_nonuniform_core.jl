# Verification of the node-centered WENO-5 core: exact through degree 2 (nonlinear) and degree 4
# (linear ideal-weight decomposition).

using Test
using MethodOfLines
using Symbolics
using ForwardDiff

const WF = MethodOfLines.weno_f_nonuniform
const P = [1.0e-6]
const WENO_EPS_F64 = 1.0e-6
const WENO_EPS_F32 = 1.0f-6

# Typed function barriers for @allocated tests.
bench_weno_f64(u::Vector{Float64}, x::AbstractVector{Float64}, dx) =
    MethodOfLines.weno_f_nonuniform(u, (WENO_EPS_F64,), 0.0, x, dx)
bench_weno_f32(u::Vector{Float32}, x::AbstractVector{Float32}, dx::AbstractVector{Float32}) =
    MethodOfLines.weno_f_nonuniform(u, (WENO_EPS_F32,), 0.0f0, x, dx)
bench_weno_sub(u::Vector{Float64}, x::SubArray{Float64, 1}, dx::SubArray{Float64, 1}) =
    MethodOfLines.weno_f_nonuniform(u, (WENO_EPS_F64,), 0.0, x, dx)

# Allocation measured inside a specialized barrier; on 1.10 LTS the result is otherwise boxed.
@inline measure_alloc(f::F, args::Vararg{Any, N}) where {F, N} = @allocated f(args...)

# Public-API helpers (vector / scalar dx).
wf(u, x) = WF(u, P, 0.0, x, diff(x))
wf_scalar(u, x) = WF(u, P, 0.0, x, 0.5)

# Closed-form derivative-ideal weights (mirror of the core).
function ideal_weights(x)
    x1, x2, x3, x4, x5 = x
    d0 = ((x3 - x4) * (x3 - x5)) / ((x1 - x4) * (x1 - x5))
    d2 = ((x3 - x1) * (x3 - x2)) / ((x5 - x1) * (x5 - x2))
    d1 = 1 - d0 - d2
    return (d0, d1, d2)
end

# 3-point sub-stencil first derivative at xt (Fornberg m = 1 weights).
function sub_deriv(α, ua, ub, uc, xt)
    w1 = MethodOfLines._fornberg3_weights(α, xt)[2]
    return w1[1] * ua + w1[2] * ub + w1[3] * uc
end

# Linear ideal-weight reconstruction Σ d_k p_k'(x_i).
function linear_recon(x, u)
    d0, d1, d2 = ideal_weights(x)
    r0 = sub_deriv((x[1], x[2], x[3]), u[1], u[2], u[3], x[3])
    r1 = sub_deriv((x[2], x[3], x[4]), u[2], u[3], u[4], x[3])
    r2 = sub_deriv((x[3], x[4], x[5]), u[3], u[4], u[5], x[3])
    return d0 * r0 + d1 * r1 + d2 * r2
end

@testset "WENO Non-Uniform Core (Direct Derivative Reconstruction)" begin

    @testset "Polynomial exactness (degree <= 2)" begin
        xs = [0.0, 0.6, 1.4, 2.1, 3.3]
        xc = xs[3]

        f0(x) = 1.7;                  df0(x) = 0.0
        f1(x) = 1.7 + 0.9x;           df1(x) = 0.9
        f2(x) = 1.7 + 0.9x - 0.4x^2;  df2(x) = 0.9 - 0.8x

        @test wf(f0.(xs), xs) ≈ df0(xc) atol = 1.0e-14
        @test wf(f1.(xs), xs) ≈ df1(xc) atol = 1.0e-13
        @test wf(f2.(xs), xs) ≈ df2(xc) atol = 1.0e-12

        # Scalar-dx fallback hits the identical core.
        @test wf_scalar(f2.(xs), xs) == wf(f2.(xs), xs)

        # Not exact for degree 3 at finite h.
        f3(x) = x^3
        @test !isapprox(wf(f3.(xs), xs), 3xc^2; atol = 1.0e-6)
    end

    @testset "Linear ideal-weight identity (degree <= 4)" begin
        for xs in ([0.0, 0.6, 1.4, 2.1, 3.3], [-0.3, 0.4, 0.55, 1.9, 2.4])
            xc = xs[3]
            # Convex partition: Σ d_k = 1, d_k in [0, 1].
            d0, d1, d2 = ideal_weights(xs)
            @test d0 + d1 + d2 ≈ 1.0 atol = 1.0e-14
            @test all(0 .<= (d0, d1, d2) .<= 1)

            for (g, dg) in (
                    (x -> x^3 - 2x, x -> 3x^2 - 2),
                    (x -> x^4 - 0.5x^2 + x, x -> 4x^3 - x + 1),
                    (x -> 2.3x^4 - 1.1x^3, x -> 9.2x^3 - 3.3x^2),
                )
                @test linear_recon(xs, g.(xs)) ≈ dg(xc) rtol = 1.0e-11 atol = 1.0e-11
            end
        end
    end

    @testset "Order of convergence (MMS)" begin
        f(x) = sin(1.3x) + 0.5x
        df(x) = 1.3cos(1.3x) + 0.5

        # Self-similar refinement: fixed offsets isolate the asymptotic order.
        o = [-2.0, -1.13, 0.08, 0.91, 2.0]
        xc = 1.0
        hs = [0.2, 0.1, 0.05, 0.025, 0.0125]

        errs = map(hs) do h
            xs = xc .+ h .* o
            abs(wf(f.(xs), xs) - df(xs[3]))
        end
        orders = [log2(errs[k] / errs[k + 1]) for k in 1:(length(hs) - 1)]

        @test orders[end] > 3.85
        @test orders[end - 1] > 3.7
        lh = log.(hs[3:end]); le = log.(errs[3:end]); n = length(lh)
        slope = (n * sum(lh .* le) - sum(lh) * sum(le)) / (n * sum(lh .^ 2) - sum(lh)^2)
        @test slope > 3.8
        @test all(>(3.0), orders)
    end

    @testset "Extreme grid stretching (1:1e6)" begin
        grids = (
            [0.0, 1.0, 1.0 + 1.0e-6, 2.0 + 1.0e-6, 3.0 + 1.0e-6],
            [0.0, 1.0e-6, 2.0e-6, 1.0, 2.0],
        )
        for g in grids
            @test isfinite(wf((x -> x^2).(g), g))

            # Linear field: exact derivative on any grid.
            lin(x) = 2.0 + 3.0x
            @test wf(lin.(g), g) ≈ 3.0 rtol = 1.0e-6

            # Stability (not accuracy) bound for a quadratic.
            @test abs(wf((x -> x^2).(g), g) - 2 * g[3]) < 1.0

            d0, d1, d2 = ideal_weights(g)
            @test all(isfinite, (d0, d1, d2))
            @test d1 >= 0
            @test d0 + d1 + d2 ≈ 1.0 atol = 1.0e-10
        end

        finite_all = true
        for k in 0:60
            s = 10.0^(k / 10 - 3)
            g = cumsum([1.0, s, 1.0, s, 1.0]) .- 1.0
            finite_all &= isfinite(wf((x -> sin(x)).(g), g))
        end
        @test finite_all
    end

    @testset "Zero-allocation guarantee" begin
        xs = [0.0, 0.7, 1.5, 2.0, 3.1]
        u = sin.(xs)
        dxv = diff(xs)

        # Warmup every measured form.
        bench_weno_f64(u, xs, dxv)
        bench_weno_f64(u, xs, 0.5)
        MethodOfLines.weno_f_nonuniform(u, (WENO_EPS_F64,), 0.0, xs, dxv)
        measure_alloc(bench_weno_f64, u, xs, dxv)

        @test measure_alloc(bench_weno_f64, u, xs, dxv) == 0
        @test measure_alloc(bench_weno_f64, u, xs, 0.5) == 0
        @test measure_alloc(MethodOfLines.weno_f_nonuniform, u, (WENO_EPS_F64,), 0.0, xs, dxv) == 0

        xs32 = Float32.(xs); u32 = Float32.(u); dxv32 = Float32.(dxv)
        bench_weno_f32(u32, xs32, dxv32)
        measure_alloc(bench_weno_f32, u32, xs32, dxv32)
        @test measure_alloc(bench_weno_f32, u32, xs32, dxv32) == 0
    end

    @testset "Type stability" begin
        xs = [0.0, 0.6, 1.4, 2.1, 3.3]
        dxv = diff(xs)
        f2(x) = 1.7 + 0.9x - 0.4x^2
        u = f2.(xs)
        D64 = WF(u, P, 0.0, xs, dxv)

        @test (@inferred WF(u, P, 0.0, xs, dxv)) isa Float64

        xs32 = Float32.(xs); dxv32 = Float32.(dxv); p32 = Float32[1.0f-6]
        D32 = WF(Float32.(u), p32, 0.0f0, xs32, dxv32)
        @test D32 isa Float32
        @test D32 ≈ Float32(D64) rtol = 1.0f-4

        # ForwardDiff.Dual w.r.t. u[3] vs central difference.
        seed = [0.0, 0.0, 1.0, 0.0, 0.0]
        ud = ForwardDiff.Dual.(u, seed)
        Dd = WF(ud, P, 0.0, xs, dxv)
        @test Dd isa ForwardDiff.Dual
        @test ForwardDiff.value(Dd) ≈ D64 atol = 1.0e-12
        hfd = 1.0e-6
        up = copy(u); up[3] += hfd
        um = copy(u); um[3] -= hfd
        pfd = (WF(up, P, 0.0, xs, dxv) - WF(um, P, 0.0, xs, dxv)) / (2hfd)
        @test ForwardDiff.partials(Dd)[1] ≈ pfd rtol = 1.0e-5

        # Symbolics.Num: symbolic build then numeric evaluation.
        @variables uu[1:5]
        usym = collect(uu)
        Dsym = WF(usym, P, 0.0, xs, dxv)
        @test Dsym isa Num
        gnum = build_function(Dsym, usym; expression = Val{false})
        @test gnum(u) ≈ D64 atol = 1.0e-12
    end

    @testset "SubArray view ingestion (production argument types)" begin
        # SubArray views (as passed by the discretizer) must match Vector behavior in result,
        # inference, and allocation.
        global_x = [0.0, 0.3, 0.9, 1.7, 2.2, 3.0, 3.4]
        xs_view = @view global_x[2:6]
        u = sin.(collect(xs_view))
        dx_full = diff(global_x)
        dxv_view = @view dx_full[2:5]

        Dref = WF(u, P, 0.0, collect(xs_view), diff(collect(xs_view)))
        Dview = WF(u, P, 0.0, xs_view, dxv_view)
        @test Dview == Dref
        @test Dview isa Float64
        @test (@inferred WF(u, P, 0.0, xs_view, dxv_view)) isa Float64

        bench_weno_sub(u, xs_view, dxv_view)
        measure_alloc(bench_weno_sub, u, xs_view, dxv_view)
        @test measure_alloc(bench_weno_sub, u, xs_view, dxv_view) == 0
    end
end
