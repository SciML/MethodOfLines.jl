# Verification of the node-centered direct-derivative WENO-5 core (`weno_f_nonuniform`,
# FunctionalScheme{5,0}, returning du/dx at the center node x[3]).
#
# Properties under test: the nonlinear scheme is exact through degree 2 (ω_k = d_k there) while the
# linear ideal-weight decomposition is exact through degree 4; the derivative-ideal weights d_k form
# a convex partition (d_k >= 0, Σ d_k = 1) on every monotone grid, so the Shi-Hu-Shu negative branch
# is inert and is verified through convexity rather than a negative-weight path.

using Test
using MethodOfLines
using Symbolics
using ForwardDiff

const WF = MethodOfLines.weno_f_nonuniform
const P = [1.0e-6]
const WENO_EPS_F64 = 1.0e-6
const WENO_EPS_F32 = 1.0f-6

# Typed function barriers for @allocated tests (avoid closure boxing on LTS).
bench_weno_f64(u::Vector{Float64}, x::AbstractVector{Float64}, dx) =
    MethodOfLines.weno_f_nonuniform(u, (WENO_EPS_F64,), 0.0, x, dx)
bench_weno_f32(u::Vector{Float32}, x::AbstractVector{Float32}, dx::AbstractVector{Float32}) =
    MethodOfLines.weno_f_nonuniform(u, (WENO_EPS_F32,), 0.0f0, x, dx)
bench_weno_sub(u::Vector{Float64}, x::SubArray{Float64,1}, dx::SubArray{Float64,1}) =
    MethodOfLines.weno_f_nonuniform(u, (WENO_EPS_F64,), 0.0, x, dx)

# Public-API helpers (vector / scalar dx). `dx` is unused by the kernel but required by the contract.
wf(u, x) = WF(u, P, 0.0, x, diff(x))
wf_scalar(u, x) = WF(u, P, 0.0, x, 0.5)

# Closed-form derivative-ideal weights (mirror of the core; the oracle in the linear-identity set is
# the analytic derivative, so a faulty formula fails against it).
function ideal_weights(x)
    x1, x2, x3, x4, x5 = x
    d0 = ((x3 - x4) * (x3 - x5)) / ((x1 - x4) * (x1 - x5))
    d2 = ((x3 - x1) * (x3 - x2)) / ((x5 - x1) * (x5 - x2))
    d1 = 1 - d0 - d2
    return (d0, d1, d2)
end

# 3-point sub-stencil first derivative at xt via the shipped Fornberg engine (m = 1 weights).
function sub_deriv(α, ua, ub, uc, xt)
    w1 = MethodOfLines._fornberg3_weights(α, xt)[2]
    return w1[1] * ua + w1[2] * ub + w1[3] * uc
end

# Linear (ideal-weight) reconstruction Σ d_k p_k'(x_i) assembled from the shipped pieces.
function linear_recon(x, u)
    d0, d1, d2 = ideal_weights(x)
    r0 = sub_deriv((x[1], x[2], x[3]), u[1], u[2], u[3], x[3])
    r1 = sub_deriv((x[2], x[3], x[4]), u[2], u[3], u[4], x[3])
    r2 = sub_deriv((x[3], x[4], x[5]), u[3], u[4], u[5], x[3])
    return d0 * r0 + d1 * r1 + d2 * r2
end

@testset "WENO Non-Uniform Core (Direct Derivative Reconstruction)" begin

    @testset "Polynomial exactness (degree <= 2)" begin
        xs = [0.0, 0.6, 1.4, 2.1, 3.3]      # ratios 0.6:0.8:0.7:1.2, center xs[3] = 1.4
        xc = xs[3]

        f0(x) = 1.7;                  df0(x) = 0.0
        f1(x) = 1.7 + 0.9x;           df1(x) = 0.9
        f2(x) = 1.7 + 0.9x - 0.4x^2;  df2(x) = 0.9 - 0.8x

        @test wf(f0.(xs), xs) ≈ df0(xc) atol = 1e-14
        @test wf(f1.(xs), xs) ≈ df1(xc) atol = 1e-13
        @test wf(f2.(xs), xs) ≈ df2(xc) atol = 1e-12

        # Scalar-dx fallback hits the identical core: bit-for-bit equal.
        @test wf_scalar(f2.(xs), xs) == wf(f2.(xs), xs)

        # The nonlinear scheme is not exact for degree 3 at finite h (ω_k != d_k).
        f3(x) = x^3
        @test !isapprox(wf(f3.(xs), xs), 3xc^2; atol = 1e-6)
    end

    @testset "Linear ideal-weight identity (degree <= 4)" begin
        for xs in ([0.0, 0.6, 1.4, 2.1, 3.3], [-0.3, 0.4, 0.55, 1.9, 2.4])
            xc = xs[3]
            # Convex partition: Σ d_k = 1 and d_k in [0, 1] (the SHS negative branch is inert).
            d0, d1, d2 = ideal_weights(xs)
            @test d0 + d1 + d2 ≈ 1.0 atol = 1e-14
            @test all(0 .<= (d0, d1, d2) .<= 1)

            for (g, dg) in (
                    (x -> x^3 - 2x,            x -> 3x^2 - 2),
                    (x -> x^4 - 0.5x^2 + x,    x -> 4x^3 - x + 1),
                    (x -> 2.3x^4 - 1.1x^3,     x -> 9.2x^3 - 3.3x^2),
                )
                @test linear_recon(xs, g.(xs)) ≈ dg(xc) rtol = 1e-11 atol = 1e-11
            end
        end
    end

    @testset "Order of convergence (MMS)" begin
        f(x) = sin(1.3x) + 0.5x
        df(x) = 1.3cos(1.3x) + 0.5

        # Frozen offsets: inner nodes perturbed ~10-20% off uniform, ends fixed. The relative
        # geometry is constant under refinement (self-similar), so the asymptotic order is isolated.
        o = [-2.0, -1.13, 0.08, 0.91, 2.0]
        xc = 1.0
        hs = [0.2, 0.1, 0.05, 0.025, 0.0125]

        errs = map(hs) do h
            xs = xc .+ h .* o
            abs(wf(f.(xs), xs) - df(xs[3]))   # evaluated at the actual center node xs[3]
        end
        orders = [log2(errs[k] / errs[k+1]) for k in 1:length(hs)-1]

        # Asymptotic order -> 4 (this FD scheme is 4th order on non-uniform grids).
        @test orders[end] > 3.85
        @test orders[end-1] > 3.7
        # Least-squares slope over the finest region (h <= 0.05) confirms ~4th order.
        lh = log.(hs[3:end]); le = log.(errs[3:end]); n = length(lh)
        slope = (n * sum(lh .* le) - sum(lh) * sum(le)) / (n * sum(lh .^ 2) - sum(lh)^2)
        @test slope > 3.8
        # A 1st/2nd/3rd-order regression would drag every order well below this.
        @test all(>(3.0), orders)
    end

    @testset "Extreme grid stretching (1:1e6)" begin
        grids = (
            [0.0, 1.0, 1.0 + 1e-6, 2.0 + 1e-6, 3.0 + 1e-6],   # 1e-6 center cell
            [0.0, 1e-6, 2e-6, 1.0, 2.0],                       # extreme clustering near the center
        )
        for g in grids
            # Finiteness under severe ill-conditioning.
            @test isfinite(wf((x -> x^2).(g), g))

            # A linear field's derivative is exact regardless of the grid (Σ d_k = 1, every r_k
            # equals the slope), so the SHS recombination must not corrupt it.
            lin(x) = 2.0 + 3.0x
            @test wf(lin.(g), g) ≈ 3.0 rtol = 1e-6

            # Stability (not accuracy) target for a quadratic on a brutally stretched stencil.
            @test abs(wf((x -> x^2).(g), g) - 2 * g[3]) < 1.0

            # d_k stay a convex partition even at 1:1e6, so d1 < 0 is unreachable.
            d0, d1, d2 = ideal_weights(g)
            @test all(isfinite, (d0, d1, d2))
            @test d1 >= 0
            @test d0 + d1 + d2 ≈ 1.0 atol = 1e-10
        end

        # A batch of deterministic extreme-ratio grids must never produce NaN/Inf.
        finite_all = true
        for k in 0:60
            s = 10.0 ^ (k / 10 - 3)                       # cell-ratio scale spanning 1e-3 .. 1e3
            g = cumsum([1.0, s, 1.0, s, 1.0]) .- 1.0
            finite_all &= isfinite(wf((x -> sin(x)).(g), g))
        end
        @test finite_all
    end

    @testset "Zero-allocation guarantee" begin
        xs = [0.0, 0.7, 1.5, 2.0, 3.1]
        u = sin.(xs)
        dxv = diff(xs)

        bench_weno_f64(u, xs, dxv)        # warmup (compile)
        bench_weno_f64(u, xs, 0.5)

        @test @allocated(bench_weno_f64(u, xs, dxv)) == 0
        @test @allocated(bench_weno_f64(u, xs, 0.5)) == 0
        @test @allocated(MethodOfLines.weno_f_nonuniform(u, (WENO_EPS_F64,), 0.0, xs, dxv)) == 0

        xs32 = Float32.(xs); u32 = Float32.(u); dxv32 = Float32.(dxv)
        bench_weno_f32(u32, xs32, dxv32)
        @test @allocated(bench_weno_f32(u32, xs32, dxv32)) == 0
    end

    @testset "Type stability" begin
        xs = [0.0, 0.6, 1.4, 2.1, 3.3]
        dxv = diff(xs)
        f2(x) = 1.7 + 0.9x - 0.4x^2
        u = f2.(xs)
        D64 = WF(u, P, 0.0, xs, dxv)

        # Static inference: concrete Float64 return.
        @test (@inferred WF(u, P, 0.0, xs, dxv)) isa Float64

        # Float32 end-to-end.
        xs32 = Float32.(xs); dxv32 = Float32.(dxv); p32 = Float32[1.0f-6]
        D32 = WF(Float32.(u), p32, 0.0f0, xs32, dxv32)
        @test D32 isa Float32
        @test D32 ≈ Float32(D64) rtol = 1.0f-4

        # ForwardDiff.Dual: differentiate the output w.r.t. u[3]; compare to a central difference.
        seed = [0.0, 0.0, 1.0, 0.0, 0.0]
        ud = ForwardDiff.Dual.(u, seed)
        Dd = WF(ud, P, 0.0, xs, dxv)
        @test Dd isa ForwardDiff.Dual
        @test ForwardDiff.value(Dd) ≈ D64 atol = 1e-12
        hfd = 1e-6
        up = copy(u); up[3] += hfd
        um = copy(u); um[3] -= hfd
        pfd = (WF(up, P, 0.0, xs, dxv) - WF(um, P, 0.0, xs, dxv)) / (2hfd)
        @test ForwardDiff.partials(Dd)[1] ≈ pfd rtol = 1e-5

        # Symbolics.Num: symbolic build then numeric evaluation must equal the direct numeric call.
        @variables uu[1:5]
        usym = collect(uu)
        Dsym = WF(usym, P, 0.0, xs, dxv)
        @test Dsym isa Num
        gnum = build_function(Dsym, usym; expression = Val{false})
        @test gnum(u) ≈ D64 atol = 1e-12
    end

    @testset "SubArray view ingestion (production argument types)" begin
        # The discretizer passes x as a `@view s.grid[x][itap]` and the non-uniform dx as a view of
        # the spacing vector. Verify that compiler views preserve the result, inference, and zero
        # allocations, rather than only plain `Vector`s.
        global_x = [0.0, 0.3, 0.9, 1.7, 2.2, 3.0, 3.4]   # length-7 backing grid
        xs_view = @view global_x[2:6]                     # length-5 SubArray (the interior stencil)
        u = sin.(collect(xs_view))                        # discvars indexing materializes a Vector
        dx_full = diff(global_x)
        dxv_view = @view dx_full[2:5]                      # length-4 SubArray of spacings

        Dref = WF(u, P, 0.0, collect(xs_view), diff(collect(xs_view)))
        Dview = WF(u, P, 0.0, xs_view, dxv_view)
        @test Dview == Dref
        @test Dview isa Float64
        @test (@inferred WF(u, P, 0.0, xs_view, dxv_view)) isa Float64

        bench_weno_sub(u, xs_view, dxv_view)   # warmup
        @test @allocated(bench_weno_sub(u, xs_view, dxv_view)) == 0
    end
end
