# Verification of the asymmetric one-sided node-centered WENO-5 boundary reconstruction via
# Val{Target} dispatch. Left boundary: Val{1} (target x[1]), Val{2} (target x[2]). Right boundary:
# Val{4} (target x[4]), Val{5} (target x[5]). The interior Val{3} path is verified in
# weno_nonuniform_core.jl.

using Test
using MethodOfLines
using Symbolics
using ForwardDiff

const WFB = MethodOfLines.weno_f_nonuniform
const PB = [1.0e-6]
const WENO_EPS_B = 1.0e-6

# Typed function barriers for @allocated tests (avoids boxing on 1.10 LTS).
bench_b(u::Vector{Float64}, x::AbstractVector{Float64}, dx, ::Val{T}) where {T} =
    MethodOfLines.weno_f_nonuniform(u, (WENO_EPS_B,), 0.0, x, dx, Val(T))
bench_b32(u::Vector{Float32}, x::AbstractVector{Float32}, dx::AbstractVector{Float32}, ::Val{T}) where {T} =
    MethodOfLines.weno_f_nonuniform(u, (WENO_EPS_B,), 0.0f0, x, dx, Val(T))
@inline measure_alloc(f::F, args::Vararg{Any, N}) where {F, N} = @allocated f(args...)

# Full nonlinear WENO reconstruction at the requested target.
wfb(u, x, ::Val{T}) where {T} = WFB(u, PB, 0.0, x, diff(x), Val(T))

# Fornberg m = 1 (first-derivative) weights of a 3-point sub-stencil evaluated at xt.
fw(α, xt) = MethodOfLines._fornberg3_weights(α, xt)[2]

# Combined 5-node weights of the linear ideal-weight decomposition Σ d_k p_k'(x_target), assembled
# from `_weno_ideal_d0d2`.
function combined_weights(xs, ::Val{T}) where {T}
    x1, x2, x3, x4, x5 = xs
    d0, d2 = MethodOfLines._weno_ideal_d0d2(Val(T), x1, x2, x3, x4, x5)
    d1 = one(d0) - d0 - d2
    xt = xs[T]
    f0 = fw((x1, x2, x3), xt)
    f1 = fw((x2, x3, x4), xt)
    f2 = fw((x3, x4, x5), xt)
    w1 = d0 * f0[1]
    w2 = d0 * f0[2] + d1 * f1[1]
    w3 = d0 * f0[3] + d1 * f1[2] + d2 * f2[1]
    w4 = d1 * f1[3] + d2 * f2[2]
    w5 = d2 * f2[3]
    return (w1, w2, w3, w4, w5)
end

# Linear ideal-weight reconstruction Σ d_k p_k'(x_target).
function linear_recon_b(xs, u, vt::Val)
    w = combined_weights(xs, vt)
    return sum(w[k] * u[k] for k in 1:5)
end

# Symbolic 5-point Lagrange interpolant derivative weights ℓ_j'(x_target). This is the exact
# degree-4 derivative operator the d_k decomposition must reproduce node-for-node.
function lagrange_deriv_weights(xs, target)
    @variables xx
    D = Differential(xx)
    return map(1:5) do j
        ℓj = prod((xx - xs[m]) / (xs[j] - xs[m]) for m in 1:5 if m != j)
        dℓj = expand_derivatives(D(ℓj))
        Symbolics.value(substitute(dℓj, Dict(xx => xs[target])))
    end
end

# Sum of the boundary ideal weights (must be 1 by construction: d1 = 1 - d0 - d2).
function sum_d(xs, ::Val{T}) where {T}
    x1, x2, x3, x4, x5 = xs
    d0, d2 = MethodOfLines._weno_ideal_d0d2(Val(T), x1, x2, x3, x4, x5)
    return d0 + (one(d0) - d0 - d2) + d2
end

@testset "WENO Non-Uniform Boundary (Asymmetric Val{Target})" begin

    grids = (
        [0.0, 0.3, 0.9, 1.7, 2.2],
        [-0.3, 0.4, 0.55, 1.9, 2.4],
        [0.0, 0.15, 0.8, 1.05, 2.7],
    )

    @testset "Symbolic d_k identity vs 5-point Lagrange derivative" begin
        # The decomposition Σ d_k p_k'(x_target) must equal the 5-point Lagrange derivative
        # P'_{5pt}(x_target) node-for-node for every target.
        for xs in grids, T in (1, 2, 3, 4, 5)
            wcomb = combined_weights(xs, Val(T))
            wlag = lagrange_deriv_weights(xs, T)
            for k in 1:5
                @test wcomb[k] ≈ wlag[k] rtol = 1.0e-10 atol = 1.0e-12
            end
        end
    end

    @testset "Convex partition (Σ d_k = 1)" begin
        for xs in grids, T in (1, 2, 3, 4, 5)
            @test sum_d(xs, Val(T)) ≈ 1.0 atol = 1.0e-12
        end
    end

    @testset "Linear ideal-weight identity (degree <= 4)" begin
        for xs in grids, T in (1, 2, 4, 5)
            xt = xs[T]
            for (g, dg) in (
                    (x -> 1.7 + 0.9x, x -> 0.9),
                    (x -> x^2 - 0.3x, x -> 2x - 0.3),
                    (x -> x^3 - 2x, x -> 3x^2 - 2),
                    (x -> x^4 - 0.5x^2 + x, x -> 4x^3 - x + 1),
                    (x -> 2.3x^4 - 1.1x^3, x -> 9.2x^3 - 3.3x^2),
                )
                @test linear_recon_b(xs, g.(xs), Val(T)) ≈ dg(xt) rtol = 1.0e-10 atol = 1.0e-10
            end
        end
    end

    @testset "Polynomial exactness of full WENO (degree <= 2)" begin
        # Every 3-point sub-stencil derivative is exact for degree <= 2, so any SHS recombination is
        # exact regardless of the nonlinear weights.
        for xs in grids, T in (1, 2, 4, 5)
            xt = xs[T]
            f0(x) = 1.7;                 df0(x) = 0.0
            f1(x) = 1.7 + 0.9x;          df1(x) = 0.9
            f2(x) = 1.7 + 0.9x - 0.4x^2; df2(x) = 0.9 - 0.8x
            @test wfb(f0.(xs), xs, Val(T)) ≈ df0(xt) atol = 1.0e-12
            @test wfb(f1.(xs), xs, Val(T)) ≈ df1(xt) atol = 1.0e-12
            @test wfb(f2.(xs), xs, Val(T)) ≈ df2(xt) atol = 1.0e-11
        end
    end

    @testset "Scalar-dx 6-arg method hits the identical core" begin
        xs = grids[1]
        f2(x) = 1.7 + 0.9x - 0.4x^2
        u = f2.(xs)
        for T in (1, 2, 3, 4, 5)
            @test WFB(u, PB, 0.0, xs, 0.5, Val(T)) == WFB(u, PB, 0.0, xs, diff(xs), Val(T))
        end
    end

    @testset "Interior backward compatibility (5-arg == Val(3))" begin
        # The 5-arg methods must match the explicit interior dispatch exactly.
        xs = grids[2]
        u = sin.(xs)
        @test WFB(u, PB, 0.0, xs, diff(xs)) === WFB(u, PB, 0.0, xs, diff(xs), Val(3))
        @test WFB(u, PB, 0.0, xs, 0.5) === WFB(u, PB, 0.0, xs, 0.5, Val(3))
    end

    @testset "Zero-allocation guarantee" begin
        xs = [0.0, 0.7, 1.5, 2.0, 3.1]
        u = sin.(xs)
        dxv = diff(xs)
        for T in (1, 2, 3, 4, 5)
            bench_b(u, xs, dxv, Val(T))            # warmup
            measure_alloc(bench_b, u, xs, dxv, Val(T))
            @test measure_alloc(bench_b, u, xs, dxv, Val(T)) == 0
        end
        xs32 = Float32.(xs); u32 = Float32.(u); dxv32 = Float32.(dxv)
        for T in (1, 2, 4, 5)
            bench_b32(u32, xs32, dxv32, Val(T))    # warmup
            measure_alloc(bench_b32, u32, xs32, dxv32, Val(T))
            @test measure_alloc(bench_b32, u32, xs32, dxv32, Val(T)) == 0
        end
    end

    @testset "Type stability and mixed-type promotion" begin
        xs = [0.0, 0.6, 1.4, 2.1, 3.3]
        dxv = diff(xs)
        f2(x) = 1.7 + 0.9x - 0.4x^2
        u = f2.(xs)

        for T in (1, 2, 4, 5)
            D64 = WFB(u, PB, 0.0, xs, dxv, Val(T))
            @test (@inferred WFB(u, PB, 0.0, xs, dxv, Val(T))) isa Float64

            xs32 = Float32.(xs); dxv32 = Float32.(dxv); p32 = Float32[1.0f-6]
            D32 = WFB(Float32.(u), p32, 0.0f0, xs32, dxv32, Val(T))
            @test D32 isa Float32
            @test D32 ≈ Float32(D64) rtol = 1.0f-4

            # ForwardDiff.Dual seeded on u[Target] vs central difference.
            seed = zeros(5); seed[T] = 1.0
            ud = ForwardDiff.Dual.(u, seed)
            Dd = WFB(ud, PB, 0.0, xs, dxv, Val(T))
            @test Dd isa ForwardDiff.Dual
            @test ForwardDiff.value(Dd) ≈ D64 atol = 1.0e-12
            hfd = 1.0e-6
            up = copy(u); up[T] += hfd
            um = copy(u); um[T] -= hfd
            pfd = (WFB(up, PB, 0.0, xs, dxv, Val(T)) - WFB(um, PB, 0.0, xs, dxv, Val(T))) / (2hfd)
            @test ForwardDiff.partials(Dd)[1] ≈ pfd rtol = 1.0e-5

            # Symbolics.Num: symbolic build then numeric evaluation.
            @variables uu[1:5]
            usym = collect(uu)
            Dsym = WFB(usym, PB, 0.0, xs, dxv, Val(T))
            @test Dsym isa Num
            gnum = build_function(Dsym, usym; expression = Val{false})
            @test gnum(u) ≈ D64 atol = 1.0e-12
        end
    end

    @testset "SubArray view ingestion (production argument types)" begin
        global_x = [0.0, 0.3, 0.9, 1.7, 2.2, 3.0, 3.4]
        xs_view = @view global_x[2:6]
        u = sin.(collect(xs_view))
        dx_full = diff(global_x)
        dxv_view = @view dx_full[2:5]

        for T in (1, 2, 4, 5)
            Dref = WFB(u, PB, 0.0, collect(xs_view), diff(collect(xs_view)), Val(T))
            Dview = WFB(u, PB, 0.0, xs_view, dxv_view, Val(T))
            @test Dview == Dref
            @test Dview isa Float64
            @test (@inferred WFB(u, PB, 0.0, xs_view, dxv_view, Val(T))) isa Float64
        end
    end
end
