# boundary_weights_prototype.jl
using Test
using ForwardDiff
using BenchmarkTools

# ==========================================
# 🛡️ LEFT BOUNDARY (Forward Difference)
# ==========================================
function get_nonuniform_weights_1st_deriv_left(h1::T, h2::T) where T<:Real
    @assert h1 > zero(T) && h2 > zero(T) "Grid spacings (h1, h2) must be strictly positive!"
    den1 = h1 * (h1 + h2)
    den2 = h1 * h2
    den3 = h2 * (h1 + h2)

    c0 = -(2*one(T)*h1 + h2) / den1
    c1 = (h1 + h2) / den2
    c2 = -h1 / den3
    return (c0, c1, c2)
end

function get_nonuniform_weights_2nd_deriv_left(h1::T, h2::T) where T<:Real
    @assert h1 > zero(T) && h2 > zero(T) "Grid spacings (h1, h2) must be strictly positive!"
    den1 = h1 * (h1 + h2)
    den2 = h1 * h2
    den3 = h2 * (h1 + h2)

    c0 = 2*one(T) / den1
    c1 = -2*one(T) / den2
    c2 = 2*one(T) / den3
    return (c0, c1, c2)
end

# ==========================================
# 🛡️ RIGHT BOUNDARY (Backward Difference)
# ==========================================
function get_nonuniform_weights_1st_deriv_right(h1::T, h2::T) where T<:Real
    c0, c1, c2 = get_nonuniform_weights_1st_deriv_left(h1, h2)
    return (-c0, -c1, -c2)
end

function get_nonuniform_weights_2nd_deriv_right(h1::T, h2::T) where T<:Real
    return get_nonuniform_weights_2nd_deriv_left(h1, h2)
end

# ==========================================
# 🧪 FORMAL UNIT TESTS (SciML Standard)
# ==========================================
@testset "Non-Uniform Boundary Stencils (End-to-End)" begin
    h1, h2 = 0.01, 0.1

    @testset "1. Conservation of Mass (All Derivatives)" begin
        @test isapprox(sum(get_nonuniform_weights_1st_deriv_left(h1, h2)), 0.0, atol=1e-12)
        @test isapprox(sum(get_nonuniform_weights_2nd_deriv_left(h1, h2)), 0.0, atol=1e-10)
        @test isapprox(sum(get_nonuniform_weights_1st_deriv_right(h1, h2)), 0.0, atol=1e-12)
        @test isapprox(sum(get_nonuniform_weights_2nd_deriv_right(h1, h2)), 0.0, atol=1e-10)
    end

    @testset "2. Symmetry Check (Left vs Right)" begin
        w_1st_left = get_nonuniform_weights_1st_deriv_left(h1, h2)
        w_1st_right = get_nonuniform_weights_1st_deriv_right(h1, h2)
        @test all(w_1st_left .== .-w_1st_right) # 1st deriv signs flip

        w_2nd_left = get_nonuniform_weights_2nd_deriv_left(h1, h2)
        w_2nd_right = get_nonuniform_weights_2nd_deriv_right(h1, h2)
        @test all(w_2nd_left .== w_2nd_right) # 2nd deriv stays identical
    end

    @testset "3. Guardrails (Zero/Negative Grid)" begin
        @test_throws AssertionError get_nonuniform_weights_1st_deriv_left(0.0, 0.1)
        @test_throws AssertionError get_nonuniform_weights_1st_deriv_left(0.1, -0.05)
    end

    @testset "4. ForwardDiff (AD) Compatibility" begin
        w_ad = get_nonuniform_weights_1st_deriv_right(ForwardDiff.Dual(0.01, 1.0), ForwardDiff.Dual(0.1, 0.0))
        @test eltype(w_ad) <: ForwardDiff.Dual
        @test isapprox(ForwardDiff.value(sum(w_ad)), 0.0, atol=1e-12)
    end

    @testset "5. Extreme Grid Stretching (1:10000 Ratio)" begin
        w_stretch = get_nonuniform_weights_1st_deriv_left(1e-6, 1e-2)
        @test isapprox(sum(w_stretch), 0.0, atol=1e-9)
        @test abs(w_stretch[1]) > abs(w_stretch[3]) * 1000
    end

    @testset "6. Exact Analytical Accuracy (f(x) = x^2)" begin
        # f(x) = x^2
        # At x_0 = 1.0. Next points: x_1 = 1.1 (h1=0.1), x_2 = 1.3 (h2=0.2)
        x0, x1, x2 = 1.0, 1.1, 1.3
        h1_test, h2_test = x1 - x0, x2 - x1
        
        y0, y1, y2 = x0^2, x1^2, x2^2
        
        # 1st Derivative at x0 should be 2*x0 = 2.0
        w1_c0, w1_c1, w1_c2 = get_nonuniform_weights_1st_deriv_left(h1_test, h2_test)
        deriv1_calculated = w1_c0*y0 + w1_c1*y1 + w1_c2*y2
        @test isapprox(deriv1_calculated, 2.0, atol=1e-10)

        # 2nd Derivative at x0 should be 2.0
        w2_c0, w2_c1, w2_c2 = get_nonuniform_weights_2nd_deriv_left(h1_test, h2_test)
        deriv2_calculated = w2_c0*y0 + w2_c1*y1 + w2_c2*y2
        @test isapprox(deriv2_calculated, 2.0, atol=1e-10)
    end
end

# ==========================================
# 🚀 BENCHMARKING & ZERO-ALLOCATION PROOF
# ==========================================
println("\n" * "="^60)
println("🚀 RUNNING FINAL BENCHMARKS (Time & Memory Allocation)")
println("="^60)
h1_bench, h2_bench = 0.01, 0.1
b_final = @benchmark get_nonuniform_weights_1st_deriv_left($h1_bench, $h2_bench)
display(b_final)
println("\n")