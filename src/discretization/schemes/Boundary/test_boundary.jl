# test_boundary.jl
using Test
using ForwardDiff
using BenchmarkTools

include("boundary_weights.jl")

@testset "Non-Uniform Boundary Stencils (End-to-End)" begin
    h1, h2 = 0.01, 0.1

    @testset "1. Conservation of Mass" begin
        @test isapprox(sum(get_nonuniform_weights_1st_deriv_left(h1, h2)), 0.0, atol=1e-12)
        @test isapprox(sum(get_nonuniform_weights_2nd_deriv_left(h1, h2)), 0.0, atol=1e-10)
    end

    @testset "2. Symmetry Check" begin
        w_1st_left = get_nonuniform_weights_1st_deriv_left(h1, h2)
        w_1st_right = get_nonuniform_weights_1st_deriv_right(h1, h2)
        @test all(w_1st_left .== .-w_1st_right)
        
        w_2nd_left = get_nonuniform_weights_2nd_deriv_left(h1, h2)
        w_2nd_right = get_nonuniform_weights_2nd_deriv_right(h1, h2)
        @test all(w_2nd_left .== w_2nd_right)
    end

    @testset "3. Guardrails (Zero/Negative)" begin
        @test_throws AssertionError get_nonuniform_weights_1st_deriv_left(0.0, 0.1)
    end

    @testset "4. Type Promotion (Float64 + Float32)" begin
        w_mixed = get_nonuniform_weights_1st_deriv_left(0.01, 0.1f0)
        @test eltype(w_mixed) == Float64
        @test isapprox(sum(w_mixed), 0.0, atol=1e-7)
    end

    @testset "5. ForwardDiff (AD)" begin
        w_ad = get_nonuniform_weights_1st_deriv_right(ForwardDiff.Dual(0.01, 1.0), ForwardDiff.Dual(0.1, 0.0))
        @test eltype(w_ad) <: ForwardDiff.Dual
    end

    @testset "6. Extreme Grid Stretching (1:10000)" begin
        w_stretch = get_nonuniform_weights_1st_deriv_left(1e-6, 1e-2)
        @test isapprox(sum(w_stretch), 0.0, atol=1e-9)
    end

    @testset "7. Analytical Accuracy (MMS: f(x)=x^2)" begin
        x0, x1, x2 = 1.0, 1.1, 1.3
        y0, y1, y2 = x0^2, x1^2, x2^2
        
        w1 = get_nonuniform_weights_1st_deriv_left(x1-x0, x2-x1)
        @test isapprox(w1[1]*y0 + w1[2]*y1 + w1[3]*y2, 2.0, atol=1e-10)

        w2 = get_nonuniform_weights_2nd_deriv_left(x1-x0, x2-x1)
        @test isapprox(w2[1]*y0 + w2[2]*y1 + w2[3]*y2, 2.0, atol=1e-10)
    end
end

println("\n" * "="^60)
println("🚀 BENCHMARKS (Time & Allocations)")
println("="^60)
h1_bench, h2_bench = 0.01, 0.1
display(@benchmark get_nonuniform_weights_1st_deriv_left($h1_bench, $h2_bench))
println("\n")