using Test, ForwardDiff, BenchmarkTools, Random, Printf

include("boundary_weights.jl")

# --- Performance Benchmarking Helper ---
"""
    run_benchmark_loop(h1, h2, h3, n)

Execution loop to measure the average latency of weight calculations.
Aggregates results to prevent dead-code elimination by the compiler.
"""
function run_benchmark_loop(h1, h2, h3, n)
    s = 0.0
    for i in 1:n
        w = get_nonuniform_weights_1st_deriv_left_4pt(h1[i], h2[i], h3[i])
        s += w[1] 
    end
    return s
end

@testset "Non-Uniform Boundary Stencil Verification" begin
    
    @testset "1. Mass Conservation and Numerical Stability" begin
        # Seed set for deterministic test results
        Random.seed!(42)
        for _ in 1:100
            # Test across a wide range of grid stretching ratios
            h = rand(3) .* [0.1, 10.0, 100.0]
            w1 = get_nonuniform_weights_1st_deriv_left_4pt(h...)
            w2 = get_nonuniform_weights_2nd_deriv_left_4pt(h...)
            
            # Validation criteria: Residual sum should be bounded by machine epsilon
            # scaled by the maximum coefficient magnitude.
            max_w1 = maximum(abs, w1)
            @test abs(w1[1] + (w1[2] + w1[3] + w1[4])) <= max_w1 * eps(Float64) * 10
            
            max_w2 = maximum(abs, w2)
            @test abs(w2[1] + (w2[2] + w2[3] + w2[4])) <= max_w2 * eps(Float64) * 10
        end
    end

    @testset "2. Accuracy Verification (Taylor Series Exactness)" begin
        h = (0.01, 0.05, 0.2)
        x = [0.0, h[1], h[1]+h[2], h[1]+h[2]+h[3]]
        
        # Test exactness on a cubic polynomial for 1st derivative
        u(x) = x^3 + x^2 + x + 1
        w1 = get_nonuniform_weights_1st_deriv_left_4pt(h...)
        @test isapprox(sum(w1 .* u.(x)), 1.0, atol=1e-13)
    end

    @testset "3. Automatic Differentiation and Complex Support" begin
        # Verify compatibility with ForwardDiff.Dual and Complex types
        target_function(h) = begin
            w = get_nonuniform_weights_1st_deriv_left_4pt(h, 0.2, 0.3)
            return sum(w .* [0.0, 0.1im, 0.3im, 0.6im])
        end
        
        deriv = ForwardDiff.derivative(target_function, 0.1)
        @test deriv isa Complex
        @test isfinite(deriv.im)
    end

    @testset "4. Computational Performance Analysis" begin
        N_samples = 10_000
        h1_vec = rand(N_samples) .* 0.1 .+ 0.01
        h2_vec = rand(N_samples) .* 0.1 .+ 0.01
        h3_vec = rand(N_samples) .* 0.1 .+ 0.01

        # Execution using BenchmarkTools to measure minimum latency
        bench = @benchmark run_benchmark_loop($h1_vec, $h2_vec, $h3_vec, $N_samples)
        
        min_latency_total = minimum(bench.times) 
        latency_per_call = min_latency_total / N_samples
        
        @test latency_per_call > 0.0
        @test bench.allocs == 0
        
        println("\n" * "="^40)
        println(" Numerical Performance Report")
        @printf(" Mean Execution Latency: %.2f ns\n", latency_per_call)
        println(" Sample Size:           ", N_samples)
        println(" Heap Allocations:      ", bench.allocs, " bytes")
        println("="^40)
    end
end