# UNIT TESTS: Non-Uniform WENO Mathematical Engine
# Ensures strict adherence to WENO mathematical properties (e.g., Partition of Unity)

using Test
include("weno_weights.jl")
using .WENOWeights

@testset "Non-Uniform WENO Engine Test Suite" begin

    # Define a clustered non-uniform sub-stencil
    x_local = [0.0, 0.05, 0.2]
    x_eval = 0.1

    @testset "1. Ideal Weights (Partition of Unity)" begin
        d_k = get_nonuniform_ideal_weights(x_local, x_eval)

        # Mathematical Law: Sum of ideal Lagrange weights must be exactly 1.0
        @test sum(d_k) ≈ 1.0 atol=1e-14
    end

    @testset "2. Dynamic Smoothness Indicators (β_k)" begin
        beta_k = get_nonuniform_smoothness_indicators(x_local)

        # Must return exactly 3 dynamic indicators for a 3-point stencil
        @test length(beta_k) == 3
        # Indicators must be strictly positive real numbers
        @test all(beta_k .> 0.0)
    end

    @testset "3. Final Non-Linear Weights (ω_k)" begin
        # Generate components
        d_k = get_nonuniform_ideal_weights(x_local, x_eval)
        beta_k = get_nonuniform_smoothness_indicators(x_local)

        # Calculate final non-linear weights
        omega_k = get_split_nonlinear_weights(d_k, beta_k)

        # Mathematical Law 1: The final non-linear weights MUST sum exactly to 1.0
        @test sum(omega_k) ≈ 1.0 atol=1e-14
    end
end