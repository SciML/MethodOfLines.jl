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

@testset "Sanity Check: Analytical Reduction to Uniform WENO" begin
    # ==============================================================================
    # MATHEMATICAL INVARIANT TEST
    # Proves that the Shi-Hu-Shu (2002) non-linear dynamic weights analytically 
    # reduce to the ideal linear Lagrange weights when the grid is perfectly uniform.
    # This guarantees zero mathematical regression for legacy uniform grids.
    # ==============================================================================

    # 1. Perfectly Uniform Grid Configuration
    h = 0.1
    x_uni = collect(0.0:h:2h) # 3-point stencil [0.0, 0.1, 0.2]
    x_eval = 0.05

    # 2. Legacy Linear Weights
    d_k = get_nonuniform_ideal_weights(x_uni, x_eval)

    # 3. Dynamic Smoothness Indicators
    beta_k = get_nonuniform_smoothness_indicators(x_uni)

    # INVARIANT 1: On a uniform grid, all smoothness indicators MUST equalize.
    @test isapprox(beta_k[1], beta_k[2], atol=1e-14)
    @test isapprox(beta_k[2], beta_k[3], atol=1e-14)

    # 4. Shi-Hu-Shu Non-Linear Weights
    omega_k = get_split_nonlinear_weights(d_k, beta_k)

    # INVARIANT 2: The non-linear splitting must perfectly cancel out,
    # reducing omega_k exactly to the ideal weights d_k down to machine precision.
    @test isapprox(omega_k, d_k, atol=1e-14)
end