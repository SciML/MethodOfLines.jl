using Test
using Symbolics
using ForwardDiff

include("weno_symbolic.jl")
using .WENOSymbolic

@testset "WENOSymbolic Module: The Final Crucible (Ultimate Stress Test)" begin
    @variables u[1:5] x[1:5] x_eval

    @testset "1. AST Generation and Boundary Fallbacks" begin
        ast_valid = generate_safe_weno_ast(collect(u)[1:3], collect(x)[1:3], x_eval)
        @test ast_valid isa Num

        ast_fallback = generate_safe_weno_ast(collect(u)[1:4], collect(x)[1:4], x_eval)
        @test ast_fallback isa Num

        @test_throws ArgumentError generate_safe_weno_ast(collect(u)[1:2], collect(x)[1:2], x_eval)
    end

    @testset "2. Mathematical Invariants (Partition of Unity)" begin
        ast_rule = generate_safe_weno_ast(collect(u)[1:3], collect(x)[1:3], x_eval)
        f_compiled = build_function(ast_rule, collect(u)[1:3], collect(x)[1:3], x_eval, expression=Val{false})

        u_const = [7.5, 7.5, 7.5]
        x_rand = [0.0, 0.13, 2.9]
        x_target = 0.8

        result = f_compiled(u_const, x_rand, x_target)
        @test result ≈ 7.5 atol=1e-12
    end

    @testset "3. Extreme Grid Stretching (Shi-Hu-Shu Split Trigger)" begin
        ast_rule = generate_safe_weno_ast(collect(u)[1:3], collect(x)[1:3], x_eval)
        f_compiled = build_function(ast_rule, collect(u)[1:3], collect(x)[1:3], x_eval, expression=Val{false})

        u_test = [1.0, 2.5, 4.0]
        x_extreme = [0.0, 0.000001, 100.0] 
        x_target = 0.0000005

        result = f_compiled(u_test, x_extreme, x_target)
        @test isfinite(result)
        @test typeof(result) == Float64
    end

    @testset "4. Type Stability & Float32 GPU Readiness" begin
        ast_rule = generate_safe_weno_ast(collect(u)[1:3], collect(x)[1:3], x_eval)
        f_compiled = build_function(ast_rule, collect(u)[1:3], collect(x)[1:3], x_eval, expression=Val{false})

        u_f32 = Float32[1.0, 2.5, 4.0]
        x_f32 = Float32[0.0, 0.4, 1.1]
        x_target_f32 = Float32(0.5)

        result_f32 = f_compiled(u_f32, x_f32, x_target_f32)
        @test typeof(result_f32) == Float32

        f_compiled(u_f32, x_f32, x_target_f32) # Warm-up
        allocs_f32 = @allocated f_compiled(u_f32, x_f32, x_target_f32)
        @test allocs_f32 == 0
    end

    @testset "5. ForwardDiff Jacobian Compatibility (Stiff PDE Ready)" begin
        ast_rule = generate_safe_weno_ast(collect(u)[1:3], collect(x)[1:3], x_eval)
        f_compiled = build_function(ast_rule, collect(u)[1:3], collect(x)[1:3], x_eval, expression=Val{false})

        x_fixed = [0.0, 0.4, 1.1]
        x_t = 0.5

        flux_wrapper(u_vec) = f_compiled(u_vec, x_fixed, x_t)

        u_test = [1.0, 2.5, 4.0]

        jac = ForwardDiff.gradient(flux_wrapper, u_test)

        @test length(jac) == 3
        @test all(isfinite, jac)
    end

    @testset "6. NaN/Inf Propagation Resilience" begin
        ast_rule = generate_safe_weno_ast(collect(u)[1:3], collect(x)[1:3], x_eval)
        f_compiled = build_function(ast_rule, collect(u)[1:3], collect(x)[1:3], x_eval, expression=Val{false})

        u_nan = [1.0, NaN, 4.0]
        x_norm = [0.0, 0.4, 1.1]

        result_nan = f_compiled(u_nan, x_norm, 0.5)
        @test isnan(result_nan)
    end
end