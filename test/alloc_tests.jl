# Allocation tests for critical performance-sensitive functions
using Test
using MethodOfLines

@testset "Allocation Tests - Fornberg Weight Calculation" begin
    # Test that the Fornberg algorithm has stable allocation behavior
    # Note: We can't achieve zero allocations due to the nature of the algorithm,
    # but we can ensure allocations scale appropriately with input size

    # Small stencil
    x_small = collect(Float64, -1:1)
    alloc_small_1 = @allocated MethodOfLines.calculate_weights(1, 0.0, x_small)
    alloc_small_2 = @allocated MethodOfLines.calculate_weights(2, 0.0, x_small)

    # Medium stencil
    x_med = collect(Float64, -2:2)
    alloc_med_1 = @allocated MethodOfLines.calculate_weights(1, 0.0, x_med)
    alloc_med_2 = @allocated MethodOfLines.calculate_weights(2, 0.0, x_med)

    # Large stencil
    x_large = collect(Float64, -4:4)
    alloc_large_1 = @allocated MethodOfLines.calculate_weights(1, 0.0, x_large)
    alloc_large_2 = @allocated MethodOfLines.calculate_weights(2, 0.0, x_large)

    # Allocations should scale roughly quadratically with stencil size (due to matrix allocation)
    # This is a regression test - if allocations grow unexpectedly, it indicates a problem
    @test alloc_small_1 < 2000  # Should be around 400-800 bytes
    @test alloc_small_2 < 2000
    @test alloc_med_1 < 3000
    @test alloc_med_2 < 3000
    @test alloc_large_1 < 5000
    @test alloc_large_2 < 5000

    # Test that Hermite-based weights have reasonable allocations
    x_hermite = collect(Float64, -2:2)
    alloc_hermite = @allocated MethodOfLines.calculate_weights(2, 0.0, x_hermite; dfdx = true)
    @test alloc_hermite < 20000  # Hermite method allocates more due to matrix operations
end

@testset "Allocation Regression Tests - Core Functions" begin
    # These tests ensure that key internal functions maintain stable allocation behavior
    # The absolute values may change, but sudden increases indicate regressions

    # Test half_range utility
    alloc_hr = @allocated MethodOfLines.half_range(5)
    @test alloc_hr == 0  # This should be allocation-free

    # Test safe_vcat with empty arrays (common case)
    a = []
    b = []
    alloc_empty = @allocated MethodOfLines.safe_vcat(a, b)
    @test alloc_empty < 100  # Should allocate minimal for empty result

    # Test safe_vcat with one empty array
    c = [1, 2, 3]
    alloc_one_empty = @allocated MethodOfLines.safe_vcat(a, c)
    @test alloc_one_empty < 100  # Should just return c without allocation
end
