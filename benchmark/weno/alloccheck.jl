# Static allocation-freedom audit of the WENO kernels (AllocCheck).
# Local-only: AllocCheck's GPUCompiler/LLVM dependency is too heavy for the CI matrix;
# CI retains the @allocated checks in test/Components/. Error-throw paths are excluded
# (AllocCheck default ignore_throw = true).
# Usage: julia benchmark/weno/alloccheck.jl
#        julia benchmark/run_benchmarks.jl --alloccheck

if abspath(PROGRAM_FILE) == @__FILE__
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

using Test
using MethodOfLines
using AllocCheck

const V64 = Vector{Float64}
# Coordinate view type MOL passes on uniform grids.
const XVIEW = typeof(@view StepRangeLen(0.0, 0.1, 100)[48:52])

alloccheck_results = @testset "WENO kernels are statically allocation-free" begin
    @testset "uniform kernel (scalar dx)" begin
        @test isempty(check_allocs(MethodOfLines.weno_f, (V64, V64, Float64, XVIEW, Float64)))
    end

    @testset "non-uniform kernel (vector dx, center target)" begin
        @test isempty(check_allocs(MethodOfLines.weno_f, (V64, V64, Float64, V64, V64)))
    end

    @testset "non-uniform boundary targets" begin
        for T in (1, 2, 4, 5)
            b = MethodOfLines.WENONonUniformBoundary{T}()
            @test isempty(check_allocs(b, (V64, V64, Float64, V64, V64)))
        end
    end

    @testset "Fornberg 3-point weight helper" begin
        @test isempty(
            check_allocs(MethodOfLines._fornberg3_weights, (NTuple{3, Float64}, Float64))
        )
    end
end
