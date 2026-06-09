using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using MethodOfLines, Aqua, JET, Test

@testset "Aqua" begin
    Aqua.test_all(MethodOfLines)
end

@testset "JET" begin
    JET.test_package(MethodOfLines; target_defined_modules = true)
end
