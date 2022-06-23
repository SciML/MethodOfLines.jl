using MethodOfLines, Test, ModelingToolkit, SymbolicUtils

@testset "count differentials 1D" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)

    Dx = Differential(x)
    eq = Dt(u(t, x)) ~ -Dx(u(t, x))
    @test first(MethodOfLines.differential_order(eq.rhs, x.val)) == 1
    @test isempty(MethodOfLines.differential_order(eq.rhs, t.val))
    @test first(MethodOfLines.differential_order(eq.lhs, t.val)) == 1
    @test isempty(MethodOfLines.differential_order(eq.lhs, x.val))

    Dxx = Differential(x)^2
    eq = Dt(u(t, x)) ~ Dxx(u(t, x))
    @test first(MethodOfLines.differential_order(eq.rhs, x.val)) == 2
    @test isempty(MethodOfLines.differential_order(eq.rhs, t.val))
    @test first(MethodOfLines.differential_order(eq.lhs, t.val)) == 1
    @test isempty(MethodOfLines.differential_order(eq.lhs, x.val))

    Dxxxx = Differential(x)^4
    eq = Dt(u(t, x)) ~ -Dxxxx(u(t, x))
    @test first(MethodOfLines.differential_order(eq.rhs, x.val)) == 4
    @test isempty(MethodOfLines.differential_order(eq.rhs, t.val))
    @test first(MethodOfLines.differential_order(eq.lhs, t.val)) == 1
    @test isempty(MethodOfLines.differential_order(eq.lhs, x.val))
end

@testset "count differentials 2D" begin
    @parameters t x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    Dt = Differential(t)

    eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y))
    @test first(MethodOfLines.differential_order(eq.rhs, x.val)) == 2
    @test first(MethodOfLines.differential_order(eq.rhs, y.val)) == 2
    @test isempty(MethodOfLines.differential_order(eq.rhs, t.val))
    @test first(MethodOfLines.differential_order(eq.lhs, t.val)) == 1
    @test isempty(MethodOfLines.differential_order(eq.lhs, x.val))
    @test isempty(MethodOfLines.differential_order(eq.lhs, y.val))
end

@testset "count with mixed terms" begin
    @parameters t x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    Dx = Differential(x)
    Dy = Differential(y)
    Dt = Differential(t)

    eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y)) + Dx(Dy(u(t, x, y)))
    @test MethodOfLines.differential_order(eq.rhs, x.val) == Set([2, 1])
    @test MethodOfLines.differential_order(eq.rhs, y.val) == Set([2, 1])
end

@testset "Kuramoto–Sivashinsky equation" begin
    @parameters x, t
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dx2 = Differential(x)^2
    Dx3 = Differential(x)^3
    Dx4 = Differential(x)^4

    α = 1
    β = 4
    γ = 1
    eq =
        Dt(u(x, t)) +
        u(x, t) * Dx(u(x, t)) +
        α * Dx2(u(x, t)) +
        β * Dx3(u(x, t)) +
        γ * Dx4(u(x, t)) ~ 0
    @test MethodOfLines.differential_order(eq.lhs, x.val) == Set([4, 3, 2, 1])
end

@test_broken begin #@testset "Flatten division" begin
    @parameters x y z t

    #@test_broken isequal(operation(((y^(-1.0+eps(Float64)))*x~0).lhs), *)

    @test_broken isequal(operation(MethodOfLines.flatten_division((x / z ~ 0).lhs)), *)
    @test_broken isequal(operation(MethodOfLines.flatten_division((x * y / z ~ 0).lhs)), *)
    @test_broken isequal(
        operation(MethodOfLines.flatten_division((x / (y * z) ~ 0).lhs)),
        *,
    )
    @test_broken isequal(
        operation(MethodOfLines.flatten_division((x * y / (z * t) ~ 0).lhs)),
        *,
    )

end

@testset "Split terms" begin
    @parameters t x y r
    @variables u(..) v(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dr = Differential(r)
    Dxx = Differential(x)^2

    t_min = 0.0
    t_max = 2.0
    x_min = 0.0
    x_max = 2.0
    y_min = 0.0
    y_max = 2.0

    analytic_sol_func(t, x, y) = exp(x + y) * cos(x + y + 4t)

    bcs = [
        u(0, x) ~ -x * (x - 1) * sin(x),
        v(0, x) ~ -x * (x - 1) * sin(x),
        u(t, 0) ~ 0.0,
        u(t, 1) ~ 0.0,
        v(t, 0) ~ 0.0,
        v(t, 1) ~ 0.0,
        u(t_min, x, y) ~ analytic_sol_func(t_min, x, y),
        u(t, x_min, y) ~ analytic_sol_func(t, x_min, y),
        u(t, x_max, y) ~ analytic_sol_func(t, x_max, y),
        u(t, x, y_min) ~ analytic_sol_func(t, x, y_min),
        u(t, x, y_max) ~ analytic_sol_func(t, x, y_max),
        u(0, r) ~ -r * (r - 1) * sin(r),
        Dr(u(t, 0)) ~ 0.0,
        u(t, 1) ~ sin(1),
    ]

    for bc in bcs
        terms = MethodOfLines.split_terms(bc)
        @test terms isa Vector

    end
end

@testset "Periodic Wraparound" begin
    I = CartesianIndex(2, 5)
    @test MethodOfLines._wrapperiodic(I, 2, 2, 4) == CartesianIndex(2, 2)

    I = CartesianIndex(1, 4)
    @test MethodOfLines._wrapperiodic(I, 2, 1, 4) == CartesianIndex(4, 4)

    I = CartesianIndex(-1, 2)
    @test MethodOfLines._wrapperiodic(I, 2, 1, 4) == CartesianIndex(2, 2)
end
