# 1D diffusion problem

# Packages and inclusions
using ModelingToolkit, MethodOfLines, LinearAlgebra, Test
using ModelingToolkit: Differential, get_unknowns
using NonlinearSolve
using DomainSets

# Handrolled discrete laplace problem
@testset "Hand rolled 1D Laplace" begin
    @variables a b c d
    eqs = [0 ~ a - 1,
        0 ~ a + c - 2 * b,
        0 ~ b + d - 2 * c,
        0 ~ d - 1]
    @named ns = NonlinearSystem(eqs, [a, b, c, d], [])
    sns = structural_simplify(ns)
    prob = NonlinearProblem(sns, zeros(length(get_unknowns(sns))), [])

    sol = NonlinearSolve.solve(prob, NewtonRaphson())

    @test map(v -> sol[v], [a, b, c, d]) ≈ ones(4)
end 

# Laplace's Equation, same as above but with MOL discretization
@testset "1D Laplace - constant solution" begin
    @parameters x
    @variables u(..)
    Dxx = Differential(x)^2

    eq = Dxx(u(x)) ~ 0
    dx = 1 / 3

    bcs = [u(0) ~ 1,
        u(1) ~ 1]

    # Space and time domains
    domains = [x ∈ Interval(0.0, 1.0)]

    @named pdesys = PDESystem([eq], bcs, domains, [x], [u(x)])
    discretization = MOLFiniteDifference([x => dx], nothing, approx_order = 2)
    prob = discretize(pdesys, discretization)

    sol = NonlinearSolve.solve(prob, NewtonRaphson())
    sol′ = sol[u(x)]
    @test sol′ ≈ ones(4)
end

# Laplace's Equation, linear solution
@testset "1D Laplace - linear solution" begin
    @parameters x
    @variables u(..)
    Dxx = Differential(x)^2

    eq = Dxx(u(x)) ~ 0
    dx = 0.1

    bcs = [u(0) ~ 1,
        u(1) ~ 2]

    # Space and time domains
    domains = [x ∈ Interval(0.0, 1.0)]

    @named pdesys = PDESystem([eq], bcs, domains, [x], [u(x)])
    discretization = MOLFiniteDifference([x => dx], nothing, approx_order = 2)

    prob = discretize(pdesys, discretization)
    sol = NonlinearSolve.solve(prob, NewtonRaphson())
    sol′ = sol[u(x)]

    @test sol′ ≈ 1.0:0.1:2.0
end

# 2D heat
@testset "2D heat equation" begin
    @parameters x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ 0
    dx = 0.1
    dy = 0.1

    bcs = [u(0, y) ~ x * y,
        u(1, y) ~ x * y,
        u(x, 0) ~ x * y,
        u(x, 1) ~ x * y]

    # Space and time domains
    domains = [x ∈ Interval(0.0, 1.0),
        y ∈ Interval(0.0, 1.0)]

    @named pdesys = PDESystem([eq], bcs, domains, [x, y], [u(x, y)])

    # Note that we pass in `nothing` for the time variable `t` here since we
    # are creating a stationary problem without a dependence on time, only space.
    discretization = MOLFiniteDifference([x => dx, y => dy], nothing, approx_order = 2)

    prob = discretize(pdesys, discretization)
    sol = NonlinearSolve.solve(prob, NewtonRaphson())

    xs = sol[x]
    ys = sol[y]
    u_sol = sol[u(x, y)]

    # test boundary
    @test all(abs.(u_sol[:, 1]) .< eps(Float32))
    @test all(abs.(u_sol[1, :]) .< eps(Float32))
    @test u_sol[2:(end - 1), end] ≈ (0.0:dy:1.0)[2:(end - 1)]
    @test u_sol[end, 2:(end - 1)] ≈ (0.0:dx:1.0)[2:(end - 1)]

    # test interior with finite differences
    interior = CartesianIndices((axes(xs)[1], axes(ys)[1]))[2:(end - 1), 2:(end - 1)]
    fd = map(interior) do I
        abs(u_sol[(I - CartesianIndex(1, 0))] + u_sol[(I + CartesianIndex(1, 0))] -
            2 * u_sol[I]) < 0.01
    end#
    @test all(fd)
end
