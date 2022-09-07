# 1d wave equation
# Packages and inclusions
using ModelingToolkit, MethodOfLines, LinearAlgebra, Test, OrdinaryDiffEq, DomainSets
using ModelingToolkit: Differential

@testset "Test 00: Test solution interface, time dependent" begin
    # Parameters, variables, and derivatives
    @parameters t x y
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dy = Differential(y)

    # 1D PDE and boundary conditions
    eq = Dt(u(t, x, y)) ~ -Dx(u(t, x, y)) + Dy(u(t, x, y))

    asf(x, y) = (0.5 / (0.2 * sqrt(2.0 * 3.1415))) * exp(-(x + y - 1.0)^2 / (2.0 * 0.2^2))
    bcs = [u(0, x, y) ~ asf(x, y),
        u(t, 0, y) ~ u(t, 2, y),
        u(t, x, 0) ~ u(t, x, 2)]

    # Space and time domains
    domains = [t ∈ Interval(0.0, 2.0),
        x ∈ Interval(0.0, 2.0),
        y ∈ Interval(0.0, 2.0)]

    # PDE system
    @named pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])

    # Method of lines discretization
    dx = dy = 2 / 8
    order = 1
    discretization = MOLFiniteDifference([x => dx, y => dy], t; advection_scheme=WENOScheme())
    # explicitly specify upwind order

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys, discretization)

    # Solve ODE problem
    using OrdinaryDiffEq
    sol = solve(prob, Euler(), dt=0.025, saveat=0.1)

    grid = get_discrete(pdesys, discretization)


    traditional_sol = zeros(Float64, length(sol.t), 9, 9)

    solu = sol.original_sol[grid[u(t, x, y)]]

    for i in 1:length(sol.t)
        for j in 1:9
            for k in 1:9
                traditional_sol[i, j, k] = solu[j, k][i]
            end
        end
    end


    @test sol[u(t, x, y)] == traditional_sol

    @test sol[1] isa Float64
    @test sol[:] isa Vector{Float64}
    @test sol[1:2] isa Array{Float64,1}
    @test sol[[1, 2, 3]] isa Array{Float64,1}
    @test sol(pi / 2, pi / 2, pi / 2) isa Float64
    @test sol(pi / 2, pi / 2, :) isa Vector{Float64}
    @test sol(pi / 2, :, :) isa Matrix{Float64}

    @test sol[t] == sol.t
    @test (sol[x] == sol[y])
    @test (sol[y] isa StepRangeLen)

    @variables u[1:9, 1:9](..)

    @test sol[u[1, 2](t, x, y)] isa Float64
    @test sol[u[5, 5](t, x, y)] isa Float64
end

@testset "Test 00: Test solution interface, time independent" begin
    @parameters x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ 0
    dx = 1 / 8
    dy = 1 / 8

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
    discretization = MOLFiniteDifference([x => dx, y => dy], nothing, approx_order=2)

    prob = discretize(pdesys, discretization)
    sol = NonlinearSolve.solve(prob, NewtonRaphson())
    grid = get_discrete(pdesys, discretization)

    solu = sol.original_sol[grid[u(x, y)]]

    @test sol[u(x, y)] == solu

    @test sol(pi / 4, pi / 4) isa Float64
    @test sol(pi / 4, :) isa Vector{Float64}

    @test (sol[x] == sol[y])
    @test (sol[y] isa StepRangeLen)
end
