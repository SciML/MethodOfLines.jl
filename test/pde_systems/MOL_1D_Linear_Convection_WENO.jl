# 1D linear convection problem

# Packages and inclusions
using ModelingToolkit, MethodOfLines, DiffEqBase, LinearAlgebra, Test, DomainSets
#using Plots
# Tests

@testset "Test 00: Dt(u(t,x)) ~ -Dx(u(t,x))" begin
    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    # 1D PDE and boundary conditions
    eq = Dt(u(t, x)) ~ -Dx(u(t, x))

    asf(x) = (0.5 / (0.2 * sqrt(2.0 * 3.1415))) * exp(-(x - 1.0)^2 / (2.0 * 0.2^2))
    bcs = [u(0, x) ~ asf(x),
        u(t, 0) ~ u(t, 2)]

    # Space and time domains
    domains = [t ∈ Interval(0.0, 2.0),
        x ∈ Interval(0.0, 2.0)]

    # PDE system
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    # Method of lines discretization
    dx = 2 / 80
    order = 1
    discretization = MOLFiniteDifference([x => dx], t, advection_scheme = WENOScheme())

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys, discretization)

    # Solve ODE problem
    using OrdinaryDiffEq
    sol = solve(prob, SSPRK33(), dt = 0.01, saveat = 0.1)

    x_interval = sol[x][2:end]
    utrue = asf.(x_interval)
    # Plot and save results
    # plot(x_sol, u, seriestype = :scatter,label="Analytic solution")
    # plot!(x_sol, sol.u[end], label="Numeric solution")
    # plot!(x_sol, u.-sol.u[end], label="Differential Error")

    # savefig("plots/MOL_Linear_Convection_Test00.png")

    @test sol[u(t, x)][end, 2:end]≈utrue atol=0.1
end

@testset "Test 00a: Dt(u(t,x)) ~ Dx(u(t,x))" begin
    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    # 1D PDE and boundary conditions
    eq = Dt(u(t, x)) ~ Dx(u(t, x))

    asf(x) = (0.5 / (0.2 * sqrt(2.0 * 3.1415))) * exp(-(x - 1.0)^2 / (2.0 * 0.2^2))
    bcs = [u(0, x) ~ asf(x),
        u(t, 0) ~ u(t, 2)]

    # Space and time domains
    domains = [t ∈ Interval(0.0, 2.0),
        x ∈ Interval(0.0, 2.0)]

    # PDE system
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    # Method of lines discretization
    dx = 2 / 80
    order = 1
    discretization = MOLFiniteDifference([x => dx], t, advection_scheme = WENOScheme())

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys, discretization)

    # Solve ODE problem
    using OrdinaryDiffEq
    sol = solve(prob, SSPRK33(), dt = 0.01, saveat = 0.1)

    x_interval = sol[x][2:end]
    utrue = asf.(x_interval)
    # Plot and save results
    # plot(x_sol, u, seriestype = :scatter,label="Analytic solution")
    # plot!(x_sol, sol.u[end], label="Numeric solution")
    # plot!(x_sol, u.-sol.u[end], label="Differential Error")

    # savefig("plots/MOL_Linear_Convection_Test00.png")

    @test sol[u(t, x)][end, 2:end]≈utrue atol=0.1
end

@testset "Test 00b: Dt(u(t,x)) - Dx(u(t,x)) ~ 0" begin
    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    # 1D PDE and boundary conditions
    eq = Dt(u(t, x)) - Dx(u(t, x)) ~ 0

    asf(x) = (0.5 / (0.2 * sqrt(2.0 * 3.1415))) * exp(-(x - 1.0)^2 / (2.0 * 0.2^2))
    bcs = [u(0, x) ~ asf(x),
        u(t, 0) ~ u(t, 2)]

    # Space and time domains
    domains = [t ∈ Interval(0.0, 2.0),
        x ∈ Interval(0.0, 2.0)]

    # PDE system
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    # Method of lines discretization
    dx = 2 / 80
    order = 1
    discretization = MOLFiniteDifference([x => dx], t, advection_scheme = WENOScheme())

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys, discretization)

    # Solve ODE problem
    using OrdinaryDiffEq
    sol = solve(prob, SSPRK33(), dt = 0.01, saveat = 0.1)

    x_interval = sol[x][2:end]
    utrue = asf.(x_interval)
    # Plot and save results
    # plot(x_sol, u, seriestype = :scatter,label="Analytic solution")
    # plot!(x_sol, sol.u[end], label="Numeric solution")
    # plot!(x_sol, u.-sol.u[end], label="Differential Error")

    # savefig("plots/MOL_Linear_Convection_Test00.png")

    @test sol[u(t, x)][end, 2:end]≈utrue atol=0.1
end

@testset "Test 00c: Dt(u(t,x)) + Dx(u(t,x)) ~ 0" begin
    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    # 1D PDE and boundary conditions
    eq = Dt(u(t, x)) + Dx(u(t, x)) ~ 0

    asf(x) = (0.5 / (0.2 * sqrt(2.0 * 3.1415))) * exp(-(x - 1.0)^2 / (2.0 * 0.2^2))
    bcs = [u(0, x) ~ asf(x),
        u(t, 0) ~ u(t, 2)]

    # Space and time domains
    domains = [t ∈ Interval(0.0, 2.0),
        x ∈ Interval(0.0, 2.0)]

    # PDE system
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    # Method of lines discretization
    dx = 2 / 80
    order = 1
    discretization = MOLFiniteDifference([x => dx], t, advection_scheme = WENOScheme())
    # explicitly specify upwind order

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys, discretization)

    # Solve ODE problem
    using OrdinaryDiffEq
    sol = solve(prob, SSPRK33(), dt = 0.01, saveat = 0.1)

    x_interval = sol[x][2:end]
    utrue = asf.(x_interval)
    # Plot and save results
    # plot(x_sol, u, seriestype = :scatter,label="Analytic solution")
    # plot!(x_sol, sol.u[end], label="Numeric solution")
    # plot!(x_sol, u.-sol.u[end], label="Differential Error")

    # savefig("plots/MOL_Linear_Convection_Test00.png")

    @test sol[u(t, x)][end, 2:end]≈utrue atol=0.1
end

@testset "Test 01: Dt(u(t,x)) ~ -Dx(u(t,x)) + 0.001" begin

    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    # 1D PDE and boundary conditions
    eq = Dt(u(t, x)) ~ -Dx(u(t, x)) + 0.001
    asf(x) = (0.5 / (0.2 * sqrt(2.0 * 3.1415))) * exp(-(x - 1.0)^2 / (2.0 * 0.2^2))
    bcs = [u(0, x) ~ asf(x),
        u(t, 0) ~ u(t, 2)]

    # Space and time domains
    domains = [t ∈ Interval(0.0, 2.0),
        x ∈ Interval(0.0, 2.0)]

    # PDE system
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    # Method of lines discretization
    dx = 2 / 80
    order = 1
    discretization = MOLFiniteDifference([x => dx], t, advection_scheme = WENOScheme())

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys, discretization)

    # Solve ODE problem
    using OrdinaryDiffEq
    sol = solve(prob, SSPRK33(), dt = 0.01, saveat = 0.1)

    # Test
    x_interval = sol[x][2:end]
    utrue = asf.(x_interval)
    # Plot and save results
    # plot(x_sol, u, seriestype = :scatter,label="Analytic solution")
    # plot!(x_sol, sol.u[end], label="Numeric solution")
    # plot!(x_sol, u.-sol.u[end], label="Differential Error")

    # savefig("plots/MOL_Linear_Convection_Test00.png")

    @test sol[u(t, x)][end, 2:end]≈utrue atol=0.1
end

@testset "Test 02: Dt(u(t,x)) ~ -v*Dx(u(t,x))" begin
    # Parameters, variables, and derivatives
    @parameters t x v
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    v = 1.0

    # 1D PDE and boundary conditions
    eq = Dt(u(t, x)) ~ -v * Dx(u(t, x))
    asf(x) = (0.5 / (0.2 * sqrt(2.0 * 3.1415))) * exp(-(x - 1.0)^2 / (2.0 * 0.2^2))
    bcs = [u(0, x) ~ asf(x),
        u(t, 0) ~ u(t, 2)]

    # Space and time domains
    domains = [t ∈ Interval(0.0, 2.0),
        x ∈ Interval(0.0, 2.0)]

    # PDE system
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    # Method of lines discretization
    dx = 2 / 80
    order = 1
    discretization = MOLFiniteDifference([x => dx], t, advection_scheme = WENOScheme())

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys, discretization)

    # Solve ODE problemdoes this mean that - doesn't seem to affect things
    using OrdinaryDiffEq
    sol = solve(prob, SSPRK33(), dt = 0.01, saveat = 0.1)

    # Plot and save results
    # using Plots
    # plot(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,1]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,2]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,3]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,4]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,5]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,6]))
    # savefig("MOL_1D_Linear_Convection_Test02.png")

    # Test
    x_interval = sol[x][2:end]
    utrue = asf.(x_interval)
    # Plot and save results
    # plot(x_sol, u, seriestype = :scatter,label="Analytic solution")
    # plot!(x_sol, sol.u[end], label="Numeric solution")
    # plot!(x_sol, u.-sol.u[end], label="Differential Error")

    # savefig("plots/MOL_Linear_Convection_Test00.png")

    @test sol[u(t, x)][end, 2:end]≈utrue atol=0.1
end
@testset "Test 03: Dt(u(t,x)) ~ -Dx(v(t,x))*u(t,x)-v(t,x)*Dx(u(t,x)) with v(t,x)=1" begin
    # Parameters, variables, and derivatives
    @parameters t x
    @variables v(..) u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    # 1D PDE and boundary conditions
    eq = [Dt(u(t, x)) ~ -(Dx(v(t, x)) * u(t, x) + v(t, x) * Dx(u(t, x))),
        v(t, x) ~ 1.0]
    asf(x) = (0.5 / (0.2 * sqrt(2.0 * 3.1415))) * exp(-(x - 1.0)^2 / (2.0 * 0.2^2))
    bcs = [u(0, x) ~ asf(x),
        u(t, 0) ~ u(t, 2),
        v(0, x) ~ 1.0,
        v(t, 0) ~ v(t, 2)]

    # Space and time domains

    # Space and time domains
    domains = [t ∈ Interval(0.0, 2.0),
        x ∈ Interval(0.0, 2.0)]

    # PDE system
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x), v(t, x)])

    # Method of lines discretization
    dx = 2 / 80
    order = 1
    discretization = MOLFiniteDifference([x => dx], t, advection_scheme = WENOScheme())

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys, discretization)

    # Solve ODE problem
    using OrdinaryDiffEq
    sol = solve(prob, SSPRK33(), dt = 0.01, saveat = 0.1)

    #Plot and save results
    # using Plots
    # plot(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,1]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,2]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,3]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,4]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,5]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,6]))
    # savefig("MOL_1D_Linear_Convection_Test03.png")

    # Test
    x_interval = sol[x][2:end]
    utrue = asf.(x_interval)
    # Plot and save results
    # plot(x_sol, u, seriestype = :scatter,label="Analytic solution")
    # plot!(x_sol, sol.u[end], label="Numeric solution")
    # plot!(x_sol, u.-sol.u[end], label="Differential Error")

    # savefig("plots/MOL_Linear_Convection_Test00.png")

    @test sol[u(t, x)][end, 2:end]≈utrue atol=0.1
end

@testset "Test 04: Dt(u(t,x)) ~ -Dx(v(t,x))*u(t,x)-v(t,x)*Dx(u(t,x)) with v(t,x)=0.999 + 0.001 * t * x " begin
    # Parameters, variables, and derivatives
    @parameters t x
    @variables v(..) u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    # 1D PDE and boundary conditions
    eq = [Dt(u(t, x)) ~ -Dx(v(t, x)) * u(t, x) - v(t, x) * Dx(u(t, x)),
        v(t, x) ~ 0.999 + 0.001 * t * x]
    asf(x) = (0.5 / (0.2 * sqrt(2.0 * 3.1415))) * exp(-(x - 1.0)^2 / (2.0 * 0.2^2))

    bcs = [u(0, x) ~ asf(x),
        u(t, 0) ~ u(t, 2),
        v(0, x) ~ 0.999,
        v(t, 0) ~ v(t, 2)]

    # Space and time domains
    domains = [t ∈ Interval(0.0, 2.0),
        x ∈ Interval(0.0, 2.0)]

    # PDE system
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x), v(t, x)])

    # Method of lines discretization
    dx = 2 / 80
    order = 1
    discretization = MOLFiniteDifference([x => dx], t, advection_scheme = WENOScheme())

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys, discretization)

    # Solve ODE problem
    using OrdinaryDiffEq
    sol = solve(prob, SSPRK33(), dt = 0.01, saveat = 0.1)

    # Plot and save results
    # using Plots
    # plot(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,1]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,2]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,3]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,4]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,5]))
    # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,6]))
    # savefig("MOL_1D_Linear_Convection_Test04.png")

    # Test
    x_interval = sol[x][2:end]
    utrue = asf.(x_interval)
    # Plot and save results
    # plot(x_sol, u, seriestype = :scatter,label="Analytic solution")
    # plot!(x_sol, sol.u[end], label="Numeric solution")
    # plot!(x_sol, u.-sol.u[end], label="Differential Error")

    # savefig("plots/MOL_Linear_Convection_Test00.png")

    @test sol[u(t, x)][end, 2:end]≈utrue atol=0.1
end

@testset "Test 05 - Alan's Example: Dt(u(t, x)) + α * Dx(u(t, x)) ~ β * Dxx(u(t, x)) + γ * Dxxx(u(t, x)) - δ * Dxxxx(u(t, x))" begin
    @parameters t, x
    @variables u(..)

    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2
    Dxxx = Differential(x)^3
    Dxxxx = Differential(x)^4

    α = 1.1
    β = 2.1
    γ = 1.1
    δ = 3.1

    eq = Dt(u(t, x)) + α * Dx(u(t, x)) ~ β * Dxx(u(t, x)) + γ * Dxxx(u(t, x)) -
                                         δ * Dxxxx(u(t, x))
    domain = [x ∈ Interval(0.0, 2π),
        t ∈ Interval(0.0, 3.0)]

    ic_bc = [u(0.0, x) ~ cos(x)^2,
        u(t, 0.0) ~ u(t, 2π)]

    @named sys = PDESystem(eq, ic_bc, domain, [t, x], [u(t, x)])

    # Method of lines discretization
    dx = 2π / 30
    order = 2
    discretization = MOLFiniteDifference([x => dx], t, advection_scheme = WENOScheme())

    # Convert the PDE problem into an ODE problem
    prob = discretize(sys, discretization)

    # Solve ODE problem
    sol = solve(prob, Rodas4P(), saveat = 0.01)

    asf(t, x) = 0.5 * (exp(-t * 4(β + 4δ)) * cos(t * (-8γ - 2α) + 2x) + 1)

    solu = sol[u(t, x)]

    x_grid = sol[x]
    t_grid = sol[t]

    exact = [asf(t, x) for t in t_grid, x in x_grid]
    for i in eachindex(t_grid)
        norm_exact = exact[i, :] ./ maximum(exact[i, :])
        norm_usol = solu[i, :] ./ maximum(solu[i, :])
        @test norm_exact≈norm_usol atol=0.1
    end
end
# @testset "Test 05: Dt(u(t,x)) ~ -Dx(v(t,x)*u(t,x)) with v(t, x) ~ 1.0" begin
#     # Parameters, variables, and derivatives
#     @parameters t x
#     @variables v(..) u(..)
#     Dt = Differential(t)
#     Dx = Differential(x)

#     # 1D PDE and boundary conditions
#     eq = [Dt(u(t, x)) ~ -Dx(v(t, x) * u(t, x)),
#         v(t, x) ~ 1.0 ]
#     asf(x) = (0.5 / (0.2 * sqrt(2.0 * 3.1415))) * exp(-(x - 1.0)^2 / (2.0 * 0.2^2))

#     bcs = [u(0, x) ~ asf(x),
#         u(t, 0) ~ u(t, 2),
#         v(0, x) ~ v(t, 2)]

#     # Space and time domains
#     domains = [t ∈ Interval(0.0, 2.0),
#         x ∈ Interval(0.0, 2.0)]

#     # PDE system
#     @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x), v(t, x)])

#     # Method of lines discretization
#     dx = 2 / 80
#     order = 1
#     discretization = MOLFiniteDifference([x => dx], t, advection_scheme=WENOScheme())

#     # Convert the PDE problem into an ODE problem
#     prob = discretize(pdesys, discretization)

#     # Solve ODE problem
#     using OrdinaryDiffEq
#     sol = solve(prob, FBDF(), saveat=0.1)

#     # Plot and save results
#     # using Plots
#     # plot(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,1]))
#     # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,2]))
#     # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,3]))
#     # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,4]))
#     # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,5]))
#     # plot!(prob.space[2],Array(prob.extrapolation[1]*sol[:,1,6]))
#     # savefig("MOL_1D_Linear_Convection_Test04.png")

#     # Test
#     x_interval = sol[x][2:end]
#     utrue = asf.(x_interval)
#     # Plot and save results
#     # plot(x_sol, u, seriestype = :scatter,label="Analytic solution")
#     # plot!(x_sol, sol.u[end], label="Numeric solution")
#     # plot!(x_sol, u.-sol.u[end], label="Differential Error")

#     # savefig("plots/MOL_Linear_Convection_Test00.png")

#     @test sol[u(t, x)][end, 2:end] ≈ utrue atol = 0.1
# end
