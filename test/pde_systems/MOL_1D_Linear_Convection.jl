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
    discretization = MOLFiniteDifference([x => dx], t)
    # explicitly specify upwind order
    discretization_upwind = MOLFiniteDifference([x => dx], t; upwind_order=order)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys, discretization)
    prob_upwind = discretize(pdesys, discretization_upwind)

    # Solve ODE problem
    using OrdinaryDiffEq
    sol = solve(prob, Euler(), dt=0.025, saveat=0.1)
    sol_upwind = solve(prob_upwind, Euler(), dt=0.025, saveat=0.1)

    x_interval = infimum(domains[2].domain)+dx:dx:supremum(domains[2].domain)
    u = asf.(x_interval)
    # Plot and save results

    # Test
    t_f = size(sol, 3)

    x_sol = x_interval
    t_f = size(sol, 3)
    exact = u
    # plot(x_sol, u, seriestype = :scatter,label="Analytic solution")
    # plot!(x_sol, sol.u[end], label="Numeric solution")
    # plot!(x_sol, u.-sol.u[end], label="Differential Error")

    # savefig("plots/MOL_Linear_Convection_Test00.png")


    @test sol.u[end] ≈ u atol = 0.1
    @test sol_upwind.u[end] ≈ u atol = 0.1
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
    discretization = MOLFiniteDifference([x => dx], t)
    # explicitly specify upwind order
    discretization_upwind = MOLFiniteDifference([x => dx], t; upwind_order=order)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys, discretization)
    prob_upwind = discretize(pdesys, discretization_upwind)

    # Solve ODE problem
    using OrdinaryDiffEq
    sol = solve(prob, Euler(), dt=0.025, saveat=0.1)
    sol_upwind = solve(prob_upwind, Euler(), dt=0.025, saveat=0.1)

    x_interval = infimum(domains[2].domain)+dx:dx:supremum(domains[2].domain)
    u = asf.(x_interval)
    # Plot and save results

    # Test
    t_f = size(sol, 3)

    x_sol = x_interval
    t_f = size(sol, 3)
    exact = u
    # plot(x_sol, u, seriestype = :scatter,label="Analytic solution")
    # plot!(x_sol, sol.u[end], label="Numeric solution")
    # plot!(x_sol, u.-sol.u[end], label="Differential Error")

    # savefig("plots/MOL_Linear_Convection_Test00.png")


    @test sol[u(t, x)][end, :] ≈ u atol = 0.1
    @test sol_upwind[u(t, x)][end, :] ≈ u atol = 0.1
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
    discretization = MOLFiniteDifference([x => dx], t)
    # explicitly specify upwind order
    discretization_upwind = MOLFiniteDifference([x => dx], t; upwind_order=order)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys, discretization)
    prob_upwind = discretize(pdesys, discretization_upwind)

    # Solve ODE problem
    using OrdinaryDiffEq
    sol = solve(prob, Euler(), dt=0.025, saveat=0.1)
    sol_upwind = solve(prob_upwind, Euler(), dt=0.025, saveat=0.1)

    x_interval = infimum(domains[2].domain)+dx:dx:supremum(domains[2].domain)
    u = asf.(x_interval)
    # Plot and save results

    # Test
    t_f = size(sol, 3)

    x_sol = x_interval
    t_f = size(sol, 3)
    exact = u
    # plot(x_sol, u, seriestype = :scatter,label="Analytic solution")
    # plot!(x_sol, sol.u[end], label="Numeric solution")
    # plot!(x_sol, u.-sol.u[end], label="Differential Error")

    # savefig("plots/MOL_Linear_Convection_Test00.png")


    @test sol[u(t, x)][end, :] ≈ u atol = 0.1
    @test sol_upwind[u(t, x)][end, :] ≈ u atol = 0.1
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
    discretization = MOLFiniteDifference([x => dx], t)
    # explicitly specify upwind order
    discretization_upwind = MOLFiniteDifference([x => dx], t; upwind_order=order)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys, discretization)
    prob_upwind = discretize(pdesys, discretization_upwind)

    # Solve ODE problem
    using OrdinaryDiffEq
    sol = solve(prob, Euler(), dt=0.025, saveat=0.1)
    sol_upwind = solve(prob_upwind, Euler(), dt=0.025, saveat=0.1)

    x_interval = infimum(domains[2].domain)+dx:dx:supremum(domains[2].domain)
    u = asf.(x_interval)
    # Plot and save results

    # Test
    t_f = size(sol, 3)

    x_sol = x_interval
    t_f = size(sol, 3)
    exact = u
    # plot(x_sol, u, seriestype = :scatter,label="Analytic solution")
    # plot!(x_sol, sol.u[end], label="Numeric solution")
    # plot!(x_sol, u.-sol.u[end], label="Differential Error")

    # savefig("plots/MOL_Linear_Convection_Test00.png")


    @test sol[u(t, x)][end, :] ≈ u atol = 0.1
    @test sol_upwind[u(t, x)][end, :] ≈ u atol = 0.1
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
    discretization = MOLFiniteDifference([x => dx], t)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys, discretization)

    # Solve ODE problem
    using OrdinaryDiffEq
    sol = solve(prob, Euler(), dt=0.025, saveat=0.1)


    # Test
    x_interval = infimum(domains[2].domain)+dx:dx:supremum(domains[2].domain)
    u = asf.(x_interval)
    # Plot and save results

    # Test
    t_f = size(sol, 3)

    x_sol = x_interval
    t_f = size(sol, 3)
    exact = u
    # plot(x_sol, u, seriestype = :scatter,label="Analytic solution")
    # plot!(x_sol, sol.u[end], label="Numeric solution")
    # plot!(x_sol, u.-sol.u[end], label="Differential Error")

    # savefig("plots/MOL_Linear_Convection_Test01.png")


    @test sol[u(t, x)][end, :] ≈ u atol = 0.1
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
    discretization = MOLFiniteDifference([x => dx], t, upwind_order=1)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys, discretization)

    # Solve ODE problemdoes this mean that - doesn't seem to affect things
    using OrdinaryDiffEq
    sol = solve(prob, Euler(), dt=0.025, saveat=0.1)

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
    x_interval = infimum(domains[2].domain)+dx:dx:supremum(domains[2].domain)
    u = asf.(x_interval)
    # Plot and save results

    # Test
    t_f = size(sol, 3)

    x_sol = x_interval
    t_f = size(sol, 3)
    exact = u
    # plot(x_sol, u, seriestype = :scatter,label="Analytic solution")
    # plot!(x_sol, sol.u[end], label="Numeric solution")
    # plot!(x_sol, u.-sol.u[end], label="Differential Error")

    # savefig("plots/MOL_1D_Linear_Convection_Test02.png")


    # savefig("plots/MOL_Linear_Convection_Test02.png")
    # plot()
    # anim = @animate for i in eachindex(sol.t)
    #     plot!(x_sol, sol.u[i], label="Numeric solution at $(sol.t[i])")
    # end
    # gif(anim, "plots/MOL_Linear_Convection_Test02.gif", fps = 5)

    @test sol[u(t, x)][end, :] ≈ u atol = 0.1
end
@test_broken begin#@testset "Test 03: Dt(u(t,x)) ~ -Dx(v(t,x))*u(t,x)-v(t,x)*Dx(u(t,x)) with v(t,x)=1" begin
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
        v(t, 0) ~ 1.0,
        v(t, 2) ~ 1.0
    ]

    # Space and time domains


    # Space and time domains
    domains = [t ∈ Interval(0.0, 2.0),
        x ∈ Interval(0.0, 2.0)]


    # PDE system
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x), v(t, x)])

    # Method of lines discretization
    dx = 2 / 80
    order = 1
    discretization = MOLFiniteDifference([x => dx], t)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys, discretization)

    # Solve ODE problem
    using OrdinaryDiffEq
    sol = solve(prob, Euler(), dt=0.025, saveat=0.1)

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
    x_interval = infimum(domains[2].domain)+dx:dx:supremum(domains[2].domain)-dx
    u = asf.(x_interval)
    # Plot and save results

    # Test
    t_f = size(sol, 3)

    x_sol = x_interval
    t_f = size(sol, 3)
    exact = u
    # plot(x_sol, u, seriestype = :scatter,label="Analytic solution")
    # plot!(x_sol, sol.u[end], label="Numeric solution")
    # plot!(x_sol, u.-sol.u[end], label="Differential Error")

    # savefig("plots/MOL_Linear_Convection_Test03.png")


    @test sol[u(t, x)][end, :] ≈ u atol = 0.1
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
        v(t, 0) ~ 0.999,
        v(t, 2) ~ 0.9999 + 0.0001 * t * 2.0]

    # Space and time domains
    domains = [t ∈ Interval(0.0, 2.0),
        x ∈ Interval(0.0, 2.0)]

    # PDE system
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x), v(t, x)])

    # Method of lines discretization
    dx = 2 / 80
    order = 1
    discretization = MOLFiniteDifference([x => dx], t)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys, discretization)

    # Solve ODE problem
    using OrdinaryDiffEq
    sol = solve(prob, Euler(), dt=0.025, saveat=0.1)

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
    x_interval = infimum(domains[2].domain)+dx:dx:supremum(domains[2].domain)
    u = asf.(x_interval)
    # Plot and save results

    # Test
    t_f = size(sol, 3)

    x_sol = x_interval
    t_f = size(sol, 3)
    exact = u
    # plot(x_sol, u, seriestype = :scatter,label="Analytic solution")
    # plot!(x_sol, sol.u[end], label="Numeric solution")
    # plot!(x_sol, u.-sol.u[end], label="Differential Error")

    # savefig("plots/MOL_Linear_Convection_Test04.png")

    # plot()
    # anim = @animate for i in eachindex(sol.t)
    #     plot!(x_sol, sol.u[i], label="Numeric solution at $(sol.t[i])")
    # end
    # gif(anim, "plots/MOL_Linear_Convection_Test04.gif", fps = 5)



    @test sol[u(t, x)][end, :] ≈ u atol = 0.1
end


@test_broken begin #@testset "Test 05: Dt(u(t,x)) ~ -Dx(v(t,x)*u(t,x)) with v(t,x)=0.999 + 0.001 * t * x " begin
    # Parameters, variables, and derivatives
    @parameters t x
    @variables v(..) u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    # 1D PDE and boundary conditions
    eq = [Dt(u(t, x)) ~ -Dx(v(t, x) * u(t, x)),
        v(t, x) ~ 0.999 + 0.001 * t * x]
    asf(x) = (0.5 / (0.2 * sqrt(2.0 * 3.1415))) * exp(-(x - 1.0)^2 / (2.0 * 0.2^2))

    bcs = [u(0, x) ~ asf(x),
        u(t, 0) ~ u(t, 2),
        v(0, x) ~ 0.999,
        v(t, 0) ~ 0.999,
        v(t, 2) ~ 0.9999 + 0.0001 * t * 2.0]

    # Space and time domains
    domains = [t ∈ Interval(0.0, 2.0),
        x ∈ Interval(0.0, 2.0)]

    # PDE system
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x), v(t, x)])

    # Method of lines discretization
    dx = 2 / 80
    order = 1
    discretization = MOLFiniteDifference([x => dx], t)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys, discretization)

    # Solve ODE problem
    using OrdinaryDiffEq
    sol = solve(prob, Euler(), dt=0.025, saveat=0.1)

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
    x_interval = infimum(domains[2].domain)+dx:dx:supremum(domains[2].domain)
    u = asf.(x_interval)
    # Plot and save results

    # Test
    t_f = size(sol, 3)

    x_sol = x_interval
    t_f = size(sol, 3)
    exact = u
    # plot(x_sol, u, seriestype = :scatter,label="Analytic solution")
    # plot!(x_sol, sol.u[end], label="Numeric solution")
    # plot!(x_sol, u.-sol.u[end], label="Differential Error")

    # savefig("plots/MOL_Linear_Convection_Test04.png")

    # plot()
    # anim = @animate for i in eachindex(sol.t)
    #     plot!(x_sol, sol.u[i], label="Numeric solution at $(sol.t[i])")
    # end
    # gif(anim, "plots/MOL_Linear_Convection_Test04.gif", fps = 5)



    @test sol[u(t, x)][end, :] ≈ u atol = 0.1
end
