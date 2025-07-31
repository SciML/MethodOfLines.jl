# 2D diffusion problem

# Packages and inclusions
using ModelingToolkit, MethodOfLines, LinearAlgebra, Test, OrdinaryDiffEq, DomainSets
using ModelingToolkit: Differential

# Tests
@testset "Test 00: Dt(u(t,x,y)) ~ Dxx(u(t,x,y)) + Dyy(u(t,x,y))" begin

    # Variables, parameters, and derivatives
    @parameters t x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    Dt = Differential(t)
    t_min = 0.0
    t_max = 2.0
    x_min = 0.0
    x_max = 2.0
    y_min = 0.0
    y_max = 2.0
    dx = 0.1
    dy = 0.2
    order = 4

    # Analytic solution
    analytic_sol_func(t, x, y) = exp(x + y) * cos(x + y + 4t)

    # Equation
    eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y))

    # Initial and boundary conditions
    bcs = [u(t_min, x, y) ~ analytic_sol_func(t_min, x, y),
        u(t, x_min, y) ~ analytic_sol_func(t, x_min, y),
        u(t, x_max, y) ~ analytic_sol_func(t, x_max, y),
        u(t, x, y_min) ~ analytic_sol_func(t, x, y_min),
        u(t, x, y_max) ~ analytic_sol_func(t, x, y_max)]

    # Space and time domains
    domains = [t ∈ Interval(t_min, t_max),
        x ∈ Interval(x_min, x_max),
        y ∈ Interval(y_min, y_max)]

    # Space and time domains
    @named pdesys = PDESystem([eq], bcs, domains, [t, x, y], [u(t, x, y)])

    # Test against exact solution
    Nx = floor(Int64, (x_max - x_min) / dx) + 1
    Ny = floor(Int64, (y_max - y_min) / dy) + 1

    # Method of lines discretization
    discretization = MOLFiniteDifference([x => dx, y => dy], t; approx_order = order)
    prob = ModelingToolkit.discretize(pdesys, discretization)
    # Solution of the ODE system
    sol = solve(prob, Tsit5())
    r_space_x = sol[x]
    r_space_y = sol[y]
    asf = [analytic_sol_func(t_max, X, Y) for X in r_space_x, Y in r_space_y]
    asf[1, 1] = asf[1, end] = asf[end, 1] = asf[end, end] = 0.0

    # Test against exact solution
    sol′ = sol[u(t, x, y)]
    @test asf≈sol′[end, :, :] atol=0.4

    #Plot
    #using Plots
    #heatmap(sol′)
    #savefig("MOL_Linear_Diffusion_2D_Test00.png")
end

@testset "Test 01: Dt(u(t,x,y)) ~ Dx( a(x,y,u) * Dx(u(t,x,y))) + Dy( a(x,y,u) * Dy(u(t,x,y)))" begin

    # Variables, parameters, and derivatives
    @parameters t x y
    @variables u(..)
    Dx = Differential(x)
    Dy = Differential(y)
    Dt = Differential(t)
    t_min = 0.0
    t_max = 2.0
    x_min = 0.0
    x_max = 2.0
    y_min = 0.0
    y_max = 2.0

    # Analytic solution
    analytic_sol_func(t, x, y) = exp(x + y) * cos(x + y + 4t)

    # Equation
    eq = Dt(u(t, x, y)) ~
         Dx((u(t, x, y)^2 / exp(x + y)^2 + sin(x + y + 4t)^2)^0.5 *
            Dx(u(t, x, y))) +
         Dy((u(t, x, y)^2 / exp(x + y)^2 + sin(x + y + 4t)^2)^0.5 *
            Dy(u(t, x, y)))

    # Initial and boundary conditions
    bcs = [u(t_min, x, y) ~ analytic_sol_func(t_min, x, y),
        u(t, x_min, y) ~ analytic_sol_func(t, x_min, y),
        u(t, x_max, y) ~ analytic_sol_func(t, x_max, y),
        u(t, x, y_min) ~ analytic_sol_func(t, x, y_min),
        u(t, x, y_max) ~ analytic_sol_func(t, x, y_max)]

    # Space and time domains
    domains = [t ∈ Interval(t_min, t_max),
        x ∈ Interval(x_min, x_max),
        y ∈ Interval(y_min, y_max)]

    # Space and time domains
    @named pdesys = PDESystem([eq], bcs, domains, [t, x, y], [u(t, x, y)])

    # Method of lines discretization
    dx = 0.1
    dy = 0.2
    discretization = MOLFiniteDifference([x => dx, y => dy], t)
    prob = ModelingToolkit.discretize(pdesys, discretization)

    # Solution of the ODE system
    sol = solve(prob, Rosenbrock23())

    # Test against exact solution
    sol = solve(prob, Tsit5())
    r_space_x = sol[x]
    r_space_y = sol[y]
    asf = [analytic_sol_func(t_max, X, Y) for X in r_space_x, Y in r_space_y]
    asf[1, 1] = asf[1, end] = asf[end, 1] = asf[end, end] = 0.0

    m = max(asf...)
    sol′ = sol[u(t, x, y)]

    @test asf / m≈sol′[end, :, :] / m atol=0.4 # TODO: use lower atol when g(x) is improved in MOL_discretize.jl

    #Plot
    #using Plots
    #heatmap(sol′)
    #savefig("MOL_NonLinear_Diffusion_2D_Test01.png")

end
