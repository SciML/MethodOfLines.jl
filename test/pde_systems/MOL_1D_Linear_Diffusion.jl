# 1D diffusion problem

# Packages and inclusions
using ModelingToolkit, MethodOfLines, LinearAlgebra, Test, OrdinaryDiffEq, DomainSets
using ModelingToolkit: Differential

# Tests
@testset "Test 00: Dt(u(t,x)) ~ Dxx(u(t,x))" begin
    # Method of Manufactured Solutions
    u_exact = (x, t) -> exp.(-t) * cos.(x)

    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    # 1D PDE and boundary conditions
    eq = Dt(u(t, x)) ~ Dxx(u(t, x))
    bcs = [u(0, x) ~ cos(x),
        u(t, 0) ~ exp(-t),
        u(t, Float64(π)) ~ -exp(-t)]

    # Space and time domains
    domains = [t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, Float64(π))]

    # PDE system
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    # Method of lines discretization
    dx = range(0.0, Float64(π), length=30)
    dx_ = dx[2] - dx[1]

    order = 2
    discretization = MOLFiniteDifference([x => dx_], t)
    discretization_edge = MOLFiniteDifference([x => dx_], t; grid_align=edge_align)
    # Explicitly specify order of centered difference
    discretization_centered = MOLFiniteDifference([x => dx_], t; approx_order=order)
    # Higher order centered difference
    discretization_approx_order4 = MOLFiniteDifference([x => dx_], t; approx_order=4)

    for disc in [discretization, discretization_edge, discretization_centered, discretization_approx_order4]
        # Convert the PDE problem into an ODE problem
        prob = discretize(pdesys, disc)

        # Solve ODE problem      # Solve ODE problem
        sol = solve(prob, Tsit5(), saveat=0.1)

        x_disc = sol[x][2:end-1]
        t_disc = sol[t]
        u_approx = sol[u(t, x)][:, 2:end-1]

        # Test against exact solution
        for i in 1:length(sol)
            exact = u_exact(x_disc, t_disc[i])
            @test all(isapprox.(u_approx[i, :], exact, atol=0.01))
        end
    end
end


@testset "Test 01: Dt(u(t,x)) ~ D*Dxx(u(t,x))" begin
    # Parameters, variables, and derivatives
    @parameters t x D
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    # 1D PDE and boundary conditions
    eq = Dt(u(t, x)) ~ D * Dxx(u(t, x))
    bcs = [u(0, x) ~ -x * (x - 1) * sin(x),
        u(t, 0) ~ 0.0,
        u(t, 1) ~ 0.0]

    # Space and time domains
    domains = [t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, 1.0)]

    # PDE system
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)], [D => 10.0])

    # Method of lines discretization
    dx = 1 / (5pi)
    order = 2
    discretization = MOLFiniteDifference([x => dx], t)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys, discretization)

    # Solve ODE problem
    sol = solve(prob, Tsit5(), saveat=0.1)

    solu = sol[u(t, x)]
    # Test
    n = size(solu, 2)

    @test solu[end, :] ≈ zeros(n) atol = 0.001
end

@testset "Test 02: Dt(u(t,x)) ~ Dx(D(t,x))*Dx(u(t,x))+D(t,x)*Dxx(u(t,x))" begin
    #@test_broken begin
    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..) D(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    D = (t, x) -> 0.999 + 0.001 * t * x
    DxD = expand_derivatives(Dx(D(t, x)))

    # 1D PDE and boundary conditions

    eq = [Dt(u(t, x)) ~ DxD * Dx(u(t, x)) + D(t, x) * Dxx(u(t, x)),]

    bcs = [u(0, x) ~ -x * (x - 1) * sin(x),
        u(t, 0) ~ 0.0,
        u(t, 1) ~ 0.0]

    # Space and time domains
    domains = [t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, 1.0)]

    # PDE system
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    # Method of lines discretization
    N = 11
    order = 2
    discretization = MOLFiniteDifference([x => N], t)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys, discretization)

    # Solve ODE problem
    sol = solve(prob, Tsit5(), saveat=0.1)

    solu = sol[u(t, x)]
    # Test
    n = size(solu, 2)

    @test solu[end, :] ≈ zeros(n) atol = 0.001
end

@testset "Test 03: Dt(u(t,x)) ~ Dxx(u(t,x)), homogeneous Neumann BCs, order 8" begin
    # Method of Manufactured Solutions
    u_exact = (x, t) -> exp.(-t) * cos.(x)

    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    # 1D PDE and boundary conditions
    eq = Dt(u(t, x)) ~ Dxx(u(t, x))
    bcs = [u(0, x) ~ cos(x),
        Dx(u(t, 0)) ~ 0,
        Dx(u(t, Float64(pi))) ~ 0]

    # Space and time domains
    domains = [t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, Float64(pi))]

    # PDE system
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    # Method of lines discretization
    dx = range(0.0, Float64(π), length=300)
    dx_ = dx[2] - dx[1]#range(0.0,Float64(π),length=300)
    order = 8
    discretization = MOLFiniteDifference([x => dx_], t)
    discretization_edge = MOLFiniteDifference([x => dx_], t; grid_align=edge_align)
    # Convert the PDE problem into an ODE problem
    for disc in [discretization, discretization_edge]
        prob = discretize(pdesys, disc)

        # Solve ODE problem
        sol = solve(prob, Tsit5(), saveat=0.1)

        x_sol = sol[x]
        t_sol = sol[t]

        # Plots
        # if shouldplot
        #     anim = @animate for (i,T) in enumerate(t_sol)
        #         exact = u_exact(x_sol, T)
        #         plot(x_sol, exact, seriestype = :scatter,label="Analytic solution")
        #         plot!(x_sol, sol.u[i], label="Numeric solution")
        #         plot!(x_sol, log10.(abs.(exact-sol.u[i])), label="log10 Error at t = $(t_sol[i])")
        #     end
        #     gif(anim, "plots/MOL_Linear_Diffusion_1D_Test03_$disc.gif", fps = 5)
        # end

        u_approx = sol[u(t, x)]
        # Test against exact solution
        for i in 1:length(sol)
            exact = u_exact(x_sol, t_sol[i])
            @test all(isapprox.(u_approx[i, :], exact, atol=0.01))
            @test sum(u_approx[i, :]) ≈ 0 atol = 1e-10
        end
    end
end

@testset "Test 03a: Dt(u(t,x)) ~ Dxx(u(t,x)), Neumann BCs order 4" begin
    # Method of Manufactured Solutions
    u_exact = (x, t) -> exp.(-t) * sin.(x)

    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    # 1D PDE and boundary conditions
    eq = Dt(u(t, x)) ~ Dxx(u(t, x))
    bcs = [u(0, x) ~ sin(x),
        Dx(u(t, 0)) ~ exp(-t),
        Dx(u(t, Float64(pi))) ~ -exp(-t)]

    # Space and time domains
    domains = [t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, Float64(pi))]

    # PDE system
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    # Method of lines discretization
    dx = range(0.0, Float64(π), length=30)
    dx_ = dx[2] - dx[1]
    order = 2
    discretization = MOLFiniteDifference([x => dx_], t, approx_order=2)
    discretization_edge = MOLFiniteDifference([x => dx_], t; grid_align=edge_align, approx_order=2)

    # Convert the PDE problem into an ODE problem
    for (j, disc) ∈ enumerate([discretization, discretization_edge])
        prob = discretize(pdesys, disc)

        # Solve ODE problem
        sol = solve(prob, Tsit5(), saveat=0.1)

        solx = sol[x]
        solt = sol[t]

        # # Plots
        # if shouldplot
        #     anim = @animate for (i,T) in enumerate(t)
        #         exact = u_exact(x, T)
        #         plot(x, exact, seriestype = :scatter,label="Analytic solution")
        #         plot!(x, sol.u[i], label="Numeric solution")
        #         plot!(x, log10.(abs.(exact-sol.u[i])), label="log10 Error at t = $(t[i])")
        #     end
        #     gif(anim, "plots/MOL_Linear_Diffusion_1D_Test03a_$disc.gif", fps = 5)
        # end
        # Test against exact solution
        # exact integral based on Neumann BCs
        u_approx = sol[u(t, x)]
        integral_u_exact = t -> sum(u_approx[1, :] * dx_) + 2 * (exp(-t) - 1)
        for i in 1:length(solt)
            exact = u_exact(solx, solt[i])
            @test all(isapprox.(u_approx[i, :], exact, atol=0.01))
            # test mass conservation
            integral_u_approx = sum(u_approx[i, :] * dx_)
            @test integral_u_exact(solt[i]) ≈ integral_u_approx atol = 0.01
        end
    end
end

@testset "Test 04: Dt(u(t,x)) ~ Dxx(u(t,x)), Neumann + Dirichlet BCs" begin
    # Method of Manufactured Solutions
    u_exact = (x, t) -> exp.(-t) * sin.(x)

    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    # 1D PDE and boundary conditions
    eq = Dt(u(t, x)) ~ Dxx(u(t, x))
    bcs = [u(0, x) ~ sin(x),
        u(t, 0) ~ 0.0,
        Dx(u(t, Float64(pi))) ~ -exp(-t)]

    # Space and time domains
    domains = [t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, Float64(pi))]

    # PDE system
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    # Method of lines discretization
    dx = range(0.0, Float64(π), length=30)
    dx_ = dx[2] - dx[1]
    order = 2
    discretization = MOLFiniteDifference([x => dx_], t)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys, discretization)

    # Solve ODE problem
    sol = solve(prob, Tsit5(), saveat=0.1)
    u_approx = sol[u(t, x)]
    discx = sol[x]
    disct = sol[t]

    # Test against exact solution
    for i in 1:length(sol[t])
        exact = u_exact(discx, disct[i])
        @test all(isapprox.(u_approx[i, :], exact, atol=0.01))
    end
end

@testset "Test 05: Dt(u(t,x)) ~ Dxx(u(t,x)), Robin BCs, Order 4" begin
    # Method of Manufactured Solutions
    u_exact = (x, t) -> exp.(-t) * sin.(x)

    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    # 1D PDE and boundary conditions
    eq = Dt(u(t, x)) ~ Dxx(u(t, x))
    bcs = [u(0, x) ~ sin(x),
        u(t, -1.0) + 3Dx(u(t, -1.0)) ~ exp(-t) * (sin(-1.0) + 3cos(-1.0)),
        4u(t, 1.0) + Dx(u(t, 1.0)) ~ exp(-t) * (4sin(1.0) + cos(1.0))]

    # Space and time domains
    domains = [t ∈ Interval(0.0, 1.0),
        x ∈ Interval(-1.0, 1.0)]

    # PDE system
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    # Method of lines discretizationinclusions

    dx = 0.01
    order = 4
    discretization = MOLFiniteDifference([x => dx], t; approx_order=order)
    discretization_edge = MOLFiniteDifference([x => dx], t; approx_order=order, grid_align=edge_align)

    for disc ∈ [discretization, discretization_edge]
        # Convert the PDE problem into an ODE problem
        prob = discretize(pdesys, disc)

        # Solve ODE problem
        sol = solve(prob, Rodas4(), saveat=0.1)
        x_disc = sol[x]
        t_disc = sol[t]
        u_approx = sol[u(t, x)]

        # Test against exact solution
        for i in 1:length(sol[t])
            exact = u_exact(x_disc, t_disc[i])
            @test all(isapprox.(u_approx[i, :], exact, atol=0.1))
        end
    end
end


@testset "Test 06: Dt(u(t,x)) ~ Dxx(u(t,x)), time-dependent Robin BCs, Order 6" begin
    # Method of Manufactured Solutions
    u_exact = (x, t) -> exp.(-t) * sin.(x)

    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    # 1D PDE and boundary conditions
    eq = Dt(u(t, x)) ~ Dxx(u(t, x))
    bcs = [u(0, x) ~ sin(x),
        t^2 * u(t, -1.0) + 3Dx(u(t, -1.0)) ~ exp(-t) * (t^2 * sin(-1.0) + 3cos(-1.0)),
        4u(t, 1.0) + t * Dx(u(t, 1.0)) ~ exp(-t) * (4sin(1.0) + t * cos(1.0))]

    # Space and time domains
    domains = [t ∈ Interval(0.0, 1.0),
        x ∈ Interval(-1.0, 1.0)]

    # PDE system
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    # Method of lines discretization
    dx = 0.01
    order = 6
    discretization = MOLFiniteDifference([x => dx], t, approx_order=order)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys, discretization)

    # Solve ODE problem
    sol = solve(prob, Rodas4(), reltol=1e-6, saveat=0.1)

    u_approx = sol[u(t, x)]
    discx = sol[x]
    t = sol.t

    # Test against exact solution
    for i in 1:length(t)

        exact = u_exact(discx, t[i])
        @test all(isapprox.(u_approx[i, :], exact, atol=0.06))
    end
end

@testset "Test 07: Dt(u(t,r)) ~ 1/r^2 * Dr(r^2 * Dr(u(t,r))) (Spherical Laplacian), order 4" begin
    # Method of Manufactured Solutions
    # general solution of the spherical Laplacian equation
    # satisfies Dr(u(t,0)) = 0
    u_exact = (r, t) -> exp.(-t) * sin.(r) ./ r

    # Parameters, variables, and derivatives
    @parameters t r
    @variables u(..)
    Dt = Differential(t)
    Dr = Differential(r)

    # 1D PDE and boundary conditions

    eq = Dt(u(t, r)) ~ 1 / r^2 * Dr(r^2 * Dr(u(t, r)))
    bcs = [u(0, r) ~ sin(r) / r,
        Dr(u(t, 0)) ~ 0,
        u(t, 1) ~ exp(-t) * sin(1)]
    #    Dr(u(t,1)) ~ -exp(-t) * sin(1)]

    # Space and time domains
    domains = [t ∈ Interval(0.0, 1.0),
        r ∈ Interval(0.0, 1.0)]

    # PDE system
    @named pdesys = PDESystem(eq, bcs, domains, [t, r], [u(t, r)])

    # Method of lines discretization
    dr = 0.1
    order = 4
    discretization = MOLFiniteDifference([r => dr], t, approx_order=4)
    prob = discretize(pdesys, discretization)

    # Solve ODE problem
    sol = solve(prob, Rodas4(), saveat=0.1)

    u_approx = sol[u(t, r)]
    r = sol[r][2:end-1]
    t = sol[t]
    # if shouldplot
    #     anim = @animate for (i,T) in enumerate(t)
    #         exact = u_exact(r, T)
    #         plot(r, exact, seriestype = :scatter,label="Analytic solution")
    #         plot!(r, sol.u[i], label="Numeric solution")
    #         plot!(r, log10.(abs.(exact-sol.u[i])), label="log10 Error at t = $(t[i])")
    #     end
    #     gif(anim, "plots/MOL_Linear_Diffusion_1D_Test07.gif", fps = 5)
    # end


    # Test against exact solution
    for i in 1:length(t)
        exact = u_exact(r, t[i])
        @test all(isapprox.(u_approx[i, 2:end-1], exact, atol=0.06))
    end
end

@testset "Test 08: Dt(u(t,r)) ~ 4/r^2 * Dr(r^2 * Dr(u(t,r))) (Spherical Laplacian)" begin
    # Method of Manufactured Solutions
    # general solution of the spherical Laplacian equation
    # satisfies Dr(u(t,0)) = 0
    u_exact = (r, t) -> exp.(-4t) * sin.(r) ./ r

    # Parameters, variables, and derivatives
    @parameters t r
    @variables u(..)
    Dt = Differential(t)
    Dr = Differential(r)

    # 1D PDE and boundary conditions

    eq = Dt(u(t, r)) ~ 4 / r^2 * Dr(r^2 * Dr(u(t, r)))
    bcs = [u(0, r) ~ sin(r) / r,
        Dr(u(t, 0)) ~ 0,
        u(t, 1) ~ exp(-4t) * sin(1)]

    # Space and time domains
    domains = [t ∈ Interval(0.0, 1.0),
        r ∈ Interval(0.0, 1.0)]

    # PDE system
    @named pdesys = PDESystem(eq, bcs, domains, [t, r], [u(t, r)])

    # Method of lines discretization
    dr = 0.1
    order = 2
    discretization = MOLFiniteDifference([r => dr], t)
    prob = discretize(pdesys, discretization)

    # Solve ODE problem
    sol = solve(prob, Tsit5(), saveat=0.1)

    u_approx = sol[u(t, r)]
    r = sol[r][2:end-1]
    t = sol[t]

    # if shouldplot
    #     anim = @animate for (i,T) in enumerate(t)
    #         exact = u_exact(r, T)
    #         plot(r, exact, seriestype = :scatter,label="Analytic solution")
    #         plot!(r, sol.u[i], label="Numeric solution")
    #         plot!(r, log10.(abs.(exact-sol.u[i])), label="log10 Error at t = $(t[i])")
    #     end
    #     gif(anim, "plots/MOL_Linear_Diffusion_1D_Test08.gif", fps = 5)
    # end

    # Test against exact solution
    for i in 1:length(t)
        exact = u_exact(r, t[i])
        @test all(isapprox.(u_approx[i, 2:end-1], exact, atol=0.06))
    end
end

@testset "Test 10: linear diffusion, two variables, mixed BCs, order 6" begin
    # Method of Manufactured Solutions
    u_exact = (x, t) -> exp.(-t) * cos.(x)
    v_exact = (x, t) -> exp.(-t) * sin.(x)

    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..) v(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Dx^2

    # 1D PDE and boundary conditions
    eqs = [Dt(u(t, x)) ~ Dxx(u(t, x)),
        Dt(v(t, x)) ~ Dxx(v(t, x))]
    bcs = [u(0, x) ~ cos(x),
        v(0, x) ~ sin(x),
        u(t, 0) ~ exp(-t),
        Dx(u(t, 1)) ~ -exp(-t) * sin(1),
        Dx(v(t, 0)) ~ exp(-t),
        v(t, 1) ~ exp(-t) * sin(1)]

    # Space and time domains
    domains = [t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, 1.0)]

    # PDE system
    @named pdesys = PDESystem(eqs, bcs, domains, [t, x], [u(t, x), v(t, x)])

    # Method of lines discretization
    l = 100
    dx = range(0.0, 1.0, length=l)
    dx_ = dx[2] - dx[1]
    order = 6
    discretization = MOLFiniteDifference([x => dx_], t)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys, discretization)

    # Solve ODE problem
    sol = solve(prob, Tsit5(), saveat=0.1)

    x_sol = sol[x][2:end-1]
    t_sol = sol[t]
    solu = sol[u(t, x)]
    solv = sol[v(t, x)]

    # Test against exact solution
    for i in 1:length(sol)
        @test all(isapprox.(u_exact(x_sol, t_sol[i]), solu[i, 2:end-1], atol=0.01))
        @test all(isapprox.(v_exact(x_sol, t_sol[i]), solv[i, 2:end-1], atol=0.01))
    end
end

@testset "Test 11: linear diffusion, two variables, mixed BCs, with parameters" begin
    @parameters t x
    @parameters Dn, Dp
    @variables u(..) v(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    eqs = [Dt(u(t, x)) ~ Dn * Dxx(u(t, x)) + u(t, x) * v(t, x),
        Dt(v(t, x)) ~ Dp * Dxx(v(t, x)) - u(t, x) * v(t, x)]
    bcs = [u(0, x) ~ sin(pi * x / 2),
        v(0, x) ~ sin(pi * x / 2),
        u(t, 0) ~ 0.0, Dx(u(t, 1)) ~ 0.0,
        v(t, 0) ~ 0.0, Dx(v(t, 1)) ~ 0.0]

    domains = [t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, 1.0)]

    @named pdesys = PDESystem(eqs, bcs, domains, [t, x], [u(t, x), v(t, x)], [Dn => 0.5, Dp => 2])
    discretization = MOLFiniteDifference([x => 0.1], t)
    prob = discretize(pdesys, discretization)
    @test prob.p == [0.5, 2]
    # Make sure it can be solved
    sol = solve(prob, Tsit5())
end

@testset "Test 12: linear diffusion, two variables, mixed BCs, different independent variables order 4" begin
    # Method of Manufactured Solutions
    u_exact = (x, t) -> exp.(-t) * cos.(x)
    v_exact = (y, t) -> exp.(-t) * sin.(y)

    # Parameters, variables, and derivatives
    @parameters t x y
    @variables u(..) v(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Dx^2
    Dy = Differential(y)
    Dyy = Dy^2

    # 1D PDE and boundary conditions
    eqs = [Dt(u(t, x)) ~ Dxx(u(t, x)),
        Dt(v(t, y)) ~ Dyy(v(t, y))]
    bcs = [u(0, x) ~ cos(x),
        v(0, y) ~ sin(y),
        u(t, 0) ~ exp(-t),
        Dx(u(t, 1)) ~ -exp(-t) * sin(1),
        Dy(v(t, 0)) ~ exp(-t),
        v(t, 2) ~ exp(-t) * sin(2)]

    # Space and time domains
    domains = [t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, 1.0),
        y ∈ Interval(0.0, 2.0)]

    # PDE system
    @named pdesys = PDESystem(eqs, bcs, domains, [t, x, y], [u(t, x), v(t, y)])

    # Method of lines discretization
    l = 100
    dx = range(0.0, 1.0, length=l)
    dx_ = dx[2] - dx[1]
    dy = range(0.0, 2.0, length=l)
    dy_ = dy[2] - dy[1]
    order = 4
    discretization = MOLFiniteDifference([x => dx_, y => dy_], t)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys, discretization)

    # Solve ODE problem
    sol = solve(prob, Tsit5(), saveat=0.1)

    x_sol = sol[x]
    y_sol = sol[y]
    t_sol = sol[t]
    solu = sol[u(t, x)]
    solv = sol[v(t, y)]

    # Test against exact solution
    for i in 1:length(sol)
        @test all(isapprox.(u_exact(x_sol, t_sol[i]), solu[i, :], atol=0.01))
        @test all(isapprox.(v_exact(y_sol, t_sol[i]), solv[i, :], atol=0.01))
    end
end

@testset "Test 12: linear diffusion, two variables, mixed BCs, different independent variables in a vector Order 2" begin
    # Method of Manufactured Solutions
    u_exact = (x, t) -> exp.(-t) * cos.(x)
    v_exact = (y, t) -> exp.(-t) * sin.(y)

    # Parameters, variables, and derivatives
    @parameters t x y
    @variables u[1:2](..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Dx^2
    Dy = Differential(y)
    Dyy = Dy^2

    # 1D PDE and boundary conditions
    eqs = [Dt(u[1](t, x)) ~ Dxx(u[1](t, x)),
        Dt(u[2](t, y)) ~ Dyy(u[2](t, y))]
    bcs = [u[1](0, x) ~ cos(x),
        u[2](0, y) ~ sin(y),
        u[1](t, 0) ~ exp(-t),
        Dx(u[1](t, 1)) ~ -exp(-t) * sin(1),
        Dy(u[2](t, 0)) ~ exp(-t),
        u[2](t, 2) ~ exp(-t) * sin(2)]

    # Space and time domains
    domains = [t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, 1.0),
        y ∈ Interval(0.0, 2.0)]

    # PDE system
    @named pdesys = PDESystem(eqs, bcs, domains, [t, x, y], [u[1](t, x), u[2](t, y)])

    # Method of lines discretization
    l = 100
    dx = range(0.0, 1.0, length=l)
    dx_ = dx[2] - dx[1]
    dy = range(0.0, 2.0, length=l)
    dy_ = dy[2] - dy[1]
    order = 2
    discretization = MOLFiniteDifference([x => dx_, y => dy_], t)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys, discretization)

    # Solve ODE problem
    sol = solve(prob, Tsit5(), saveat=0.1)

    solu1 = sol[u[1](t, x)]
    solu2 = sol[u[2](t, y)]

    x_sol = sol[x]
    y_sol = sol[y]
    t_sol = sol[t]

    # Test against exact solution
    for i in 1:length(t_sol)
        @test all(isapprox.(u_exact(x_sol, t_sol[i]), solu1[i, :], atol=0.01))
        @test all(isapprox.(v_exact(y_sol, t_sol[i]), solu2[i, :], atol=0.01))
    end
end

@testset "Test 13: one linear diffusion with mixed BCs, one ODE" begin
    # Method of Manufactured Solutions
    u_exact = (x, t) -> exp.(-t) * sin.(x)
    v_exact = (t) -> exp.(-t)

    @parameters t x
    @variables u(..) v(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Dx^2

    # 1D PDE and boundary conditions
    eqs = [Dt(u(t, x)) ~ Dxx(u(t, x)),
        Dt(v(t)) ~ -v(t)]
    bcs = [u(0, x) ~ sin(x),
        v(0) ~ 1,
        u(t, 0) ~ 0,
        Dx(u(t, 1)) ~ exp(-t) * cos(1)]

    # Space and time domains
    domains = [t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, 1.0)]

    # PDE system
    @named pdesys = PDESystem(eqs, bcs, domains, [t, x], [u(t, x), v(t)])

    # Method of lines discretization
    l = 100
    dx = range(0.0, 1.0, length=l)
    dx_ = dx[2] - dx[1]
    order = 2
    discretization = MOLFiniteDifference([x => dx_], t)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys, discretization)

    # Solve ODE problem
    sol = solve(prob, Tsit5(), saveat=0.1)

    x_sol = sol[x]
    t_sol = sol[t]
    solu = sol[u(t, x)]
    solv = sol[v(t)]

    # Test against exact solution
    for i in 1:length(sol)
        @test all(isapprox.(u_exact(x_sol, t_sol[i]), solu[i, :], atol=0.01))
        @test v_exact(t_sol[i]) ≈ solv[i] atol = 0.01
    end
end

@testset "Test 14: linear diffusion, two variables connected at an interface, dirichlet bcs" begin
    @parameters t x1 x2
    @variables c1(..)
    @variables c2(..)
    Dt = Differential(t)

    Dx1 = Differential(x1)
    Dxx1 = Dx1^2

    Dx2 = Differential(x2)
    Dxx2 = Dx2^2

    eqs = [Dt(c1(t, x1)) ~ Dxx1(c1(t, x1)),
            Dt(c2(t, x2)) ~ Dxx2(c2(t, x2))]

    bcs = [c1(0, x1) ~ -x1 * (x1 - 1) * sin(x1),
           c2(0, x2) ~ x2 * (x2 - 1) * sin(x2),
           c1(t, 0) ~ 0,
           c1(t, 0.5) ~ c2(t, 0.5),
           c2(t, 1) ~ 0]

    domains = [t ∈ Interval(0.0, 1.0),
               x1 ∈ Interval(0.0, 0.5),
               x2 ∈ Interval(0.5, 1.0)]

    @named pdesys = PDESystem(eqs, bcs, domains, [t, x1, x2], [c1(t, x1), c2(t, x2)])

    l = 10

    discretization = MOLFiniteDifference([x1 => l, x2 => l], t)

    prob = discretize(pdesys, discretization)

    sol = solve(prob, Tsit5(), saveat=0.1)

    x1_sol = sol[x1]
    x2_sol = sol[x2]
    t_sol = sol[t]
    solc1 = sol[c1(t, x1)]
    solc2 = sol[c2(t, x2)]

    solc = vcat(solc1[end, :], solc2[end, 2:end])
    @test solc ≈ zeros(length(solc)) atol = 0.001
end
