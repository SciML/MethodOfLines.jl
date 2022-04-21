using ModelingToolkit, MethodOfLines, LinearAlgebra, OrdinaryDiffEq
using DomainSets

@testset "Inviscid Burgers equation, 1D, u(0, x) ~ x" begin
    @parameters x t
    @variables u(..)
    Dx = Differential(x)
    Dt = Differential(t)
    x_min = 0.0
    x_max = 1.0
    t_min = 0.0
    t_max = 1.0

    analytic_u(t, x) = x / (t + 1)

    eq = Dt(u(t, x)) + u(t, x) * Dx(u(t, x)) ~ 0

    bcs = [u(0, x) ~ x,
        u(t, x_min) ~ 0]

    domains = [t ∈ Interval(t_min, t_max),
        x ∈ Interval(x_min, x_max)]

    dx = 0.05

    pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    disc = MOLFiniteDifference([x => dx], t, upwind_order=1)

    prob = discretize(pdesys, disc)

    sol = solve(prob, Tsit5())

    x_disc = 0:dx:1

    for t in sol.t
        u_analytic = analytic_u.([t], x_disc)
        u_disc = sol[u]

        @test all(isapprox.(u_analytic, u_disc), atol=1e-3)
    end
end

@testset "Inviscid Burgers equation, 1D, u(0, x) ~ x, Non-Uniform" begin
    @parameters x t
    @variables u(..)
    Dx = Differential(x)
    Dt = Differential(t)
    x_min = 0.0
    x_max = 1.0
    t_min = 0.0
    t_max = 1.0

    analytic_u(t, x) = x / (t + 1)

    eq = Dt(u(t, x)) + u(t, x) * Dx(u(t, x)) ~ 0

    bcs = [u(0, x) ~ x,
        u(t, x_min) ~ 0]

    domains = [t ∈ Interval(t_min, t_max),
        x ∈ Interval(x_min, x_max)]

    pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    dx = 0:0.05:1
    dx = collect(dx)
    dx[2:end-1] .= dx[2:end-1] .+ rand([0.001, -0.001], length(dx[2:end-1]))

    disc = MOLFiniteDifference([x => dx], t, upwind_order=1)

    prob = discretize(pdesys, disc)

    sol = solve(prob, Tsit5())

    x_disc = dx

    for t in sol.t
        u_analytic = analytic_u.([t], x_disc)
        u_disc = sol[u]

        @test all(isapprox.(u_analytic, u_disc), atol=1e-3)
    end
end

# # Exact solutions from: https://www.sciencedirect.com/science/article/pii/S0898122110003883
# @test_broken begin# @testset "Test 01: Burger's Equation 2D" begin
#        @parameters x y t
#        @variables u(..) v(..)
#        Dt = Differential(t)
#        Dx = Differential(x)
#        Dy = Differential(y)
#        Dxx = Differential(x)^2
#        Dyy = Differential(y)^2

#        R = 80.0

#        x_min = y_min = t_min = 0.0
#        x_max = y_max = 1.0
#        t_max = 0.05

#        u_exact(x,y,t) = 3/4 - 1/(4*(1+exp(R*(-t-4x+4y)/32)))
#        v_exact(x,y,t) = 3/4 + 1/(4*(1+exp(R*(-t-4x+4y)/32)))

#        eq = [Dt(u(x,y,t)) + u(x,y,t)*Dx(u(x,y,t)) + v(x,y,t)*Dy(u(x,y,t)) ~ 1/R *(Dxx(u(x,y,t)) + Dyy(u(x,y,t))),
#              Dt(v(x,y,t)) + u(x,y,t)*Dx(v(x,y,t)) + v(x,y,t)*Dy(v(x,y,t)) ~ 1/R *(Dxx(v(x,y,t)) + Dyy(v(x,y,t)))]

#        domains = [x ∈ Interval(x_min, x_max),
#                   y ∈ Interval(y_min, y_max),
#                   t ∈ Interval(t_min, t_max)]

#        bcs = [u(x,y,0) ~ u_exact(x,y,0),
#               u(0,y,t) ~ u_exact(0,y,t),
#               u(x,0,t) ~ u_exact(x,0,t),
#               u(1,y,t) ~ u_exact(1,y,t),
#               u(x,1,t) ~ u_exact(x,1,t),

#               v(x,y,0) ~ v_exact(x,y,0),
#               v(0,y,t) ~ v_exact(0,y,t),
#               v(x,0,t) ~ v_exact(x,0,t),
#               v(1,y,t) ~ v_exact(1,y,t),
#               v(x,1,t) ~ v_exact(x,1,t)]

#        @named pdesys = PDESystem(eq,bcs,domains,[t,x,y],[u(x,y,t),v(x,y,t)])

#        # Method of lines discretization
#        dx = 0.05
#        dy = 0.05


#        order = 4

#        discretization = MOLFiniteDifference([x=>dx, y=>dy], t, approx_order=order)

#        # Convert the PDE problem into an ODE problem
#        prob = discretize(pdesys,discretization)

#        sol = solve(prob,Tsit5())

#        Nx = floor(Int64, (x_max - x_min) / dx) + 1
#        Ny = floor(Int64, (y_max - y_min) / dy) + 1

#        @variables u[1:Nx,1:Ny](t)
#        @variables v[1:Nx,1:Ny](t)
#        t = sol[t]
#        # anim = @animate for k in 1:length(t)
#        #        solu′ = reshape([sol[u[(i-1)*Ny+j]][k] for i in 1:Nx for j in 1:Ny],(Nx,Ny))
#        #        solv′ = reshape([sol[v[(i-1)*Ny+j]][k] for i in 1:Nx for j in 1:Ny],(Nx,Ny))
#        #        heatmap(solu′)
#        # end
#        # gif(anim, "plots/Burgers2Dsol.gif", fps = 5)


#        solu′ = reshape([sol[u[(i-1)*Ny+j]][end] for i in 1:Nx for j in 1:Ny],(Nx,Ny))
#        solv′ = reshape([sol[v[(i-1)*Ny+j]][end] for i in 1:Nx for j in 1:Ny],(Nx,Ny))

#        r_space_x = x_min:dx:x_max
#        r_space_y = y_min:dy:y_max

#        asfu = reshape([u_exact(t_max,r_space_x[i],r_space_y[j]) for j in 1:Ny for i in 1:Nx],(Nx,Ny))
#        asfv = reshape([v_exact(t_max,r_space_x[i],r_space_y[j]) for j in 1:Ny for i in 1:Nx],(Nx,Ny))

#        asfu[1,1] = asfu[1, end] = asfu[end, 1] = asfu[end, end] = 0.
#        asfv[1,1] = asfv[1, end] = asfv[end, 1] = asfv[end, end] = 0.

#        # anim = @animate for T in t
#        #        asfu = reshape([u_exact(T,r_space_x[i],r_space_y[j]) for j in 1:Ny for i in 1:Nx],(Nx,Ny))
#        #        asfv = reshape([v_exact(T,r_space_x[i],r_space_y[j]) for j in 1:Ny for i in 1:Nx],(Nx,Ny))

#        #        heatmap(asfu)
#        # end
#        # gif(anim, "plots/Burgers2Dexact.gif", fps = 5)


#        mu = max(asfu...)
#        mv = max(asfv...)
#        @test_broken asfu / mu ≈ solu′ / mu  atol=0.2
#        @test_broken asfv / mv ≈ solv′ / mv  atol=0.2
# end
