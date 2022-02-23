# 1D diffusion problem

# Packages and inclusions
using ModelingToolkit,MethodOfLines,LinearAlgebra,Test,OrdinaryDiffEq, DomainSets
using ModelingToolkit: Differential

# Beam Equation
@test_broken begin #@testset "Test 00: Beam Equation" begin
    @parameters x, t
    @variables u(..)
    Dt = Differential(t)
    Dtt = Differential(t)^2
    Dx = Differential(x)
    Dxx = Differential(x)^2
    Dx3 = Differential(x)^3
    Dx4 = Differential(x)^4

    g = -9.81
    EI = 1
    mu = 1
    L = 10.0
    dx = 0.4

    eq = Dtt(u(t,x)) ~ -mu*EI*Dx4(u(t,x)) + mu*g

    bcs = [u(0, x) ~ 0,
           u(t,0) ~ 0,
           Dx(u(t,0)) ~ 0,
           Dxx(u(t, L)) ~ 0,
           Dx3(u(t, L)) ~ 0]

    # Space and time domains
    domains = [t ∈ Interval(0.0,1.0),
               x ∈ Interval(0.0,L)]

    @named pdesys = PDESystem(eq,bcs,domains,[t,x],[u(t,x)])
    discretization = MOLFiniteDifference([x=>dx],t, approx_order=4)
    prob = discretize(pdesys,discretization)
end

# Beam Equation with Velocity
@test_broken begin#@testset "Test 01: Beam Equation with velocity" begin
    @parameters x, t
    @variables u(..), v(..)
    Dt = Differential(t)
    Dtt = Differential(t)^2
    Dx = Differential(x)
    Dxx = Differential(x)^2
    Dx3 = Differential(x)^3
    Dx4 = Differential(x)^4

    g = -9.81
    EI = 1
    mu = 1
    L = 10.0
    dx = 0.4

    eqs = [v(t, x) ~ Dt(u(t,x)),
           Dt(v(t,x)) ~ -mu*EI*Dx4(u(t,x)) + mu*g]

    bcs = [u(0, x) ~ 0,
           v(0, x) ~ 0,
           u(t,0) ~ 0,
           v(t,0) ~ 0,
           Dxx(u(t, L)) ~ 0,
           Dx3(u(t, L)) ~ 0]

    # Space and time domains
    domains = [t ∈ Interval(0.0,1.0),
               x ∈ Interval(0.0,L)]

    @named pdesys = PDESystem(eqs,bcs,domains,[t,x],[u(t,x),v(t,x)])
    discretization = MOLFiniteDifference([x=>dx],t, approx_order=4)
    prob = discretize(pdesys,discretization)
end

@test_broken begin#@testset "KdV Single Soliton equation" begin
    @parameters x, t
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dx2 = Differential(x)^2
    Dx3 = Differential(x)^3

    α = 6
    β = 1
    eq = Dt(u(x,t)) ~ -α*u(x,t)*Dx(u(x,t)) - β*Dx3(u(x,t)) 

    u_analytic(x,t;z = (x - t)/2) =  1/2*sech(z)^2
    du(x,t;z = (x-t)/2) = 1/2*tanh(z)*sech(z)^2
    ddu(x,t; z= (x-t)/2) = 1/4*(2*tanh(z)^2 + sech(z)^2)*sech(z)^2
    bcs = [u(x,0) ~ u_analytic(x,0),
           u(-10,t) ~ u_analytic(-10,t),
           u(10,t) ~ u_analytic(10,t),
           Dx(u(-10,t)) ~ du(-10,t),
           Dx(u(10,t)) ~ du(10,t),
           Dx2(u(-10,t)) ~ ddu(-10,t),
           Dx2(u(10,t)) ~ ddu(10,t)]

    # Space and time domains
    domains = [x ∈ Interval(-10.0,10.0),
               t ∈ Interval(0.0,1.0)]
    # Discretization
    dx = 0.4; dt = 0.2

    fail

    discretization = MOLFiniteDifference([x=>dx],t;upwind_order=1,grid_align=center_align)
    @named pdesys = PDESystem(eq,bcs,domains,[x,t],[u(x,t)])
    prob = discretize(pdesys,discretization)

    sol = solve(prob,Rosenbrock23(),saveat=0.1,dt=dt)

    @test sol.retcode == :Success

    xs = domains[1].domain.lower+dx+dx+dx:dx:domains[1].domain.upper
    ts = sol.t

    u_predict = sol.u
    u_real = [[u_analytic(x, t) for x in xs] for t in ts]
    

       # anim = @animate for (i,T) in enumerate(ts) 
       #        plot(xs, u_real[i], seriestype = :scatter,label="Analytic solution")
       #        plot!(xs, sol.u[i], label="Numeric solution")
       #        plot!(xs, log.(abs.(u_real[i]-sol.u[i])), label="Log Error at t = $(ts[i])")
       # end
       # gif(anim, "plots/MOL_Higher_order_1D_KdV_single_soliton.gif", fps = 5)


    @test all(isapprox.(u_predict, u_real, rtol = 0.03))
end