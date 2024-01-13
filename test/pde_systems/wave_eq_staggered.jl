using ModelingToolkit, MethodOfLines, DomainSets, Test, Symbolics, SymbolicUtils, LinearAlgebra
using OrdinaryDiffEq

@testset "1D wave equation, staggered grid, Mixed BC" begin
    @parameters t x
    @variables ρ(..) ϕ(..)
    Dt = Differential(t);
    Dx = Differential(x);

    a = 5.0;
    L = 8.0;
    dx = 0.125;
    dt = (dx/a)^2;
    tmax = 10.0;

    initialFunction(x) = exp(-(x)^2);
    eq = [Dt(ρ(t,x)) + Dx(ϕ(t,x)) ~ 0,
          Dt(ϕ(t,x)) + a^2 * Dx(ρ(t,x)) ~ 0]
    bcs = [ρ(0,x) ~ initialFunction(x),
           ϕ(0.0,x) ~ 0.0,
           Dx(ρ(t,L)) ~ 0.0,
           ϕ(t,-L) ~ 0.0];

    domains = [t in Interval(0.0, tmax),
               x in Interval(-L, L)];

    @named pdesys = PDESystem(eq, bcs, domains, [t,x], [ρ(t,x), ϕ(t,x)]);
    

    discretization = MOLFiniteDifference([x=>dx], t, grid_align=MethodOfLines.StaggeredGrid(), edge_aligned_var=ϕ(t,x));
    prob = discretize(pdesys, discretization);
    
    sol = solve(prob, SplitEuler(), dt=dt);
    
    test_ind = floor(Int, (2(L-dx)/a)/(dt))
    @test maximum(sol[1:128,1] .- sol[1:128,test_ind]) < max(dx^2, dt)
    @test maximum(sol[1:128,1] .- sol[1:128,2*test_ind]) < 10*max(dx^2, dt)
end


@testset "1D wave equation, staggered grid, Neumann BC" begin
    @parameters t x
    @variables ρ(..) ϕ(..)
    Dt = Differential(t);
    Dx = Differential(x);

    a = 5.0;
    L = 8.0;
    dx = 0.125;
    dt = dx/a;
    tmax = 10.0;

    initialFunction(x) = exp(-(x)^2);
    eq = [Dt(ρ(t,x)) + Dx(ϕ(t,x)) ~ 0,
          Dt(ϕ(t,x)) + a^2 * Dx(ρ(t,x)) ~ 0]
    bcs = [ρ(0,x) ~ initialFunction(x),
           ϕ(0.0,x) ~ 0.0,
           Dt(ρ(t,L)) - (1/a)*Dt(ϕ(t,L)) ~ 0.0,
           Dt(ρ(t,-L)) + (1/a)*Dt(ϕ(t,-L)) ~ 0.0]

    domains = [t in Interval(0.0, tmax),
               x in Interval(-L, L)];

    @named pdesys = PDESystem(eq, bcs, domains, [t,x], [ρ(t,x), ϕ(t,x)]);
    

    discretization = MOLFiniteDifference([x=>dx], t, grid_align=MethodOfLines.StaggeredGrid(), edge_aligned_var=ϕ(t,x));
    prob = discretize(pdesys, discretization);
    sol = solve(prob, SplitEuler(), dt=(dx/a)^2);
    @test maximum(sol[:,end]) < 1e-3
end


@testset "1D wave equation, staggered grid, Periodic BC" begin
    @parameters t x
    @variables ρ(..) ϕ(..)
    Dt = Differential(t);
    Dx = Differential(x);

    a = 5.0;
    L = 8.0;
    dx = 0.125;
    dt = (dx/a)^2;
    tmax = 10.0;

    initialFunction(x) = exp(-(x-L/2)^2);
    eq = [Dt(ρ(t,x)) + Dx(ϕ(t,x)) ~ 0,
          Dt(ϕ(t,x)) + a^2 * Dx(ρ(t,x)) ~ 0]
    bcs = [ρ(0,x) ~ initialFunction(x),
           ϕ(0.0,x) ~ 0.0,
           ρ(t,L) ~ ρ(t,-L),
           ϕ(t,-L) ~ ϕ(t,L)];

    domains = [t in Interval(0.0, tmax),
               x in Interval(-L, L)];

    @named pdesys = PDESystem(eq, bcs, domains, [t,x], [ρ(t,x), ϕ(t,x)]);
    

    discretization = MOLFiniteDifference([x=>dx], t, grid_align=MethodOfLines.StaggeredGrid(), edge_aligned_var=ϕ(t,x));
    prob = discretize(pdesys, discretization);
    
    sol = solve(prob, SplitEuler(), dt=dt);
    
    test_ind = round(Int, ((2*L-dx)/a)/(dt))
    @test maximum(sol[1:128,1] .- sol[1:128,test_ind]) < max(dx^2, dt)
    @test maximum(sol[1:128,1] .- sol[1:128,2*test_ind]) < 10*max(dx^2, dt)
end

@testset "1D inhomogenous nonlinear wave equation, staggered grid, Mixed BC" begin
    @parameters x
    @variables t, ρ(..), ϕ(..)
    Dt = Differential(t);
    Dx = Differential(x);

    β = 0.11/(2*0.01);
    a = 0.5;
    L = 1.0;
    dx = 1/2^7;
    tspan = (0.0, 6.0)

    function RHS(ρ, ϕ)
        return -β*(ϕ^2)*sign(ϕ)/ρ
    end

    eqs = [Dt(ρ(t,x)) + Dx(ϕ(t,x)) ~ 0,
           Dt(ϕ(t,x)) + a^2*Dx(ρ(t,x)) ~ RHS(ρ(t, x), ϕ(t,x))]
    bcs = [ρ(0,x) ~ 50.0*exp(-((x-0.5)*20)^2) + 50.0,
           ϕ(0,x) ~ 0.0,
           Dx(ρ(t,L)) ~ 0.0,
           ϕ(t,0.0) ~ 0.0]
    domains = [t in Interval(tspan[1], tspan[end]),
               x in Interval(0.0, L)]

    @named pdesys = PDESystem(eqs, bcs, domains, [t,x], [ρ(t,x), ϕ(t,x)])
    discretization = MOLFiniteDifference([x=>dx], t, grid_align=MethodOfLines.StaggeredGrid(), edge_aligned_var=ϕ(t,x));
    prob = discretize(pdesys, discretization);
    @time sol = solve(prob, SplitEuler(), dt=(dx/a)^2, adaptive=false)
    @test sol.retcode == ReturnCode.Success
    # p = plot()
    # for i in 1:tspan[end]*((a/dx)^2)/10:length(sol)
    #     ind = floor(Int, i)
    #     plot!(p, sol.u[ind], label="t = $(@sprintf("%.2f", sol.t[ind]))")
    # end
    # p
end
