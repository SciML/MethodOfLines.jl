using Test, OrdinaryDiffEq, ModelingToolkit, MethodOfLines, DomainSets, Plots;

function CN_solve(prob, dt)
    u0 = prob.u0
    len = floor(Int, length(u0)/2);
    tsteps = prob.tspan[1]:dt:prob.tspan[2]
    dynamical_f1 = DynamicalODEFunction(prob.f).f1;
    dynamical_f2 = DynamicalODEFunction(prob.f).f2;

    function calc_du!(du, u, p, t)
        placeholder = copy(du);
        dynamical_f1(du, u, p, t);
        dynamical_f2(placeholder, [(u[1:len] + dt*du[1:len]); u[len+1:end]], p, t);
        du[len+1:end] .= placeholder[1:len+1];
        return;
    end

    function update(u, p, t)
        du = zeros(length(u));
        calc_du!(du, u, p, t);
        return u .+ du*dt;
    end

    sol = zeros(length(u0), length(tsteps));
    sol[:,1] .= u0;
    for i in 2:length(tsteps)
        sol[:,i] = update(sol[:,i-1], 0, 0);
    end
    return sol;
end

#@testset "1D wave equation, staggered grid, interior only" begin
    @parameters t x
    @variables ρ(..) ϕ(..)
    Dt = Differential(t);
    Dx = Differential(x);

    a = 5.0;#1.0/2.0;
    L = 8.0;
    dx = 1.0;#0.125;
    dt = dx/a;
    tmax = 1000.0;

    initialFunction(x) = exp(-(x)^2);
    #initialFunction(x) = abs(x-L)/(2*L);
    #initialFunction(x) = tanh(-x)+1;
    eq = [Dt(ρ(t,x)) + Dx(ϕ(t,x)) ~ 0,
          Dt(ϕ(t,x)) + a^2 * Dx(ρ(t,x)) ~ 0]
    bcs = [ρ(0,x) ~ initialFunction(x),
           ϕ(0.0,x) ~ 0.0,
           ρ(t,-L) ~ initialFunction(-L)*exp(-(t^2)),
           ρ(t,L) ~ 0.0,
           ϕ(t,-L) ~ exp((t^2)),
           Dt(ϕ(t,L)) ~ -a^2*Dx(ρ(t,L))];

    domains = [t in Interval(0.0, tmax),
               x in Interval(-L, L)];

    @named pdesys = PDESystem(eq, bcs, domains, [t,x], [ρ(t,x), ϕ(t,x)]);
    

    discretization = MOLFiniteDifference([x=>dx], t, grid_align=MethodOfLines.StaggeredGrid());
    prob = discretize(pdesys, discretization);
    
    sol = CN_solve(prob, dt);
    
    function plot_sol(sol; time_range=1:200:1000)
        p_rho = plot();
        for i in time_range
            plot!(p_rho, sol[1:floor(Int, length(sol[:,1])/2),i])
        end
        plot!(p_rho, title="Density: ρ");
        return plot(p_rho);
    end

    plot_sol(sol)
    @test 1==1
#end
