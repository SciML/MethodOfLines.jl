using OrdinaryDiffEq, ModelingToolkit, MethodOfLines, DomainSets, Plots;

@parameters t x
@variables ρ(..) ϕ(..)
Dt = Differential(t);
Dx = Differential(x);

a = 0.125;
L = 3.0;
dx = 0.25;

relu(x) = max(x,0);
#initialFunction(x) = relu(x-L/4) - 2*relu(x-(2*L/4)) + relu(x-(3*L/4))
initialFunction(x) = relu(x-6*L/8) - 2*relu(x-(7*L/8));

eq = [Dt(ρ(t,x)) + Dx(ϕ(t,x)) ~ 0,
      Dt(ϕ(t,x)) + a^2 * Dx(ρ(t,x)) ~ 0]
bcs = [ρ(0.0,x) ~ initialFunction(x),
       ϕ(0.0,x) ~ 0.0,
       ρ(t,0) ~ ρ(t,L),
       ϕ(t,0) ~ ϕ(t,L)];


domains = [t in Interval(0.0, 10.0),
           x in Interval(0.0, L)];

@named pdesys = PDESystem(eq, bcs, domains, [t,x], [ρ(t,x), ϕ(t,x)]);

#discretization = MOLFiniteDifference([x=>dx], t, grid_align=MethodOfLines.CenterAlignedGrid());
discretization = MOLFiniteDifference([x=>dx], t, grid_align=MethodOfLines.StaggeredGrid());

prob = discretize(pdesys, discretization);

# sol = solve(prob, RK4(), saveat=10.0, dt=dx);
# p = plot();
# for i in 1:length(sol[t])
#     plot!(p, sol[x], sol[ρ(t,x)][i,:], label="t = $(sol[t][i])")
# end
# display(p)
