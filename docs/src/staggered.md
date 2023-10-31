# Staggered Grids

For more examples of staggered grid implementations for the wave equation, see the tests `/test/pde_systems/wave_eq_staggered.jl` 

Hyperbolic PDEs are often most written as a system of first-order equations. When the original system is of second-order (e.g., wave equation), this results in a natural splitting of variables as demonstrated below. Staggered-grid numerical schemes have been developed for such equations as efficient and conservative finite difference schemes.

To utilize this functionality, only two differences are needed to the user-interface. `grid_align=MethodOfLines.StaggeredGrid()` and defining which variable is edge aligned (vs center aligned), `edge_aligned_var=ϕ(t,x)`

Solvers should be chosen carefully, the only officially supported solver is `SplitEuler`, but in theory any SSP solver could be used.

Staggered grid functionality is still in its infancy. Please open issues if unexpected results occur or needed functionality is not present.

```
using OrdinaryDiffEq, ModelingToolkit, MethodOfLines, DomainSets
@parameters t x
@variables ρ(..) ϕ(..)
Dt = Differential(t);
Dx = Differential(x);

a = 5.0; #wave speed
L = 8.0;
dx = 0.125;
dt = dx/a; # CFL condition
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
```
