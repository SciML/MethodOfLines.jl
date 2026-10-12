# Staggered Grids

For more examples of staggered grid implementations for the wave equation, see the tests `/test/pde_systems/wave_eq_staggered.jl`

Hyperbolic PDEs are often most written as a system of first-order equations. When the original system is of second-order (e.g., wave equation), this results in a natural splitting of variables as demonstrated below. Staggered-grid numerical schemes have been developed for such equations as efficient and conservative finite difference schemes.

To utilize this functionality, only two differences are needed to the user-interface. `grid_align=MethodOfLines.StaggeredGrid()` and defining which variable is edge aligned (vs center aligned), `edge_aligned_var=ϕ(t,x)`

`edge_aligned_var` also accepts a collection, such as `edge_aligned_var=[ϕ(t,x), ψ(t,x)]`, to edge align several variables. Every dependent variable it does not name is center aligned. A variable without a spatial derivative, such as the current `J` in Maxwell's equations with a Drude medium (`Dt(J(t,x)) ~ ωp^2 * E(t,x)`), should share the alignment of the variable it couples to pointwise, here the center aligned `E`.

A center aligned variable sits on the grid points, and an edge aligned one half a step past them: a derivative of a center aligned variable is taken between its points `i` and `i + 1`, and of an edge aligned one between `i - 1` and `i`. Initial conditions, coefficients that depend on the coordinates, and the interpolation of the solution all use these points; `sol[u(t, x)]` holds the values at the points of `u`. A Dirichlet condition on a domain wall needs the variable to sit on the grid points in the direction normal to that wall, so it must be center aligned there.

In more than one dimension the alignment is set per direction: an entry `u => x` or `u => (x, y)` edge aligns `u` in those directions only, and a bare `u` edge aligns it in all of its directions. On a Yee grid each component of E is staggered along itself and each component of H across itself. With perfectly conducting walls, where tangential E vanishes, the 3D layout is

```julia
edge_aligned_var = [Ex(t, x, y, z) => x, Ey(t, x, y, z) => y, Ez(t, x, y, z) => z,
                    Hx(t, x, y, z) => (y, z), Hy(t, x, y, z) => (x, z), Hz(t, x, y, z) => (x, y)]
```

and the 2D TE layout is `[Hz(t, x, y), Ex(t, x, y) => x, Ey(t, x, y) => y]`.

`discretize` returns a `SplitODEProblem` whose `f1` holds the time derivatives of the variables edge aligned in an odd number of directions and whose `f2` holds those of every other variable; `f1 + f2` is the full right hand side. A staggered derivative always couples the two groups, so for Maxwell's equations they are E and H.

Solvers should be chosen carefully, the only officially supported solver is `SplitEuler`, but in theory any SSP solver could be used.

Staggered grid functionality is still in its infancy. Please open issues if unexpected results occur or needed functionality is not present.

Staggered grids emit the interior of each PDE as a single symbolic array equation over slices, keeping the symbolic equation count independent of the grid resolution.

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
sys, tspan = symbolic_discretize(pdesys, discretization)
prob = ODEProblem(mtkcompile(sys), nothing, tspan);

sol = solve(prob, SplitEuler(), dt=dt);
```
