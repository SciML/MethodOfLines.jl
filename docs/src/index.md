# MethodOfLines.jl

MethodOfLines.jl is a package for automated finite difference discretization
of symbolicaly-defined PDEs in N dimensions.

It uses symbolic expressions for systems of partial differential equations as defined with `ModelingToolkit.jl`, and `Interval` from `DomainSets.jl` to define the space(time) over which the simulation runs.

See (here)[/tutorials/brusselator.md] for a full tutorial.

Allowable terms in the system include, but are not limited to
- Advection
- Diffusion
- Reaction
- Nonlinear Diffusion
- Spherical laplacian
- Any julia function of the symbolic parameters/dependant variables and other parameters in the environment that's defined on the whole domain.

Boundary conditions include, but are not limited to:
- Dirichlet
- Neumann (can also include time derivative)
- Robin (can also include time derivative)
- Periodic
- Any function, subject to the assumptions below

At the moment the centered difference, upwind difference, nonlinear laplacian and spherical laplacian schemes are implemented. If you know of a scheme with better stability or accuracy in any specific case, please post an issue with a link to a paper.

## Known Limitations

At the moment the package is able to discretize almost any system, with some assumptions listed below

- That the grid is cartesian.
- That the equation is first order in time.
- That periodic boundary conditions are of the simple form `u(t, x_min) ~ u(t, x_max)`, or the same with lhs and rhs reversed. Note that this generalises to higher dimensions.
- That boundary conditions do not contain references to derivatives which are not in the direction of the boundary, except in time.
- That initial conditions are of the form `u(...) ~ ...`, and don't reference the initial time derivative.
- That simple derivative terms are purely of a dependant variable, for example `Dx(u(t,x,y))` is allowed but `Dx(u(t,x,y)*v(t,x,y))`, `Dx(u(t,x)+1)` or `Dx(f(u(t,x)))` are not. As a workaround please expand such terms with the product/chain rules and use the linearity of the derivative operator, or define a new dependant variable by adding an equation for it like `eqs = [Differential(x)(w(t,x))~ ... , w(t,x) ~ v(t,x)*u(t,x)]`. An exception to this is if the differential is a nonlinear or spherical laplacian, in which case only the innermost argument should be wrapped.

If any of these limitations are a problem for you please post an issue and we will prioritize removing them. If you discover a limitation that isn't listed here, pleae post an issue with example code.
