# [MethodOfLines.jl] (@id index)

MethodOfLines.jl is a package for automated finite difference discretization
of symbolicaly-defined PDEs in N dimensions.

It uses symbolic expressions for systems of partial differential equations as defined with `ModelingToolkit.jl`, and `Interval` from `DomainSets.jl` to define the space(time) over which the simulation runs.

The package's handling is quite general, it is recommended to try out your system of equations and post an issue if you run in to trouble. If you want to solve it, we want to support it.

Issues with questions on usage are also welcome as they help us improve the docs.

See [here](@ref brusselator) for a full tutorial, involving the Brusselator equation.

Allowable terms in the system include, but are not limited to
- Advection
- Diffusion
- Reaction
- Nonlinear Diffusion
- Spherical laplacian
- Any Julia function of the symbolic parameters/dependant variables and other parameters in the environment that's defined on the whole domain. Note that more complicated functions may require registration with `@register`, see the [ModelingToolkit.jl docs](https://mtk.sciml.ai/stable/basics/Validation/#User-Defined-Registered-Functions-and-Types).

Boundary conditions include, but are not limited to:
- Dirichlet
- Neumann (can also include time derivative)
- Robin (can also include time derivative)
- Periodic
- Any function, subject to the assumptions below

At the moment the centered difference, upwind difference, nonlinear laplacian and spherical laplacian schemes are implemented. If you know of a scheme with better stability or accuracy in any specific case, please post an issue with a link to a paper.

## [Known Limitations] (@id limitations)

At the moment the package is able to discretize almost any system, with some assumptions listed below

- That the grid is cartesian.
- That the equation is first order in time.
- Boundary conditions in time are supplied as initial conditions, not at the end of the simulation interal. If your system requires a final condition, please use a change of variables to rectify this. This is unlikely to change due to upstream constraints.
- Intergral equations are not supported.
- That dependant variables always have the same argument signature, except in BCs.
- That periodic boundary conditions are of the simple form `u(t, x_min) ~ u(t, x_max)`, or the same with lhs and rhs reversed. Note that this generalises to higher dimensions.
- That boundary conditions do not contain references to derivatives which are not in the direction of the boundary, except in time.
- That initial conditions are of the form `u(...) ~ ...`, and don't reference the initial time derivative.
- That simple derivative terms are purely of a dependant variable, for example `Dx(u(t,x,y))` is allowed but `Dx(u(t,x,y)*v(t,x,y))`, `Dx(u(t,x)+1)` or `Dx(f(u(t,x)))` are not. As a workaround please expand such terms with the product rule and use the linearity of the derivative operator, or define a new auxiliary dependant variable by adding an equation for it like `eqs = [Differential(x)(w(t,x))~ ... , w(t,x) ~ v(t,x)*u(t,x)]`, along with appropriate BCs/ICs. An exception to this is if the differential is a nonlinear or spherical laplacian, in which case only the innermost argument should be wrapped.
- That odd order derivatives do not multiply or divide each other. A workaround is to wrap all but one derivative per term in an auxiliary variable, such as `dxu(x, t) ~ Differential(x)(u(x, t))`.

The performance hit from auxiliary variables should be negligable due to a structural simplification step.

If any of these limitations are a problem for you please post an issue and we will prioritize removing them. If you discover a limitation that isn't listed here, pleae post an issue with example code.

### If you have any usage questions or feature requests, please post an issue
