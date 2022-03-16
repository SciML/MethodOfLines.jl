# MethodOfLines.jl

MethodOfLines.jl is a package for automated finite difference discretization
of symbolicaly-defined PDEs in N dimensions.

It uses symbolic expressions for systems of partial differential equations as defined with `ModelingToolkit.jl`, and `Interval` from `DomainSets.jl` to define the space(time) over which the simulation runs.

Allowable terms in the system and bcs include, but are not limited to
- Advection
- Diffusion
- Reaction
- Nonlinear Diffusion
- Spherical laplacian
- Any julia function of the symbolic parameters/dependant variables and other parameters in the environment that's defined on the whole domain.

# Discretization
It discrertizes the above with a `MOLFiniteDifference`, with the following interface:

```
eq = [your system of equations, see examples for possibilities]
bcs = [your boundary conditions, see examples for possibilities]

domain = [your domain, a vector of Intervals i.e. x ∈ Interval(x_min, x_max)]

@named pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])

discretization = MOLFiniteDifference(dxs, 
                                      <your choice of continuous variable, usually time>; 
                                      upwind_order = <Currently unstable at any value other than 1>, 
                                      approx_order = <Order of derivative approximation, starting from 2> 
                                      grid_align = <your grid type choice>)
prob = discretize(pdesys, discretization)
```
Where `dxs` is a vector of pairs of parameters to the grid step in this dimension, i.e. `[x=>0.2, y=>0.1]`

Note that the second argument to `MOLFiniteDifference` is optional, all parameters can be discretized if all required boundary conditions are specified.

Currently supported grid types: `center_align` and `edge_align`. Edge align will give better accuracy with Neumann Boundary conditions.

`center_align`: naive grid, starting from lower boundary, ending on upper boundary with step of `dx`

`edge_align`: offset grid, set halfway between the points that would be generated with center_align, with extra points at either end that are above and below the supremum and infimum by `dx/2`. This improves accuracy for neumann BCs.

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
