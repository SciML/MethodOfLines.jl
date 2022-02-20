# MethodOfLines.jl

Provides automatic discretization of symbolic PDE systems as defined with ModelingToolkit.jl.

Feature requests and issues welcome.

## Usage:
```
discretization = MOLFiniteDifference(dxs, 
                                      <your choice of continuous variable, usually time>; 
                                      upwind_order = <currently hard coded to 1>, 
                                      approx_order = Order of derivative approximation, starting from 2 
                                      grid_align = your grid type choice>)
prob = discretize(pdesys, discretization)
```
Where `dxs` is a vector of pairs of parameters to the grid step in this dimension, i.e. `[x=>0.2, y=>0.1]`

Note that the second argument to `MOLFiniteDifference` is optional, all parameters can be discretized if all boundary conditions are specified

Currently supported grid types: `center_align` and `edge_align`. Edge align will give better accuracy with Neumann Boundary conditions.

`center_align`: naive grid, starting from lower boundary, ending on upper boundary with step of `dx`

`edge_align`: offset grid, set halfway between the points that would be generated with center_align, with extra points at either end that are above and below the supremum and infimum by `dx/2`.

## Assumptions
- That the term of a boundary condition is defined on the edge of the domain and is applied additively and has no multiplier/divisor/power etc.
- That periodic boundary conditions are of the simple form `u(t, x_min) ~ u(t, x_max)`. Note that this generalises to higher dimensions
- That boundary conditions only contain references to the variable on which they are defined at the edge of the domain, i.e. if `u(t,0)` is defined there are no references to `v(t,0)`. Note that references to dependent variables with all of their arguments are allowed such as `w(t)` or `v(t,x)` if the condition is on `u(t,x,y_min)`.
- That initial conditions are of the form `u(...) ~ ...`, and doesn't reference the initial time derivative.
- That simple derivative terms are purely of a dependant variable, for example `Dx(u(t,x,y))`is allowed but `Dx(u(t,x,y)*v(t,x,y))`, `Dx(u(t,x)+1)` or `Dx(f(u(t,x)))` are not. As a workaround please expand such terms with the product/chain rules and use the linearity of the derivative operator, or define a new dependant variable equal to the term to be differentiated. Exceptions to this are the nonlinear or spherical laplacian, which have special handling.

If any of these limitations are a problem for you please post an issue and we will prioritize removing them.

## Coming soon:
- Fewer Assumptions

## Full Example:
```
## 2D Diffusion

# Dependencies
using ModelingToolkit, MethodOfLines, LinearAlgebra, OrdinaryDiffEq, DomainSets
using ModelingToolkit: Differential

# Variables, parameters, and derivatives
@parameters t x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
Dt = Differential(t)
# Domain edges
t_min= 0.
t_max = 2.0
x_min = 0.
x_max = 2.
y_min = 0.
y_max = 2.

# Discretization parameters
dx = 0.1; dy = 0.2
order = 2

# Analytic solution for boundary conditions
analytic_sol_func(t, x, y) = exp(x + y) * cos(x + y + 4t)

# Equation
eq  = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y))

# Initial and boundary conditions
bcs = [u(t_min,x,y) ~ analytic_sol_func(t_min, x, y),
        u(t, x_min, y) ~ analytic_sol_func(t, x_min, y),
        u(t, x_max, y) ~ analytic_sol_func(t, x_max, y),
        u(t, x,y_min) ~ analytic_sol_func(t, x, y_min),
        u(t, x, y_max) ~ analytic_sol_func(t, x, y_max)]

# Space and time domains
domains = [t ∈ Interval(t_min, t_max),
           x ∈ Interval(x_min, x_max),
           y ∈ Interval(y_min, y_max)]

# PDE system
@named pdesys = PDESystem([eq], bcs, domains, [t, x, y], [u(t, x, y)])

# Method of lines discretization
discretization = MOLFiniteDifference([x=>dx,y=>dy],t;centered_order=order)
prob = ModelingToolkit.discretize(pdesys,discretization)

# Solution of the ODE system
sol = solve(prob,Tsit5())
```
