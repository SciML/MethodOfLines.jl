# MethodOfLines.jl

Provides automatic discretization of symbolic PDE systems as defined with ModelingToolkit.jl.

Feature requests and issues welcome.

## Usage:
```
discretization = MOLFiniteDifference(dxs, 
                                      <your choice of continuous variable, usually time>; 
                                      upwind_order = <currently hard coded to 1>, 
                                      centered_order = <currently hard coded to 2>, 
                                      grid_align = your grid type choice>)
prob = discretize(pdesys, discretization)
```
Where `dxs` is a vector of pairs of parameters to the grid step in this dimension, i.e. `[x=>0.2, y=>0.1]`

Note that the second argument to `MOLFiniteDifference` is optional, all parameters can be discretized if all boundary conditions are specified

Currently supported grid types: `center_align` and `edge_align`

`center_align`: naive grid, starting from lower boundary, ending on upper boundary with step of `dx`

`edge_align`: offset grid, set halfway between the points that would be generated with center_align, with extra points at either end that are above and below the supremum and infimum by `dx/2`.

Currently boundary conditions defined in terms of derivatives at the boundary are unsupported above 1 discretized dimension. Periodic conditions are also unsupported.

## Coming soon:
- Arbitrary approximation order
- Above mentioned unsupported boundary conditions supported

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

## Credit 
The basis for this package was provided by the following contributors:
- Valentin Sulzer @tinosulzer
- Akash Garg @akashgarg
- Emmanuel Luján @emmanuellujan
- Christopher Rackauckas @ChrisRackauckas
- @mjsheikh
- Yingbo Ma @YingboMa
