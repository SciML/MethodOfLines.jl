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


## Coming soon:
- Fewer Assumptions.
- More robust testing and validation.
- Benchmarks.
## Full Example:
```
## 2D Diffusion

# Dependencies
using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, DomainSets
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
discretization = MOLFiniteDifference([x=>dx, y=>dy], t; approx_order = order)
prob = ModelingToolkit.discretize(pdesys, discretization)

# Solution of the ODE system
sol = solve(prob,Tsit5())
```

# Reaction Diffusion example, Brusselator Equation
```
using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, DomainSets


@parameters x y t
@variables u(..) v(..)
Dt = Differential(t)
Dx = Differential(x)
Dy = Differential(y)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

∇²(u) = Dxx(u) + Dyy(u)

brusselator_f(x, y, t) = (((x-0.3)^2 + (y-0.6)^2) <= 0.1^2) * (t >= 1.1) * 5.

x_min = y_min = t_min = 0.0
x_max = y_max = 1.0
t_max = 11.5

α = 10.

u0(x,y,t) = 22(y*(1-y))^(3/2)
v0(x,y,t) = 27(x*(1-x))^(3/2)

eq = [Dt(u(x,y,t)) ~ 1. + v(x,y,t)*u(x,y,t)^2 - 4.4*u(x,y,t) + α*∇²(u(x,y,t)) + brusselator_f(x, y, t),
       Dt(v(x,y,t)) ~ 3.4*u(x,y,t) - v(x,y,t)*u(x,y,t)^2 + α*∇²(v(x,y,t))]

domains = [x ∈ Interval(x_min, x_max),
              y ∈ Interval(y_min, y_max),
              t ∈ Interval(t_min, t_max)]

# Periodic BCs
bcs = [u(x,y,0) ~ u0(x,y,0),
       u(0,y,t) ~ u(1,y,t),
       u(x,0,t) ~ u(x,1,t),

       v(x,y,0) ~ v0(x,y,0),
       v(0,y,t) ~ v(1,y,t),
       v(x,0,t) ~ v(x,1,t)] 

@named pdesys = PDESystem(eq,bcs,domains,[x,y,t],[u(x,y,t),v(x,y,t)])

# Method of lines discretization
N = 32

dx = 1/N
dy = 1/N

order = 2

discretization = MOLFiniteDifference([x=>dx, y=>dy], t, approx_order=order)

# Convert the PDE problem into an ODE problem
println("Discretization:")
@time prob = discretize(pdesys,discretization)

println("Solve:")
@time sol = solve(prob, TRBDF2(),saveat=0.01)
```