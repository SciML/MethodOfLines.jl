
# [Tutorial] (@id brusselator)
## Using the Brusselator PDE as an example

The Brusselator PDE is defined as follows:

```math
\begin{align}
\frac{\partial u}{\partial t} &= 1 + u^2v - 4.4u + \alpha(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}) + f(x, y, t)\\
\frac{\partial v}{\partial t} &= 3.4u - u^2v + \alpha(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2})
\end{align}
```

where

```math
f(x, y, t) = \begin{cases}
5 & \quad \text{if } (x-0.3)^2+(y-0.6)^2 ≤ 0.1^2 \text{ and } t ≥ 1.1 \\
0 & \quad \text{else}
\end{cases}
```

and the initial conditions are

```math
\begin{align}
u(x, y, 0) &= 22\cdot (y(1-y))^{3/2} \\
v(x, y, 0) &= 27\cdot (x(1-x))^{3/2}
\end{align}
```

with the periodic boundary condition

```math
\begin{align}
u(x+1,y,t) &= u(x,y,t) \\
u(x,y+1,t) &= u(x,y,t)
\end{align}
```

on a timespan of ``t \in [0,11.5]``.

## Solving with MethodOfLines

With `ModelingToolkit.jl`, we first symbolicaly define the system, see also the docs for (`PDESystem`)[https://mtk.sciml.ai/stable/systems/PDESystem/]:

```julia
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
```
For a list of limitations constraining which systems will work, see [here](@ref limitations)

## Method of lines discretization

Then, we create the discretization, leaving the time dimension undiscretized by supplying `t` as an argument. Optionally, all dimensions can be discretized in this step, just remove the argument `t` and supply `t=>dt` in the `dxs`. See [here](@ref molfd) for more information on the `MOLFiniteDifference` constructor arguments and options.

```julia
N = 32

dx = 1/N
dy = 1/N

order = 2

discretization = MOLFiniteDifference([x=>dx, y=>dy], t, approx_order=order grid_type=center_align)
```
Next, we discretize the system, converting the `PDESystem` in to an `ODEProblem` or `NonlinearProblem`.

```julia
# Convert the PDE problem into an ODE problem
println("Discretization:")
@time prob = discretize(pdesys,discretization)
```

## How it works

MethodOfLines.jl makes heavy use of `Symbolics.jl` and `SymbolicUtils.jl`, namely it's rule matching features to recognize terms which require particular discretizations.

Given your discretization and `PDESystem`, we take each independent variable defined on the space to be discretized and create a corresponding range. We then take each dependant variable and create an array of symbolic variables to represent it in its discretized form. 

Next, the boundary conditions are discretized, creating an equation for each point on the boundary in terms of the discretized variables, replacing any space derivatives in the direction of the boundary with their upwind finite difference expressions.

After that, the system of PDEs is discretized, first matching each PDE to each dependant variable by which variable is highest order in each PDE, with precedance given to time derivatives. Then, the PDEs are discretized creating a finite difference equation for each point in their matched dependant variables discrete form, less the number of boundary equations. These equations are removed from around the boundary, so each PDE only has discrete equations on its variable's interior.

Now we have a system of equations which are either ODEs, linear, or nonlinear equations and an equal number of unknowns. See (here)[] for the system that is generated for the Brusselator at low point count. The structure of the system is simplified with `ModelingToolkit.structural_simplify`, and then either an `ODEProblem` or `NonlinearProblem` is returned. Under the hood, the `ODEProblem` generates a fast semidiscretization, written in julia with `RuntimeGeneratedFunctions`. See (here)[] for an example of the generated code for the Brusselator system at low point count. 

Now your problem can be solved with an appropriate ODE solver, or Nonlinear solver if you have not supplied a time dimension in the `MOLFiniteDifference` constructor. Include these solvers with `using OrdinaryDiffEq` or `using NonlinearSolve`, then call `sol = solve(prob, AppropriateSolver())` or `sol = NonlinearSolve.solve(prob, AppropriateSolver())`. For more information on the available solvers, see the docs for (`DifferentialEquations.jl`)[https://diffeq.sciml.ai/stable/solvers/ode_solve/] and (`NonlinearSolve.jl`)[http://nonlinearsolve.sciml.ai/dev/solvers/NonlinearSystemSolvers/].

```julia
println("Solve:")
@time sol = solve(prob, TRBDF2(), saveat=0.1)
```

To retrieve your solution, for example for `u`, use `sol[u]`. To get the time axis, use `sol.t`.

To get the generated code for your system, use `code = ODEFunctionExpr(prob)`, or `MethodOfLines.generate_code(pdesys, discretization, "my_generated_code_filename.jl")`, which will create a file called `my_generated_code_filename.jl` in `pwd()`. This can be useful to find errors in the discretization, but note that it is not recommended to use this code directly, calling `solve(prob, AppropriateSolver())` will handle this for you.