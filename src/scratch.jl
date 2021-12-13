include("MethodOfLines.jl")
using ModelingToolkit, DomainSets, DiffEqBase

# Variables, parameters, and derivatives
@parameters t x y
@variables u(..)
Dx = Differential(x)
Dy = Differential(y)
Dt = Differential(t)
t_min= 0.
t_max = 2.0
x_min = 0.
x_max = 2.
y_min = 0.
y_max = 2.
dx = 0.1; dy = 0.2

order = 2

# Analytic solution
analytic_sol_func(t,x,y) = exp(x+y)*cos(x+y+4t)

# Equation
eq = Dt(u(t,x,y)) ~ Dx( (u(t,x,y)^2 / exp(x+y)^2 + sin(x+y+4t)^2)^0.5 * Dx(u(t,x,y))) +
                    Dy( (u(t,x,y)^2 / exp(x+y)^2 + sin(x+y+4t)^2)^0.5 * Dy(u(t,x,y)))

# Initial and boundary conditions
bcs = [u(t_min,x,y) ~ analytic_sol_func(t_min,x,y),
        u(t,x_min,y) ~ analytic_sol_func(t,x_min,y),
        u(t,x_max,y) ~ analytic_sol_func(t,x_max,y),
        u(t,x,y_min) ~ analytic_sol_func(t,x,y_min),
        u(t,x,y_max) ~ analytic_sol_func(t,x,y_max)]

# Space and time domains
domains = [t ∈ Interval(t_min,t_max),
        x ∈ Interval(x_min,x_max),
        y ∈ Interval(y_min,y_max)]

# Space and time domains
@named pdesys = PDESystem([eq],bcs,domains,[t,x,y],[u(t,x,y)])
discretization = MethodOfLines.MOLFiniteDifference([x=>dx,y=>dy],t;centered_order=order)

prob = ModelingToolkit.discretize(pdesys,discretization)
