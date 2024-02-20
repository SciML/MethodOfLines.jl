# [Solving the Heat Equation](@id heat)

In this tutorial, we will use the symbolic interface to solve the heat equation.

### Dirichlet boundary conditions

```@example heatd
using OrdinaryDiffEq, ModelingToolkit, MethodOfLines, DomainSets
# Method of Manufactured Solutions: exact solution
u_exact = (x, t) -> exp.(-t) * cos.(x)

# Parameters, variables, and derivatives
@parameters t x
@variables u(..)
Dt = Differential(t)
Dxx = Differential(x)^2

# 1D PDE and boundary conditions
eq = Dt(u(t, x)) ~ Dxx(u(t, x))
bcs = [u(0, x) ~ cos(x),
    u(t, 0) ~ exp(-t),
    u(t, 1) ~ exp(-t) * cos(1)]

# Space and time domains
domains = [t ∈ Interval(0.0, 1.0),
    x ∈ Interval(0.0, 1.0)]

# PDE system
@named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

# Method of lines discretization
dx = 0.1
order = 2
discretization = MOLFiniteDifference([x => dx], t)

# Convert the PDE problem into an ODE problem
prob = discretize(pdesys, discretization)

# Solve ODE problem
using OrdinaryDiffEq
sol = solve(prob, Tsit5(), saveat = 0.2)

# Plot results and compare with exact solution
discrete_x = sol[x]
discrete_t = sol[t]
solu = sol[u(t, x)]

using Plots
plt = plot()

for i in eachindex(discrete_t)
    plot!(discrete_x, solu[i, :], label = "Numerical, t=$(discrete_t[i])")
    scatter!(
        discrete_x, u_exact(discrete_x, discrete_t[i]), label = "Exact, t=$(discrete_t[i])")
end
plt
```

### Neumann boundary conditions

```@example heatn
using OrdinaryDiffEq, ModelingToolkit, MethodOfLines, DomainSets
# Method of Manufactured Solutions: exact solution
u_exact = (x, t) -> exp.(-t) * cos.(x)

# Parameters, variables, and derivatives
@parameters t x
@variables u(..)
Dt = Differential(t)
Dx = Differential(x)
Dxx = Differential(x)^2

# 1D PDE and boundary conditions
eq = Dt(u(t, x)) ~ Dxx(u(t, x))
bcs = [u(0, x) ~ cos(x),
    Dx(u(t, 0)) ~ 0.0,
    Dx(u(t, 1)) ~ -exp(-t) * sin(1)]

# Space and time domains
domains = [t ∈ Interval(0.0, 1.0),
    x ∈ Interval(0.0, 1.0)]

# PDE system
@named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

# Method of lines discretization
# Need a small dx here for accuracy
dx = 0.01
order = 2
discretization = MOLFiniteDifference([x => dx], t)

# Convert the PDE problem into an ODE problem
prob = discretize(pdesys, discretization)

# Solve ODE problem
using OrdinaryDiffEq
sol = solve(prob, Tsit5(), saveat = 0.2)

# Plot results and compare with exact solution
discrete_x = sol[x]
discrete_t = sol[t]

solu = sol[u(t, x)]

using Plots
plt = plot()

for i in eachindex(discrete_t)
    plot!(discrete_x, solu[i, :], label = "Numerical, t=$(discrete_t[i])")
    scatter!(
        discrete_x, u_exact(discrete_x, discrete_t[i]), label = "Exact, t=$(discrete_t[i])")
end
plt
```

### Robin boundary conditions

```@example heatr
using ModelingToolkit, MethodOfLines, DomainSets, OrdinaryDiffEq
# Method of Manufactured Solutions
u_exact = (x, t) -> exp.(-t) * sin.(x)

# Parameters, variables, and derivatives
@parameters t x
@variables u(..)
Dt = Differential(t)
Dx = Differential(x)
Dxx = Differential(x)^2

# 1D PDE and boundary conditions
eq = Dt(u(t, x)) ~ Dxx(u(t, x))
bcs = [u(0, x) ~ sin(x),
    u(t, -1.0) + 3Dx(u(t, -1.0)) ~ exp(-t) * (sin(-1.0) + 3cos(-1.0)),
    u(t, 1.0) + Dx(u(t, 1.0)) ~ exp(-t) * (sin(1.0) + cos(1.0))]

# Space and time domains
domains = [t ∈ Interval(0.0, 1.0),
    x ∈ Interval(-1.0, 1.0)]

# PDE system
@named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

# Method of lines discretization
# Need a small dx here for accuracy
dx = 0.05
order = 2
discretization = MOLFiniteDifference([x => dx], t)

# Convert the PDE problem into an ODE problem
prob = discretize(pdesys, discretization)

# Solve ODE problem
using OrdinaryDiffEq
sol = solve(prob, Tsit5(), saveat = 0.2)

# Plot results and compare with exact solution
discrete_x = sol[x]
discrete_t = sol[t]

solu = sol[u(t, x)]

using Plots
plt = plot()

for i in eachindex(discrete_t)
    plot!(discrete_x, solu[i, :], label = "Numerical, t=$(discrete_t[i])")
    scatter!(
        discrete_x, u_exact(discrete_x, discrete_t[i]), label = "Exact, t=$(discrete_t[i])")
end
plt
```
