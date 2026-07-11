# [Solving PIDEs (Integrals)](@id integral)

It is also possible to solve PIDEs with MethodOfLines.jl.

Consider the following system:

```math
 \frac{\partial}{\partial t}u(t, x)+2u(t, x)+5\frac{\partial}{\partial x}[\int_0^xu(t, x)dx]=1
```

On the domain:

```math
 t \in (0, 2)
 x \in (0, 2\pi)
```

With BCs and ICs:

```math
 u(0, x)=cos(x)
 \frac{\partial}{\partial x}u(t, 0)=0
 \frac{\partial}{\partial x}u(t, 2)=0
```

We can discretize such a system like this:

```@example pide
using MethodOfLines, ModelingToolkit, OrdinaryDiffEq, DomainSets, Plots

@parameters t, x
@variables u(..) cumuSum(..)
Dt = Differential(t)
Dx = Differential(x)
xmin = 0.0
xmax = 2.0pi

# Integral limits are defined with DomainSets.ClosedInterval
Ix = Integral(x in DomainSets.ClosedInterval(xmin, x)) # basically cumulative sum from 0 to x

eq = [
    cumuSum(t, x) ~ Ix(u(t, x)), # Note wrapping the argument to the derivative with an auxiliary variable
    Dt(u(t, x)) + 2 * u(t, x) + 5 * Dx(cumuSum(t, x)) ~ 1
]
bcs = [u(0.0, x) ~ cos(x), Dx(u(t, xmin)) ~ 0.0, Dx(u(t, xmax)) ~ 0]

domains = [t ∈ Interval(0.0, 2.0), x ∈ Interval(xmin, xmax)]

@named pde_system = PDESystem(eq, bcs, domains, [t, x], [u(t, x), cumuSum(t, x)])

order = 2
discretization = MOLFiniteDifference([x => 30], t, approx_order = order)

prob = MethodOfLines.discretize(pde_system, discretization)
sol = solve(prob, QNDF(), saveat = 0.1);

solu = sol[u(t, x)]

plot(sol[x], transpose(solu))
```

To have an integral over the whole domain, be sure to wrap the integral in an auxiliary variable.
Due to a limitation, the whole domain integral needs to have the same arguments as the integrand, but is constant in x. To use it in an equation one dimension lower, use a boundary value like integral(t, 0.0)

```julia
using MethodOfLines, ModelingToolkit, DomainSets, OrdinaryDiffEq, Plots

@parameters t, x
@variables integrand(..) integral(..)
Dt = Differential(t)
Dx = Differential(x)
xmin = 0.0
xmax = 2.0 * pi

Ix = Integral(x in DomainSets.ClosedInterval(xmin, xmax)) # integral over domain

eqs = [integral(t, x) ~ Ix(integrand(t, x))
       integrand(t, x) ~ t * cos(x)]

bcs = [integral(0, x) ~ 0.0,
    integrand(0, x) ~ 0.0]

domains = [t ∈ Interval(0.0, 1.0),
    x ∈ Interval(xmin, xmax)]

@named pde_system = PDESystem(eqs, bcs, domains, [t, x], [integrand(t, x), integral(t, x)])

asf(t) = 0.0

disc = MOLFiniteDifference([x => 120], t)

prob = discretize(pde_system, disc)

sol = solve(prob, Tsit5())

xdisc = sol[x]
tdisc = sol[t]

integralsol = sol[integral(t, x)]

heatmap(integralsol)
```
