# Non-Uniform Rectilinear Grids

For more information on how to use a non-uniform rectilinear grid, see the docs for [MOLFiniteDifference](@ref molfd).

MethodOfLines exports the function `chebyspace`, which can be used to conveniently construct a Chebyshev grid, which may prove more 
accurate in certain cases (benchmarking to come, watch this space). It takes the arguments `chebyspace(N, dom)` where `N` is the number of points, and `dom` is the domain set for the variable you want to discretize in this way.

```julia
using OrdinaryDiffEq, ModelingToolkit, MethodOfLines, DomainSets

# Parameters, variables, and derivatives
@parameters t x
@variables u(..)
Dt = Differential(t)
Dxx = Differential(x)^2

# 1D PDE and boundary conditions
eq  = Dt(u(t, x)) ~ Dxx(u(t, x))
bcs = [u(0, x) ~ cos(x),
        u(t, 0) ~ exp(-t),
        u(t, 1) ~ exp(-t) * cos(1)]

# Space and time domains
domains = [t ∈ Interval(0.0, 1.0),
           x ∈ Interval(0.0, 1.0)]

# PDE system
@named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

# Method of lines discretization
discx = chebyspace(100, domains[2]) # 100 point Chebyshev space, pair `x => points`
discretization = MOLFiniteDifference([discx], t)

prob = discretize(pdesys, discretization)
```