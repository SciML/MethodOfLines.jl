using MethodOfLines, DomainSets, OrdinaryDiffEq, ModelingToolkit

@parameters x t
@variables u(..)
Dx = Differential(x)
Dt = Differential(t)
x_min = 0.0
x_max = 1.0
t_min = 0.0
t_max = 6.0

analytic_u(t, x) = x / (t + 1)

eq = Dt(u(t, x)) ~ -u(t, x) * Dx(u(t, x))

bcs = [u(0, x) ~ x,
    u(t, x_min) ~ analytic_u(t, x_min),
    u(t, x_max) ~ analytic_u(t, x_max)]

domains = [t ∈ Interval(t_min, t_max),
    x ∈ Interval(x_min, x_max)]

dx = 0.05

@named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

disc = MOLFiniteDifference([x => dx], t, advection_scheme=PPMScheme(0.1))

prob = discretize(pdesys, disc)

sol = solve(prob, Euler(), dt=0.1)
