using ModelingToolkit, MethodOfLines, DomainSets, OrdinaryDiffEq, Test

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

@named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

disc = MOLFiniteDifference([x => 30], t, advection_scheme=WENOScheme())

prob = discretize(pdesys, disc; analytic = [u(t, x) => analytic_u])

sol = solve(prob, FBDF())

@test prob.f.analytic !== nothing
