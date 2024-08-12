using ModelingToolkit, MethodOfLines, DomainSets, OrdinaryDiffEq, Test

@parameters x t a
@variables u(..)
Dx = Differential(x)
Dt = Differential(t)
x_min = 0.0
x_max = 1.0 
t_min = 0.0
t_max = 6.0

function analytic_u(p, t, x) 
    if p isa Vector
        p = p[1]
    else
        p = p[1][1]
    end
    x / (t + p[1])
end
eq = Dt(u(t, x)) ~ -a * u(t, x) * Dx(u(t, x))
 
bcs = [u(0, x) ~ x,
    u(t, x_min) ~ analytic_u([1], t, x_min),
    u(t, x_max) ~ analytic_u([1], t, x_max)]

domains = [t ∈ Interval(t_min, t_max),
    x ∈ Interval(x_min, x_max)]

@named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)], [a],
    analytic_func = [u(t, x) => analytic_u], defaults = Dict(a => 1.0))

disc = MOLFiniteDifference([x => 30], t, advection_scheme = WENOScheme())

prob = discretize(pdesys, disc; analytic = pdesys.analytic_func)

sol = solve(prob, FBDF())

#@test prob.f.analytic !== nothing
