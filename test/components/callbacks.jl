using ModelingToolkit, MethodOfLines, DomainSets, Test, Symbolics, SymbolicUtils, OrdinaryDiffEq

@testset "Discrete callback" begin
    @parameters x, t
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    t_min = 0.0
    t_max = 2.0
    x_min = 0.0
    x_max = 2.0

    a = 0.1*pi

    eq = Dt(u(t, x)) ~ a*Dxx(u(t, x))

    bcs = [u(t_min, x) ~ sinpi(x), u(t, x_min) ~ sinpi(t+x), u(t, x_max) ~ sinpi(t+x)]

    domains = [t ∈ Interval(t_min, t_max), x ∈ Interval(x_min, x_max)]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    cb = MOLDiscCallback((s, p) -> prod(p), [0.1, pi])
    a_cb = cb.sym
    cbeq = Dt(u(t, x)) ~ a_cb*Dxx(u(t, x))

    @named cbpdesys = PDESystem(cbeq, bcs, domains, [t, x], [u(t, x)])

    disc = MOLFiniteDifference([x => 0.1], t; approx_order=2, callbacks = [cb])

    prob = discretize(pdesys, disc)
    cbprob = discretize(cbpdesys, disc)

    sol1 = solve(prob, Tsit5(), saveat=0.1)
    sol2 = solve(cbprob, Tsit5(), saveat=0.1) 

    @test sol1[u(t, x)] ≈ sol2[u(t, x)]
end