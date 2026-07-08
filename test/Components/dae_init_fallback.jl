using MethodOfLines, ModelingToolkit, DomainSets, Test, OrdinaryDiffEq, SciMLBase
using DiffEqBase: BrownFullBasicInit, CheckInit
using ModelingToolkit: Differential

@testset "is_implicit_dae" begin
    @parameters t x
    @variables u(..) v(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    pdae_eqs = [
        Dt(u(t, x)) ~ Dxx(u(t, x)),
        0 ~ Dxx(v(t, x)) + exp(-t) * sin(x),
    ]
    pdae_bcs = [
        u(0, x) ~ cos(x),
        v(0, x) ~ sin(x),
        u(t, 0) ~ exp(-t),
        Differential(x)(u(t, 1)) ~ -exp(-t) * sin(1),
        Differential(x)(v(t, 0)) ~ exp(-t),
        v(t, 1) ~ exp(-t) * sin(1),
    ]
    ode_eq = Dt(u(t, x)) ~ Dxx(u(t, x))
    ode_bcs = [u(0, x) ~ sin(x), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0]
    domains = [t ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0)]

    @named pdae_sys = PDESystem(pdae_eqs, pdae_bcs, domains, [t, x], [u(t, x), v(t, x)])
    @named ode_sys = PDESystem(ode_eq, ode_bcs, domains, [t, x], [u(t, x)])

    dx = 1 / 19
    pdae_prob = discretize(pdae_sys, MOLFiniteDifference([x => dx], t))
    ode_prob = discretize(ode_sys, MOLFiniteDifference([x => dx], t))

    @test MethodOfLines.is_implicit_dae(pdae_prob)
    @test !MethodOfLines.is_implicit_dae(ode_prob)
end

@testset "apply_dae_initialization_fallback" begin
    @parameters t x
    @variables u(..) v(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    eqs = [
        Dt(u(t, x)) ~ Dxx(u(t, x)),
        0 ~ Dxx(v(t, x)) + exp(-t) * sin(x),
    ]
    bcs = [
        u(0, x) ~ cos(x),
        v(0, x) ~ sin(x),
        u(t, 0) ~ exp(-t),
        Differential(x)(u(t, 1)) ~ -exp(-t) * sin(1),
        Differential(x)(v(t, 0)) ~ exp(-t),
        v(t, 1) ~ exp(-t) * sin(1),
    ]
    domains = [t ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eqs, bcs, domains, [t, x], [u(t, x), v(t, x)])
    disc = MOLFiniteDifference([x => 1 / 19], t)

    prob = discretize(pdesys, disc)
    @test haskey(prob.kwargs, :initializealg)
    @test prob.kwargs[:initializealg] isa BrownFullBasicInit

    sol = solve(prob, FBDF(), saveat = 0.1)
    @test SciMLBase.successful_retcode(sol)

    @named ode_only = PDESystem(
        [Dt(u(t, x)) ~ Dxx(u(t, x))],
        [u(0, x) ~ sin(x), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0],
        domains, [t, x], [u(t, x)]
    )
    prob_ode = discretize(ode_only, disc)
    @test !haskey(prob_ode.kwargs, :initializealg)

    prob_check = discretize(pdesys, disc; initializealg = CheckInit())
    @test prob_check.kwargs[:initializealg] isa CheckInit
    @test_throws Exception solve(prob_check, FBDF())
end
