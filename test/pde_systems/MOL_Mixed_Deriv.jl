using ModelingToolkit, MethodOfLines, LinearAlgebra, Test, OrdinaryDiffEq, DomainSets
using ModelingToolkit: Differential

# Broken in MTK
@test_broken begin #@testset "Test 00: Dtt(u) + Dtx(u(t,x)) - Dxx(u(t,x)) ~ Dxx(x)" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dtt = Differential(t)^2
    Dxx = Differential(x)^2
    Dtx = Differential(t)*Differential(x)

    assf(t, x) = sinpi(2*(t + (1+sqrt(5))*x/2)) + cospi(2*(t + (1-sqrt(5))*x/2))
    aDtf(t, x) = 2pi*cospi(2*(t + (1+sqrt(5))*x/2)) - 2pi*sinpi(2*(t + (1-sqrt(5))*x/2))
    # Where asf(t, x) ~ 0, NonlinearSolved
    xmin = -0.1118033987645
    xmax = 0.33541019624

    eq  = [Dtt(u(t, x)) + Dtx(u(t, x)) - Dxx(u(t, x)) ~ Dxx(x)]

    bcs = [u(1e-9, x) ~ assf(1e-9, x),
           Dt(u(1e-9, x)) ~ aDtf(1e-9, x),
           u(t, xmin) ~ 0,
           u(t, xmax) ~ 0]


    domain = [t ∈ Interval(1e-9, 1.0),
              x ∈ Interval(xmin, xmax)]

    @named pdesys = PDESystem(eq, bcs, domain, [t, x], [u(t,x)])

    dx = (xmax-xmin)/20
    discretization = MOLFiniteDifference([x => dx], t, advection_scheme = WENOScheme())

    prob = discretize(pdesys, discretization)
    sol = solve(prob, FBDF())

    xdisc = sol[x]
    tdisc = sol[t]
    usol = sol[u(t,x)]

    asol = [assf(t, x) for t in tdisc, x in xdisc]
    @test_broken usol ≈ asol atol = 1e-3
end

@testset "Test 01: Dt(u) ~ Dxy(u)" begin
    @parameters t x y
    @variables u(..)
    Dt = Differential(t)
    Dxy = Differential(x)*Differential(y)

    eq  = [Dt(u(t, x, y)) ~ Dxy(u(t, x, y))]

    bcs = [u(0, x, y) ~ sinpi(x + y),
           u(t, 0, y) ~ u(t, 1, y),
           u(t, x, 0) ~ u(t, x, 1)]

    domain = [t ∈ Interval(0.0, 1.0),
              x ∈ Interval(0.0, 1.0),
              y ∈ Interval(0.0, 1.0)]

    @named pdesys = PDESystem(eq, bcs, domain, [t, x, y], [u(t,x,y)])

    dx = 0.1
    dy = 0.1
    discretization = MOLFiniteDifference([x => dx, y => dy], t)

    prob = discretize(pdesys, discretization, advection_scheme = WENOScheme())
    sol = solve(prob, FBDF(), saveat = 0.1);
    @test sol.retcode == :Success
end
