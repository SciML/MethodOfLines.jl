using ModelingToolkit, MethodOfLines, LinearAlgebra, Test, OrdinaryDiffEq, DomainSets
using ModelingToolkit: Differential

@testset "Test 00: Dtt(u) + Dtx(u(t,x)) - Dxx(u(t,x)) ~ 0" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dtt = Differential(t)^2
    Dxx = Differential(x)^2
    Dtx = Differential(t)*Differential(x)

    asf(t, x) = sinpi(2*(t + (1+sqrt(5))*x/2)) + cospi(2*(t + (1-sqrt(5))*x/2))
    aDtf(t, x) = 2pi*cospi(2*(t + (1+sqrt(5))*x/2)) - 2pi*sinpi(2*(t + (1-sqrt(5))*x/2))
    # Where asf(t, x) ~ 0, NonlinearSolved
    xmin = -0.1118033987645
    xmax = 0.33541019624

    eq  = [Dtt(u(t, x)) + Dtx(u(t, x)) - Dxx(u(t, x)) ~ 0]

    bcs = [u(0, x) ~ asf(0, x),
           Dt(u(0, x)) ~ aDtf(0, x),
           u(t, xmin) ~ 0,
           u(t, xmax) ~ 0]


    domain = [t ∈ Interval(0.0, 1.0),
              x ∈ Interval(xmin, xmax)]

    @named pdesys = PDESystem(eq, bcs, domain, [t, x], [u(t,x)])

    dx = (xmax-xmin)/80
    discretization = MOLFiniteDifference([x => dx], t, advection_scheme = WENOScheme())

    prob = discretize(pdesys, discretization)
    sol = solve(prob, Tsit5())

    xdisc = sol[x]
    tdisc = sol[t]
    usol = sol[u(t,x)]

    asol = [asf(t, x) for t in tdisc, x in xdisc]
    @test usol ≈ asol atol = 1e-3
end
