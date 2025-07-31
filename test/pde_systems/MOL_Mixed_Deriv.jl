using ModelingToolkit, MethodOfLines, LinearAlgebra, Test, OrdinaryDiffEq, DomainSets,
      NonlinearSolve
using ModelingToolkit: Differential

# Broken in MTK
@testset "Test 00: Dtt(u) + Dtx(u(t,x)) - Dxx(u(t,x)) ~ Dxx(x)" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dtt = Differential(t)^2
    Dxx = Differential(x)^2
    Dtx = Differential(t) * Differential(x)

    assf(t, x) = sinpi(2 * (t + (1 + sqrt(5)) * x / 2)) +
                 cospi(2 * (t + (1 - sqrt(5)) * x / 2))
    aDtf(t, x) = 2pi * cospi(2 * (t + (1 + sqrt(5)) * x / 2)) -
                 2pi * sinpi(2 * (t + (1 - sqrt(5)) * x / 2))
    # Where asf(t, x) ~ 0, NonlinearSolved
    xmin = -0.1118033987645
    xmax = 0.33541019624

    eq = [Dtt(u(t, x)) + Dtx(u(t, x)) - Dxx(u(t, x)) ~ Dxx(x)]

    bcs = [u(1e-9, x) ~ assf(1e-9, x),
        Dt(u(1e-9, x)) ~ aDtf(1e-9, x),
        u(t, xmin) ~ 0,
        u(t, xmax) ~ 0]

    domain = [t ∈ Interval(1e-9, 1.0),
        x ∈ Interval(xmin, xmax)]

    @named pdesys = PDESystem(eq, bcs, domain, [t, x], [u(t, x)])

    dx = (xmax - xmin) / 20
    discretization = MOLFiniteDifference([x => dx], t, advection_scheme = WENOScheme())

    prob = discretize(pdesys, discretization)
    sol = solve(prob, FBDF())

    xdisc = sol[x]
    tdisc = sol[t]
    usol = sol[u(t, x)]

    asol = [assf(t, x) for t in tdisc, x in xdisc]
    @test_broken usol ≈ asol atol = 1e-3
end

@testset "Test 01: Dt(u) ~ Dxy(u)" begin
    @parameters t x y
    @variables u(..)
    Dt = Differential(t)
    Dxxy = Differential(x)^2 * Differential(y)

    eq = [Dt(u(t, x, y)) ~ Dxxy(u(t, x, y))]

    bcs = [u(0, x, y) ~ sinpi(x + y),
        u(t, 0, y) ~ sinpi(y),
        u(t, x, 0) ~ sinpi(x)]

    domain = [t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, 1.0),
        y ∈ Interval(0.0, 1.0)]

    @named pdesys = PDESystem(eq, bcs, domain, [t, x, y], [u(t, x, y)])

    dx = 0.1
    dy = 0.1
    discretization = MOLFiniteDifference([x => dx, y => dy], t)

    prob = discretize(pdesys, discretization, advection_scheme = WENOScheme())
    sol = solve(prob, FBDF(), saveat = 0.1)
    @test_broken SciMLBase.successful_retcode(sol)
end

@testset "Mixed steady state problem" begin
    @parameters x y
    @variables u(..)
    Dx = Differential(x)
    Dy = Differential(y)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    Dxy = Differential(x) * Differential(y)

    eq = [Dxx(u(x, y)) + Dyy(u(x, y)) + Dxy(u(x, y)) ~ 0]

    bcs = [u(0, y) ~ 0,
        #Dx(u(0, y)) ~ y,
        u(1, y) ~ y,
        #Dx(u(1, y)) ~ y,
        u(x, 0) ~ 0,
        #Dy(u(x, 0)) ~ x,
        u(x, 1) ~ x        #Dy(u(x, 1)) ~ x
    ]

    domain = [x ∈ Interval(0.0, 1.0),
        y ∈ Interval(0.0, 1.0)]

    analytic_u(x, y) = x * y

    @named pdesys = PDESystem(eq, bcs, domain, [x, y], [u(x, y)])

    dx = 0.1
    dy = 0.08

    disc = MOLFiniteDifference([x => 20, y => 20], order = 4)

    prob = discretize(pdesys, disc)

    sol = solve(prob, NewtonRaphson())

    @test_broken SciMLBase.successful_retcode(sol)

    solu = sol[u(x, y)]
    solx = sol[x]
    soly = sol[y]

    asol = [analytic_u(x, y) for x in solx[1:(end - 1)], y in soly[1:(end - 1)]]

    @test_broken solu[1:(end - 1), 1:(end - 1)]≈asol atol=1e-3
end

@testset "Wave Equation u_tt ~ u_xx" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dtt = Differential(t)^2
    Dxx = Differential(x)^2

    eq = [Dtt(u(t, x)) ~ Dxx(u(t, x))]

    bcs = [u(0, x) ~ sinpi(x),
        Dt(u(0, x)) ~ 0,
        u(t, 0) ~ 0,
        u(t, 1) ~ 0]

    domain = [t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, 1.0)]

    analytic_u(t, x) = sinpi(x) * cospi(t)

    @named pdesys = PDESystem(eq, bcs, domain, [t, x], [u(t, x)])

    dx = 0.01

    disc = MOLFiniteDifference([x => dx], t)

    prob = discretize(pdesys, disc)

    sol = solve(prob, FBDF(), saveat = 0.1)

    xdisc = sol[x]
    tdisc = sol[t]
    usol = sol[u(t, x)]

    asol = [analytic_u(t, x) for t in tdisc, x in xdisc]

    @test usol≈asol atol=1e-2
end
