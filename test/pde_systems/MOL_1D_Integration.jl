using MethodOfLines, ModelingToolkit, LinearAlgebra, Test, OrdinaryDiffEq, DomainSets

@testset "Test 00: Test simple integration case (0 .. x), no transformation" begin
    # test integrals
    @parameters t, x
    @variables integrand(..) cumuSum(..)
    Dt = Differential(t)
    Dx = Differential(x)
    xmin = 0.0
    xmax = 2.0 * pi

    Ix = Integral(x in DomainSets.ClosedInterval(xmin, x)) # basically cumulative sum from 0 to x

    eqs = [cumuSum(t, x) ~ Ix(integrand(t, x))
           integrand(t, x) ~ t * cos(x)]

    bcs = [cumuSum(0, x) ~ 0.0,
        integrand(0, x) ~ 0.0]

    domains = [t ∈ Interval(0.0, 1.0),
        x ∈ Interval(xmin, xmax)]

    @named pde_system = PDESystem(
        eqs, bcs, domains, [t, x], [integrand(t, x), cumuSum(t, x)])

    asf(t, x) = t * sin(x)

    disc = MOLFiniteDifference([x => 120], t)

    prob = discretize(pde_system, disc)

    sol = solve(prob, Tsit5())

    xdisc = sol[x]
    tdisc = sol[t]

    cumuSumsol = sol[cumuSum(t, x)]

    exact = [asf(t_, x_) for t_ in tdisc, x_ in xdisc]

    @test cumuSumsol≈exact atol=0.36
end

@testset "Test 00: Test simple integration case (0 .. x), with sys transformation" begin
    # test integrals
    @parameters t, x
    @variables integrand(..) cumuSum(..)
    Dt = Differential(t)
    Dx = Differential(x)
    xmin = 0.0
    xmax = 2.0 * pi

    Ix = Integral(x in DomainSets.ClosedInterval(xmin, x)) # basically cumulative sum from 0 to x

    eqs = [cumuSum(t, x) ~ Ix(t * cos(x))]

    bcs = [cumuSum(0, x) ~ 0.0]

    domains = [t ∈ Interval(0.0, 1.0),
        x ∈ Interval(xmin, xmax)]

    @named pde_system = PDESystem(
        eqs, bcs, domains, [t, x], [integrand(t, x), cumuSum(t, x)])

    asf(t, x) = t * sin(x)

    disc = MOLFiniteDifference([x => 120], t)

    @test_broken (discretize(pde_system, disc) isa ODEProblem)
    # prob = discretize(pde_system, disc)
    # sol = solve(prob, Tsit5())

    # xdisc = sol[x]
    # tdisc = sol[t]

    # cumuSumsol = sol[cumuSum(t, x)]

    # exact = [asf(t_, x_) for t_ in tdisc, x_ in xdisc]

    # @test cumuSumsol ≈ exact atol = 0.36
end

@testset "Test 01: Test integration over whole domain, (xmin .. xmax)" begin
    # test integrals
    @parameters t, x
    @variables integrand(..) cumuSum(..)
    Dt = Differential(t)
    Dx = Differential(x)
    xmin = 0.0
    xmax = 2.0 * pi

    Ix = Integral(x in DomainSets.ClosedInterval(xmin, xmax)) # integral over domain

    eqs = [cumuSum(t) ~ Ix(integrand(t, x))
           integrand(t, x) ~ t * cos(x)]

    bcs = [cumuSum(0) ~ 0.0,
        integrand(0, x) ~ 0.0]

    domains = [t ∈ Interval(0.0, 1.0),
        x ∈ Interval(xmin, xmax)]

    @named pde_system = PDESystem(eqs, bcs, domains, [t, x], [integrand(t, x), cumuSum(t)])

    asf(t) = 0.0

    disc = MOLFiniteDifference([x => 120], t)

    prob = discretize(pde_system, disc)

    sol = solve(prob, Tsit5())

    xdisc = sol[x]
    tdisc = sol[t]

    cumuSumsol = sol[cumuSum(t)]

    exact = [asf(t_) for t_ in tdisc]

    @test cumuSumsol≈exact atol=0.3
end

@testset "Test 02: Test integration with arbitrary limits, (a .. b)" begin
    # test integrals
    @parameters t, x
    @variables integrand(..) cumuSum(..)
    Dt = Differential(t)
    Dx = Differential(x)
    xmin = 0.0
    xmax = 2.0 * pi

    Ix = Integral(x in DomainSets.ClosedInterval(0.5, 3.0)) # integral over interval

    eqs = [cumuSum(t) ~ Ix(integrand(t, x))
           integrand(t, x) ~ t * cos(x)]

    bcs = [cumuSum(0) ~ 0.0,
        integrand(0, x) ~ 0.0]

    domains = [t ∈ Interval(0.0, 1.0),
        x ∈ Interval(xmin, xmax)]

    @named pde_system = PDESystem(eqs, bcs, domains, [t, x], [integrand(t, x), cumuSum(t)])

    asf(t, x) = t * sin(x)

    disc = MOLFiniteDifference([x => 120], t)

    @test_broken (discretize(pde_system, disc) isa ODEProblem)
    # prob = discretize(pde_system, disc)
    # sol = solve(prob, Tsit5(), saveat=0.1)

    # xdisc = sol[x]
    # tdisc = sol[t]

    # cumuSumsol = sol[cumuSum(t)]

    # exact = [asf(t_, 3.0) - asf(t_, 0.5) for t_ in tdisc]

    # @test cumuSumsol ≈ exact atol = 0.36
end
