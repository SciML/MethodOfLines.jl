@test_broken begin #@testset "Wave Equation" begin
    @parameters x t
    @variables u(..)

    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2
    Dtt = Differential(t)^2

    eq = Dtt(u(t, x)) ~ Dxx(u(t, x))

    bcs = [u(0, x) ~ sin(2pi*x),
        u(t, 0) ~ u(t, 1)]

    domains = [t ∈ IntervalDomain(0.0, 1.0),
        x ∈ IntervalDomain(0.0, 1.0)]

    @named pdesys = PDESystem([eq], bcs, domains, [t, x], [u(t, x)])

    discretization = MOLFiniteDifference([x => 0.1], t, approx_order=2)

    prob = discretize(pdesys, discretization)

    sol = solve(prob, Tsit5())

    @test sol[u(t, x)][end, :] ≈ sin.(2pi.*sol[x]) atol = 1e-3

end
