using ModelingToolkit, MethodOfLines, DomainSets, Test

@testset "Missing dependent variable throws ArgumentError" begin
    @parameters t, v
    @variables p(..) S(..)

    Dt = Differential(t)
    Dv = Differential(v)

    eq = [
        S(t, v) ~ -p(t, v) - Dv(p(t, v)),
        Dt(p(t, v)) ~ -Dv(S(t, v)),
    ]
    bcs = [p(0.0, v) ~ 1.0, p(t, 20.0) ~ 0.0, Dv(p(t, -10.0)) ~ 0.0]
    domains = [t ∈ Interval(0.0, 1.0), v ∈ Interval(-10.0, 20.0)]
    disc = MOLFiniteDifference([v => 5.0], t)

    @named sys = PDESystem(eq, bcs, domains, [t, v], [p(t, v)])
    term = Dv(S(t, v))
    badterm = S(t, v)
    @test_throws ArgumentError(
        "Could not expand derivatives in $term. If $badterm is a PDE unknown, add it to the PDESystem dependent-variable list."
    ) discretize(sys, disc)
end
