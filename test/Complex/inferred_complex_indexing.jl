using MethodOfLines, OrdinaryDiffEq, DomainSets, ModelingToolkit, Test, SciMLBase
using DiffEqBase: BrownFullBasicInit

@testset "Inferred-complex linear Schrödinger solution indexing" begin
    @parameters t x
    @variables ψ(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    h = 0.1
    A = 0.5
    k = 2.0
    T = 0.05
    ω = 4sin(k * h / 2)^2 / h^2
    domains = [t ∈ Interval(0.0, T), x ∈ Interval(0.0, 1.0)]
    bcs = [
        ψ(0, x) => A * exp(im * k * x),
        ψ(t, 0) ~ A * exp(-im * ω * t),
        ψ(t, 1) ~ A * exp(im * (k - ω * t)),
    ]
    sys = PDESystem(
        [im * Dt(ψ(t, x)) ~ -Dxx(ψ(t, x))], bcs, domains, [t, x], [ψ(t, x)];
        name = :inferred_complex
    )
    sol = solve(
        discretize(sys, MOLFiniteDifference([x => 11], t)); reltol = 1.0e-11,
        abstol = 1.0e-11, saveat = T / 2, initializealg = BrownFullBasicInit()
    )
    exact = A .* exp.(im .* (k .* sol[x] .- ω * T))

    @test SciMLBase.successful_retcode(sol)
    @test maximum(abs.(sol(T, sol[x]; dv = ψ(t, x)) .- exact)) < 1.0e-10
end
