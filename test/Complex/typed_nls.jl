using MethodOfLines, OrdinaryDiffEq, DomainSets, ModelingToolkit, Test, SciMLBase
using DiffEqBase: BrownFullBasicInit

@testset "Complex-typed cubic NLS plane wave" begin
    @parameters t x
    @variables ψ(..)::Complex
    Dt = Differential(t)
    Dxx = Differential(x)^2
    h = 0.1
    A = 0.5
    k = 2.0
    T = 0.05
    ω = 4sin(k * h / 2)^2 / h^2 + A^2
    z = ψ(t, x)
    domains = [t ∈ Interval(0.0, T), x ∈ Interval(0.0, 1.0)]
    bcs = [
        ψ(0, x) => A * exp(im * k * x),
        ψ(t, 0) ~ A * exp(-im * ω * t),
        ψ(t, 1) ~ A * exp(im * (k - ω * t)),
    ]
    disc = MOLFiniteDifference([x => 11], t)

    for nonlinearity in (abs2(z) * z, z * conj(z) * z)
        sys = PDESystem(
            [im * Dt(z) ~ -Dxx(z) + nonlinearity], bcs, domains, [t, x], [z];
            name = :typed_nls
        )
        prob = discretize(sys, disc)
        sol = solve(
            prob; reltol = 1.0e-11, abstol = 1.0e-11, saveat = T / 2,
            initializealg = BrownFullBasicInit()
        )
        exact = A .* exp.(im .* (k .* sol[x] .- ω * T))
        @test SciMLBase.successful_retcode(sol)
        @test maximum(abs.(sol[ψ(t, x)][end, :] .- exact)) < 1.0e-10
        @test maximum(abs.(sol(T, sol[x]; dv = z) .- exact)) < 1.0e-10
        # Transformed Complex{Num} keys must not silently alias the field.
        @test_throws ErrorException sol[conj(z)]
        @test_throws ErrorException sol[real(z) - 2im * imag(z)]
        @test_throws ErrorException sol(T, sol[x]; dv = conj(z))
        @test_throws ErrorException sol(T, sol[x]; dv = real(z) - 2im * imag(z))
    end
end
