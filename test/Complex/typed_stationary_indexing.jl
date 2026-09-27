using MethodOfLines, ModelingToolkit, DomainSets, NonlinearSolve, Test

@testset "Complex-typed stationary solution indexing" begin
    @parameters x
    @variables u(..)::Complex
    Dxx = Differential(x)^2
    z = u(x)
    sys = PDESystem(
        [Dxx(z) ~ 0], [u(0) ~ 1 + 2im, u(1) ~ 3 - im],
        [x ∈ Interval(0.0, 1.0)], [x], [z]; name = :typed_stationary
    )
    sol = solve(discretize(sys, MOLFiniteDifference([x => 11], nothing)))
    exact(xv) = 1 + 2im + (2 - 3im) * xv

    @test maximum(abs.(sol[z] .- exact.(sol[x]))) < 1.0e-10
    @test abs(sol(0.35; dv = z) - exact(0.35)) < 1.0e-10
    # Transformed Complex{Num} keys must not silently alias the field.
    @test_throws ErrorException sol[conj(z)]
    @test_throws ErrorException sol[real(z) - 2im * imag(z)]
    @test_throws ErrorException sol(0.35; dv = conj(z))
    @test_throws ErrorException sol(0.35; dv = real(z) - 2im * imag(z))
end
