using ModelingToolkit, MethodOfLines, DomainSets, Test
using ModelingToolkit: t_nounits as t, D_nounits as Dt

@testset "Product of first derivatives under UpwindScheme throws a clear ArgumentError" begin
    @parameters x y
    @variables ω(..) ψ(..)
    Dx, Dy = Differential(x), Differential(y)
    W, Ψ = ω(t, x, y), ψ(t, x, y)

    eqs = [
        Dt(W) ~ Dy(Ψ) * Dx(W) - Dx(Ψ) * Dy(W),
        Dx(Dx(Ψ)) + Dy(Dy(Ψ)) ~ -W,
    ]
    bcs = [
        ω(0, x, y) ~ sin(pi * x) * sin(pi * y), ψ(0, x, y) ~ 0,
        ω(t, 0, y) ~ 0, ω(t, 1, y) ~ 0, ω(t, x, 0) ~ 0, ω(t, x, 1) ~ 0,
        ψ(t, 0, y) ~ 0, ψ(t, 1, y) ~ 0, ψ(t, x, 0) ~ 0, ψ(t, x, 1) ~ 0,
    ]
    domains = [t ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]
    @named pde = PDESystem(eqs, bcs, domains, [t, x, y], [W, Ψ])

    err = try
        discretize(pde, MOLFiniteDifference([x => 9, y => 9], t))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("WENOScheme()", err.msg)

    disc = MOLFiniteDifference([x => 9, y => 9], t; advection_scheme = WENOScheme())
    @test discretize(pde, disc) !== nothing
end
