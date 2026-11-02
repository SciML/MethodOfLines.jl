using MethodOfLines, ModelingToolkit, DomainSets, OrdinaryDiffEq, Test
using MethodOfLines: PDEBase

@parameters t x
dx = 0.1

@testset "supported_conditions declarations" begin
    fd = PDEBase.supported_conditions(MOLFiniteDifference([x => dx], t))
    ps = PDEBase.supported_conditions(PseudospectralDiscretization([x => 8], t))
    for sc in (fd, ps)
        @test sc isa PDEBase.SupportedConditions
        @test !isempty(sc)
        @test sc.dirichlet === true
        @test sc.neumann === true
        @test sc.robin === true
        @test sc.periodic === true
        @test sc.time_dependent === true
        @test :interval in sc.geometries
        @test :box in sc.geometries
    end
    @test fd.interface === true
    @test ps.interface === false

    for sc in (fd, ps)
        markdown = sprint(show, MIME"text/markdown"(), sc)
        @test occursin("| Condition | Supported |", markdown)
        @test occursin("| Dirichlet | yes |", markdown)
        @test occursin("| Neumann | yes |", markdown)
    end
    @test occursin("| Interface | yes |", sprint(show, MIME"text/markdown"(), fd))
    @test occursin("| Interface | no |", sprint(show, MIME"text/markdown"(), ps))
end

@testset "declared support matches behaviour" begin
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    eq = Dt(u(t, x)) ~ Dx(Dx(u(t, x)))
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)]
    # Dirichlet, time-dependent Dirichlet, Neumann and Robin on the two ends
    for bcs in (
            [u(0, x) ~ 1.0, u(t, 0) ~ 1.0 + t, Dx(u(t, 1)) ~ 0.0],
            [u(0, x) ~ 1.0, Dx(u(t, 0)) + u(t, 0) ~ 1.0, u(t, 1) ~ 1.0],
        )
        @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
        prob = discretize(pdesys, MOLFiniteDifference([x => dx], t))
        @test SciMLBase.successful_retcode(solve(prob))
        prob = discretize(pdesys, PseudospectralDiscretization([x => 8], t))
        @test SciMLBase.successful_retcode(solve(prob))
    end
end
