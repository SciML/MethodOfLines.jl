using MethodOfLines, ModelingToolkit, DomainSets, OrdinaryDiffEq, Test
import PDEBase

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
    # u_t = u_xx - 1 has the exact solution 1 + t + x + x^2, which both spatial schemes
    # differentiate exactly, so wrong boundary data cannot pass.
    eq = Dt(u(t, x)) ~ Dx(Dx(u(t, x))) - 1
    tend = 0.1
    domains = [t ∈ Interval(0.0, tend), x ∈ Interval(0.0, 1.0)]
    exact(t, x) = 1 + t + x + x^2
    # Time-dependent Dirichlet on the left, then Neumann and Robin on the right
    for bcs in (
            [u(0, x) ~ exact(0, x), u(t, 0) ~ 1 + t, Dx(u(t, 1)) ~ 3],
            [u(0, x) ~ exact(0, x), u(t, 0) ~ 1 + t, Dx(u(t, 1)) + u(t, 1) ~ 6 + t],
        )
        @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
        for disc in (
                MOLFiniteDifference([x => dx], t),
                PseudospectralDiscretization([x => 8], t),
            )
            sol = solve(discretize(pdesys, disc); reltol = 1.0e-10, abstol = 1.0e-10)
            @test SciMLBase.successful_retcode(sol)
            @test maximum(abs.(sol[u(t, x)][end, :] .- exact.(tend, sol[x]))) < 1.0e-8
        end
    end
end

@testset "documentation tables render" begin
    # Evaluate the `@eval` blocks of the docs page itself, each in its own module
    docs = read(joinpath(@__DIR__, "..", "..", "docs", "src", "boundary_conditions.md"), String)
    blocks = [m.captures[1] for m in eachmatch(r"```@eval\n(.*?)```"s, docs)]
    function interface_row(block)
        md = Core.eval(Module(), Meta.parseall(block))
        html = sprint(show, MIME"text/html"(), md)
        return only(m.match for m in eachmatch(r"<tr>(?:(?!</tr>).)*Interface(?:(?!</tr>).)*</tr>"s, html))
    end
    fd_blocks = filter(b -> occursin("MOLFiniteDifference(", b), blocks)
    ps_blocks = filter(b -> occursin("PseudospectralDiscretization(", b), blocks)
    @test length(blocks) == 2
    @test length(fd_blocks) == 1
    @test length(ps_blocks) == 1
    @test occursin("yes", interface_row(only(fd_blocks)))
    @test occursin("no", interface_row(only(ps_blocks)))
end
