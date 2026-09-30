using MethodOfLines, ModelingToolkitBase, SciMLBase, Test

# Pure reaction PDE: no spatial derivatives / BCs, so stencil extents are zero.
# Boundary extrapolation must not evaluate the 6-point stencil on a 4-point grid.
@testset "Zero-extent BC extrapolation on small grids" begin
    @independent_variables t x
    @variables u(..)
    D = Differential(t)
    @named pde = PDESystem(
        [D(u(t, x)) ~ -u(t, x)], [u(0, x) ~ 1 + x],
        [t ∈ (0.0, 0.2), x ∈ (0.0, 1.0)], [t, x], [u(t, x)]
    )
    for intervals in (5, 3)
        sys, _ = SciMLBase.symbolic_discretize(
            pde, MOLFiniteDifference([x => 1 / intervals], t)
        )
        @test !isnothing(sys)
    end
end
