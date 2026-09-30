using MethodOfLines, ModelingToolkitBase, SciMLBase, OrdinaryDiffEq, Test

# Pure reaction PDE: no spatial derivatives / BCs, so stencil extents are zero.
# Boundary extrapolation must not evaluate the 6-point stencil on a 4-point grid.
# Exact solution is (1+x)*exp(-t); with no spatial truncation the residual is solver tol.
@testset "Zero-extent BC extrapolation on small grids" begin
    @independent_variables t x
    @variables u(..)
    D = Differential(t)
    @named pde = PDESystem(
        [D(u(t, x)) ~ -u(t, x)], [u(0, x) ~ 1 + x],
        [t ∈ (0.0, 0.2), x ∈ (0.0, 1.0)], [t, x], [u(t, x)]
    )
    t_end = 0.2
    for intervals in (5, 3)
        disc = MOLFiniteDifference([x => 1 / intervals], t)
        prob = discretize(pde, disc)
        @test prob isa SciMLBase.DAEProblem
        @test length(prob.u0) == intervals + 1
        sol = solve(prob; abstol = 1.0e-10, reltol = 1.0e-10)
        @test SciMLBase.successful_retcode(sol)
        xs = range(0.0, 1.0; length = intervals + 1)
        u_num = sol[u(t, x)][end, :]
        exact = @. (1 + xs) * exp(-t_end)
        # Measured max |error| ~1e-11 at these solver tols (no spatial truncation).
        @test maximum(abs, u_num .- exact) < 1.0e-9
    end
end
