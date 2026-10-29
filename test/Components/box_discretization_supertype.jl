using MethodOfLines, ModelingToolkit, DomainSets, OrdinaryDiffEq, Test
using MethodOfLines: PDEBase

@parameters t x
dx = 0.1

@testset "Finite difference and pseudospectral discretizations are box discretizations" begin
    @test MOLFiniteDifference([x => dx], t) isa PDEBase.AbstractBoxDiscretization
    @test PseudospectralDiscretization([x => 8], t) isa PDEBase.AbstractBoxDiscretization
    @test !(MOLFiniteDifference([x => dx], t) isa PDEBase.AbstractMeshDiscretization)
end

# M0 exit (FEM spec section 9): a Rectangle problem with trace conditions, lowered by
# `PDEBase.to_box_bcs` and solved by MOLFiniteDifference.
@testset "to_box_bcs lowering of Rectangle trace conditions" begin
    @parameters y k α
    @variables u(..)
    Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)
    exact(t, x, y) = exp(-t) * (x + y^2)
    forcing = -exp(-t) * (x + y^2) - 2 * exp(-t)
    rect = Rectangle(0.0 .. 1.0, 0.0 .. 1.0)
    Γ_left, Γ_right, Γ_bottom, Γ_top = (PDEBase.boundarypiece(rect, n) for n in (:left, :right, :bottom, :top))
    Dn_right, Dn_bottom, Dn_top = PDEBase.NormalDerivative.((Γ_right, Γ_bottom, Γ_top))

    heat = Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y))) + forcing
    initial = u(0, x, y) ~ x + y^2
    trace_bcs = [
        u(t, Γ_left) ~ exp(-t) * y^2,
        Dn_right(u(t, Γ_right)) ~ exp(-t),
        k * Dn_bottom(u(t, Γ_bottom)) + α * u(t, Γ_bottom) ~ α * exp(-t) * x,
        Dn_top(u(t, Γ_top)) ~ 2 * exp(-t),
    ]
    endpoint_bcs = [
        u(t, 0, y) ~ exp(-t) * y^2,
        Dx(u(t, 1, y)) ~ exp(-t),
        -k * Dy(u(t, x, 0)) + α * u(t, x, 0) ~ α * exp(-t) * x,
        Dy(u(t, x, 1)) ~ 2 * exp(-t),
    ]
    system(bcs) = PDESystem(
        [heat], [initial; bcs], [(x, y) ∈ rect, t ∈ Interval(0.0, 1.0)], [t, x, y],
        [u(t, x, y)], [k => 2.0, α => 5.0]; name = :rectangle
    )
    trace_system, endpoint_system = system(trace_bcs), system(endpoint_bcs)

    @test all(isequal.(PDEBase.to_box_bcs(trace_system).bcs, endpoint_system.bcs))

    # No explicit lowering below: `symbolic_discretize` applies it to a box discretization.
    n = 6
    discretization = MOLFiniteDifference([x => 1 / (n - 1), y => 1 / (n - 1)], t)
    solve_system(sys) = solve(discretize(sys, discretization); saveat = 0.5, abstol = 1.0e-10, reltol = 1.0e-10)
    sol_trace, sol_endpoint = solve_system(trace_system), solve_system(endpoint_system)
    @test SciMLBase.successful_retcode(sol_trace)
    @test sol_trace[u(t, x, y)] == sol_endpoint[u(t, x, y)]

    # The corner nodes shared by two faces are not part of the solution; every other node is exact
    # up to the solver tolerance.
    xs, ys, ts = sol_trace[x], sol_trace[y], sol_trace.t
    U = sol_trace[u(t, x, y)]
    for i in eachindex(ts), j in eachindex(xs), l in eachindex(ys)
        (j in (1, n) && l in (1, n)) && continue
        @test U[i, j, l] ≈ exact(ts[i], xs[j], ys[l]) atol = 1.0e-8
    end
end
