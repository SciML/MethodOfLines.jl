using ModelingToolkit, MethodOfLines, DomainSets, OrdinaryDiffEq, SciMLSensitivity, Zygote,
    ForwardDiff, Test
using DifferentiationInterface: DifferentiationInterface, AutoMooncake
using Mooncake: Mooncake
using NonlinearSolve: NewtonRaphson
import SciMLStructures as SS
using SymbolicIndexingInterface: setp_oop

# Reverse-mode gradients through `solve` and the solution interface, against ForwardDiff.
@parameters t x α β
@variables u(..) w(..)
Dt = Differential(t)
Dxx = Differential(x)^2
domains = [t ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0)]
disc = MOLFiniteDifference([x => 0.1], t)
# The symbols are built once, outside the differentiated functions, as constants for Mooncake.
const U = u(t, x)
const W = w(x)
p0 = [1.2, 2.1]
tol = (abstol = 1.0e-8, reltol = 1.0e-8)

bcs = [u(0, x) ~ cos(x), u(t, 0) ~ exp(-t), u(t, 1) ~ exp(-t) * cos(1)]
@named heat = PDESystem(
    Dt(U) ~ (α + β) * Dxx(U), bcs, domains, [t, x], [U], [α => 1.2, β => 2.1]
)
compiled = ODEProblem(mtkcompile(symbolic_discretize(heat, disc)[1]), nothing)

function solver(prob)
    set_p = setp_oop(prob, [α, β])
    return (ps; kwargs...) -> solve(
        remake(prob; p = set_p(prob, ps)); saveat = 0.1, tol..., kwargs...
    )
end

@testset "$path" for (path, prob) in (
        ("DAE path", discretize(heat, disc)), ("compiled ODE path", compiled),
    )
    solve_at = solver(prob)
    data = solve_at(p0)[U]

    loss(ps) = sum(abs2, solve_at(ps)[U] .- 0.5 .* data)
    g_forward = ForwardDiff.gradient(loss, p0)
    @test Zygote.gradient(loss, p0)[1] ≈ g_forward rtol = 1.0e-4

    function adjoint_loss(ps)
        sol = solve_at(ps; sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)))
        return sum(abs2, sol[U] .- 0.5 .* data)
    end
    @test Zygote.gradient(adjoint_loss, p0)[1] ≈ g_forward rtol = 1.0e-4

    slice(ps) = sum(solve_at(ps)[U][end, :])
    @test Zygote.gradient(slice, p0)[1] ≈ ForwardDiff.gradient(slice, p0) rtol = 1.0e-4

    ranged(ps) = sum(solve_at(ps)[U, 2:4, :])
    @test Zygote.gradient(ranged, p0)[1] ≈ ForwardDiff.gradient(ranged, p0) rtol = 1.0e-4

    # A repeated index counts twice.
    repeated(ps) = sum(abs2, solve_at(ps)[U, [2, 2], :])
    @test Zygote.gradient(repeated, p0)[1] ≈ ForwardDiff.gradient(repeated, p0) rtol = 1.0e-4

    # The grid has no dependence on the solution.
    grid(ps) = sum(solve_at(ps)[x])
    @test Zygote.gradient(grid, p0)[1] === nothing

    # Every variable, interpolated on the grid in space.
    every(ps) = sum(sum, solve_at(ps)(0.53, :))
    @test Zygote.gradient(every, p0)[1] ≈ ForwardDiff.gradient(every, p0) rtol = 1.0e-4

    # Indexing and interpolation of one solution object in the same loss.
    mixed(ps) = (sol = solve_at(ps); sum(abs2, sol[U]) + sol(0.53, 0.37; dv = U))
    @test Zygote.gradient(mixed, p0)[1] ≈ ForwardDiff.gradient(mixed, p0) rtol = 1.0e-4
end

@testset "Mooncake, $path" for (path, prob) in (
        ("DAE path", discretize(heat, disc)), ("compiled ODE path", compiled),
    )
    # The same rules under Mooncake. The parameters are set through SciMLStructures, since a
    # parameter object from `setp_oop` needs SciML/ModelingToolkit.jl#5288 under Mooncake.
    tunables, repack, _ = SS.canonicalize(SS.Tunable(), prob.p)
    solve_at(ps) = solve(remake(prob; p = repack(ps)); saveat = 0.1, tol...)
    for loss in (
            ps -> sum(abs2, solve_at(ps)[U]),
            ps -> solve_at(ps)[U, 4, 3],
            ps -> solve_at(ps)(0.53, 0.37; dv = U),
        )
        @test DifferentiationInterface.gradient(loss, AutoMooncake(), tunables) ≈
            ForwardDiff.gradient(loss, tunables) rtol = 1.0e-4
    end
end

@testset "Mooncake, homogeneous boundaries" begin
    # The boundary points are observed as integer constants.
    bcs0 = [u(0, x) ~ sinpi(x), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0]
    @named heat0 = PDESystem(
        Dt(U) ~ (α + β) * Dxx(U), bcs0, domains, [t, x], [U], [α => 1.2, β => 2.1]
    )
    compiled0 = ODEProblem(mtkcompile(symbolic_discretize(heat0, disc)[1]), nothing)
    tunables0, repack0, _ = SS.canonicalize(SS.Tunable(), compiled0.p)
    energy(ps) = sum(abs2, solve(remake(compiled0; p = repack0(ps)); saveat = 0.1, tol...)[U])
    @test DifferentiationInterface.gradient(energy, AutoMooncake(), tunables0) ≈
        ForwardDiff.gradient(energy, tunables0) rtol = 1.0e-4
end

@testset "interpolation at points" begin
    solve_at = solver(discretize(heat, disc))

    point(ps) = solve_at(ps)(0.53, 0.37; dv = U)
    @test Zygote.gradient(point, p0)[1] ≈ ForwardDiff.gradient(point, p0) rtol = 1.0e-4

    line(ps) = sum(solve_at(ps)(0.53, 0.05:0.1:0.95; dv = U))
    @test Zygote.gradient(line, p0)[1] ≈ ForwardDiff.gradient(line, p0) rtol = 1.0e-4

    # The coordinates of an evaluation have gradients too, the slopes of the interpolation.
    sol = solve_at(p0)
    coordinate(ξ) = sol(0.53, ξ; dv = U)
    @test Zygote.gradient(coordinate, 0.37)[1] ≈ ForwardDiff.derivative(coordinate, 0.37)
    both(ps, τ, ξ) = solve_at(ps)(τ, ξ; dv = U)
    g = Zygote.gradient(both, p0, 0.53, 0.37)
    @test g[1] ≈ ForwardDiff.gradient(point, p0) rtol = 1.0e-4
    @test g[2] ≈ ForwardDiff.derivative(τ -> sol(τ, 0.37; dv = U), 0.53)
    @test g[3] ≈ ForwardDiff.derivative(coordinate, 0.37)
    points(ξs) = sum(abs2, sol(0.53, ξs; dv = U))
    @test Zygote.gradient(points, [0.37, 0.61])[1] ≈ ForwardDiff.gradient(points, [0.37, 0.61])
    every(ξ) = sum(sum, sol(0.53, ξ))
    @test Zygote.gradient(every, 0.37)[1] ≈ ForwardDiff.derivative(every, 0.37)
end

@testset "time as the last argument" begin
    # The field is stored with time last, and the arguments of `sol(x, t; dv)` follow it.
    V = u(x, t)
    bcs_V = [u(x, 0) ~ cos(x), u(0, t) ~ exp(-t), u(1, t) ~ exp(-t) * cos(1)]
    @named heat_V = PDESystem(
        Dt(V) ~ (α + β) * Dxx(V), bcs_V, domains, [t, x], [V], [α => 1.2, β => 2.1]
    )
    solve_at = solver(discretize(heat_V, disc))

    field(ps) = sum(abs2, solve_at(ps)[V])
    @test Zygote.gradient(field, p0)[1] ≈ ForwardDiff.gradient(field, p0) rtol = 1.0e-4

    point(ps) = solve_at(ps)(0.37, 0.53; dv = V)
    @test Zygote.gradient(point, p0)[1] ≈ ForwardDiff.gradient(point, p0) rtol = 1.0e-4

    line(ps) = sum(solve_at(ps)(:, 0.53; dv = V))
    @test Zygote.gradient(line, p0)[1] ≈ ForwardDiff.gradient(line, p0) rtol = 1.0e-4

    sol = solve_at(p0)
    coordinate(ξ) = sol(ξ, 0.53; dv = V)
    @test Zygote.gradient(coordinate, 0.37)[1] ≈ ForwardDiff.derivative(coordinate, 0.37)
    # Without `dv` the arguments follow `sol.ivs`, time first.
    every(ξ) = sum(sum, sol(0.53, ξ))
    @test Zygote.gradient(every, 0.37)[1] ≈ ForwardDiff.derivative(every, 0.37)
end

@testset "parameter in an observed boundary value" begin
    # On the compiled path the boundary points are observed, so part of α's gradient comes
    # from the boundary value directly, not through the states.
    bcs_α = [u(0, x) ~ cos(x), u(t, 0) ~ α * exp(-t), u(t, 1) ~ exp(-t) * cos(1)]
    @named heat_α = PDESystem(
        Dt(U) ~ (α + β) * Dxx(U), bcs_α, domains, [t, x], [U], [α => 1.2, β => 2.1]
    )
    prob = ODEProblem(mtkcompile(symbolic_discretize(heat_α, disc)[1]), nothing)
    solve_at = solver(prob)

    boundary(ps) = sum(solve_at(ps)[U])
    g_forward = ForwardDiff.gradient(boundary, p0)
    @test Zygote.gradient(boundary, p0)[1] ≈ g_forward rtol = 1.0e-4

    function adjoint_boundary(ps)
        sol = solve_at(ps; sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)))
        return sum(sol[U])
    end
    @test Zygote.gradient(adjoint_boundary, p0)[1] ≈ g_forward rtol = 1.0e-4
end

@testset "complex-valued field" begin
    # A complex-valued variable is reconstructed from its real and imaginary parts, which the
    # rules do not cover: an error, not a zero gradient. The same for a typed one.
    @parameters V0
    @variables ψ(..) ϕ(..)::Complex
    ψ0 = ((1 + im) / sqrt(2)) * sinpi(2x)
    systems = (
        PDESystem(
            [im * Dt(ψ(t, x)) ~ Dxx(ψ(t, x)) + V0 * ψ(t, x)],
            [ψ(0, x) => ψ0, ψ(t, 0) ~ 0, ψ(t, 1) ~ 0], domains, [t, x], [ψ(t, x)],
            [V0 => 0.5]; name = :inferred
        ) => ψ(t, x),
        PDESystem(
            [im * Dt(ϕ(t, x)) ~ Dxx(ϕ(t, x)) + V0 * ϕ(t, x)],
            [ϕ(0, x) => ψ0, ϕ(t, 0) ~ 0, ϕ(t, 1) ~ 0], domains, [t, x], [ϕ(t, x)],
            [V0 => 0.5]; name = :typed
        ) => ϕ(t, x),
    )
    for (sys, Ψ) in systems
        prob = discretize(sys, MOLFiniteDifference([x => 20], t))
        set_V = setp_oop(prob, [V0])
        solve_V(ps) = solve(remake(prob; p = set_V(prob, ps)); saveat = 0.1)
        energy(ps) = sum(abs2, solve_V(ps)[Ψ])
        @test_throws "differentiation of the solution is not implemented" Zygote.gradient(energy, [0.5])
        point(ps) = abs2(solve_V(ps)(0.53, 0.37; dv = Ψ))
        @test_throws "differentiation of the solution is not implemented" Zygote.gradient(point, [0.5])
    end
end

@testset "stationary" begin
    # The solver's solution is a `NonlinearSolution`. The parameter in the boundary condition
    # makes the eliminated boundary point an observed one that depends on it.
    @named poisson = PDESystem(
        [Dxx(W) ~ -α * sinpi(x) + β * W], [w(0) ~ 0.0, w(1) ~ β], [x ∈ Interval(0.0, 1.0)],
        [x], [W], [α => 1.2, β => 2.1]
    )
    prob = discretize(poisson, MOLFiniteDifference([x => 0.1], nothing))
    solve_at(p) = solve(remake(prob; p), NewtonRaphson(); abstol = 1.0e-12, reltol = 1.0e-12)
    set_p = setp_oop(prob, [α, β])
    for loss in (
            ps -> sum(abs2, solve_at(set_p(prob, ps))[W]),
            ps -> solve_at(set_p(prob, ps))[W, 11],
            ps -> solve_at(set_p(prob, ps))(0.37; dv = W),
        )
        @test Zygote.gradient(loss, p0)[1] ≈ ForwardDiff.gradient(loss, p0) rtol = 1.0e-6
    end
    tunables, repack, _ = SS.canonicalize(SS.Tunable(), prob.p)
    energy(ps) = sum(abs2, solve_at(repack(ps))[W])
    @test DifferentiationInterface.gradient(energy, AutoMooncake(), tunables) ≈
        ForwardDiff.gradient(energy, tunables) rtol = 1.0e-6
end
