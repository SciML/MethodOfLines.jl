using Test, DomainSets
using Test
using DomainSets
using ModelingToolkit
using Symbolics
using MethodOfLines
using OrdinaryDiffEq
using SciMLBase: DAEProblem, successful_retcode

@testset "HermiteScheme constructor and uniform discretization" begin
    @parameters t x
    @variables u(..) v(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    scheme = @inferred HermiteScheme(deriv_var = u => v)
    @test scheme.deriv_var[u] === v
    @test !scheme.automatic
    @test (@inferred HermiteScheme()).automatic
    @test @inferred(MethodOfLines._same_hermite_variable(u, u(t, x)))
    @test MethodOfLines.extent(scheme, 2) == 0

    eqs = [
        Dt(u(t, x)) ~ -v(t, x),
        Dt(v(t, x)) ~ -Dxx(u(t, x)),
    ]
    bcs = [
        u(0, x) ~ 0.0,
        v(0, x) ~ 0.0,
        u(t, -1.0) ~ 0.0,
        u(t, 1.0) ~ 0.0,
        v(t, -1.0) ~ 0.0,
        v(t, 1.0) ~ 0.0,
    ]
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(-1.0, 1.0)]
    @named pdesys = PDESystem(eqs, bcs, domains, [t, x], [u(t, x), v(t, x)])
    discretization = MOLFiniteDifference(
        [x => 0.25], t; approx_order = 2, advection_scheme = scheme
    )

    _, tspan = symbolic_discretize(pdesys, discretization)
    @test tspan == (0.0, 0.1)
end

@testset "HermiteScheme nonuniform grid and boundary rules" begin
    @parameters t x
    @variables u(..) v(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Dx^2

    eqs = [
        Dt(u(t, x)) ~ Dxx(u(t, x)),
        Dt(v(t, x)) ~ Dxx(v(t, x)),
    ]
    bcs = [
        u(0, x) ~ cos(x),
        v(0, x) ~ -sin(x),
        Dx(u(t, 0.0)) ~ 0.0,
        Dx(u(t, 2.0)) ~ -sin(2.0),
        Dx(v(t, 0.0)) ~ -1.0,
        Dx(v(t, 2.0)) ~ -cos(2.0),
    ]
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 2.0)]
    @named pdesys = PDESystem(eqs, bcs, domains, [t, x], [u(t, x), v(t, x)])
    grid = [
        0.0, 0.1, 0.2, 0.35, 0.55, 0.8,
        1.1, 1.4, 1.65, 1.8, 1.9, 2.0,
    ]
    discretization = MOLFiniteDifference(
        [x => grid], t; approx_order = 2,
        advection_scheme = HermiteScheme(deriv_var = u => v)
    )

    _, tspan = symbolic_discretize(pdesys, discretization)
    @test tspan == (0.0, 0.1)
end

@testset "HermiteScheme directional multidimensional mapping" begin
    @parameters t x y
    @variables u(..) vx(..) vy(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    eqs = [
        Dt(u(t, x, y)) ~ -vx(t, x, y) - vy(t, x, y),
        Dt(vx(t, x, y)) ~ -Dxx(u(t, x, y)),
        Dt(vy(t, x, y)) ~ -Dyy(u(t, x, y)),
    ]
    bcs = [
        u(0, x, y) ~ 0.0,
        vx(0, x, y) ~ 0.0,
        vy(0, x, y) ~ 0.0,
        u(t, 0.0, y) ~ 0.0,
        u(t, 1.0, y) ~ 0.0,
        u(t, x, 0.0) ~ 0.0,
        u(t, x, 1.0) ~ 0.0,
        vx(t, 0.0, y) ~ 0.0,
        vx(t, 1.0, y) ~ 0.0,
        vx(t, x, 0.0) ~ 0.0,
        vx(t, x, 1.0) ~ 0.0,
        vy(t, 0.0, y) ~ 0.0,
        vy(t, 1.0, y) ~ 0.0,
        vy(t, x, 0.0) ~ 0.0,
        vy(t, x, 1.0) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.1),
        x ∈ Interval(0.0, 1.0),
        y ∈ Interval(0.0, 1.0),
    ]
    @named pdesys = PDESystem(
        eqs, bcs, domains, [t, x, y],
        [u(t, x, y), vx(t, x, y), vy(t, x, y)]
    )
    discretization = MOLFiniteDifference(
        [x => 0.25, y => 0.25], t; approx_order = 2,
        advection_scheme = HermiteScheme(
            deriv_var = [(u, x) => vx, (u, y) => vy]
        )
    )

    _, tspan = symbolic_discretize(pdesys, discretization)
    @test tspan == (0.0, 0.1)

    ambiguous = MOLFiniteDifference(
        [x => 0.25, y => 0.25], t; approx_order = 2,
        advection_scheme = HermiteScheme(deriv_var = u => vx)
    )
    @test_throws ArgumentError symbolic_discretize(pdesys, ambiguous)
end


@testset "HermiteScheme automatic transformation for 1D advection" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    u_exact(x, t) = exp(-(0.3 * (x - t))^2)
    eq = Dt(u(t, x)) ~ -Dx(u(t, x))
    bcs = [
        u(0, x) ~ u_exact(x, 0.0),
        u(t, -2.0) ~ u_exact(-2.0, t),
        u(t, 3.0) ~ u_exact(3.0, t),
    ]
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(-2.0, 3.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
    discretization = MOLFiniteDifference(
        [x => 1.0], t; advection_scheme = HermiteScheme()
    )

    sys, tspan = symbolic_discretize(pdesys, discretization)
    discrete_equations = string.(ModelingToolkit.equations(sys))
    partner_equations = filter(discrete_equations) do discrete_equation
        occursin("Differential(t, 1)((uˍx(t))", discrete_equation)
    end

    @test tspan == (0.0, 0.1)
    @test !isempty(partner_equations)
    @test any(partner_equations) do partner_equation
        occursin("(u(t))", partner_equation) && occursin("(uˍx(t))", partner_equation)
    end
    @test !any(occursin("NaN", discrete_equation) for discrete_equation in discrete_equations)

    public_discretization = MOLFiniteDifference(
        [x => 1.0], t; advection_scheme = HermiteScheme()
    )
    prob = discretize(pdesys, public_discretization)
    @test prob isa DAEProblem
    sol = solve(prob; saveat = [0.1])
    @test successful_retcode(sol)
end

@testset "HermiteScheme automatic transformation for 2D advection" begin
    @parameters t x y
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dy = Differential(y)
    α = 4 / 5
    β = 3 / 5

    u_exact(x, y, t) = exp(-0.09 * ((x - α * t)^2 + (y - β * t)^2))
    eq = Dt(u(t, x, y)) ~ -α * Dx(u(t, x, y)) - β * Dy(u(t, x, y))
    bcs = [
        u(0, x, y) ~ u_exact(x, y, 0.0),
        u(t, -2.0, y) ~ u_exact(-2.0, y, t),
        u(t, 3.0, y) ~ u_exact(3.0, y, t),
        u(t, x, -2.0) ~ u_exact(x, -2.0, t),
        u(t, x, 3.0) ~ u_exact(x, 3.0, t),
    ]
    domains = [
        t ∈ Interval(0.0, 0.1),
        x ∈ Interval(-2.0, 3.0),
        y ∈ Interval(-2.0, 3.0),
    ]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])
    discretization = MOLFiniteDifference(
        [x => 1.0, y => 1.0], t; advection_scheme = HermiteScheme()
    )

    sys, tspan = symbolic_discretize(pdesys, discretization)
    discrete_equations = string.(ModelingToolkit.equations(sys))

    @test tspan == (0.0, 0.1)
    @test length(discretization.advection_scheme.deriv_var) == 2
    @test !any(occursin("NaN", equation) for equation in discrete_equations)
    @test !any(
        occursin("Differential(x", equation) || occursin("Differential(y", equation)
            for equation in discrete_equations
    )
end

@testset "HermiteScheme automatic periodic transformation" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    eq = Dt(u(t, x)) ~ -Dx(u(t, x))
    bcs = [u(0, x) ~ sinpi(2x), u(t, 0.0) ~ u(t, 1.0)]
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)]
    @named periodic_1d = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
    discretization_1d = MOLFiniteDifference(
        [x => 0.1], t; advection_scheme = HermiteScheme()
    )

    prob_1d = discretize(periodic_1d, discretization_1d)
    sol_1d = solve(prob_1d; saveat = [0.1])
    reference_1d = sinpi.(2 .* (sol_1d[x] .- 0.1))

    @test successful_retcode(sol_1d)
    @test maximum(abs.(sol_1d[u(t, x)][end, :] .- reference_1d)) < 2.0e-4
    @test length(discretization_1d.advection_scheme.deriv_var) == 1

    nonuniform_periodic = MOLFiniteDifference(
        [x => [0.0, 0.1, 0.3, 0.6, 1.0]], t;
        advection_scheme = HermiteScheme()
    )
    @test_throws ArgumentError symbolic_discretize(periodic_1d, nonuniform_periodic)

    @parameters τ ξ η
    @variables q(..)
    Dτ = Differential(τ)
    Dξ = Differential(ξ)
    Dη = Differential(η)

    initial_2d(ξ, η) = sinpi(2ξ) * sinpi(2η)
    eq_2d = Dτ(q(τ, ξ, η)) ~ -Dξ(q(τ, ξ, η)) - Dη(q(τ, ξ, η))
    bcs_2d = [
        q(0, ξ, η) ~ initial_2d(ξ, η),
        q(τ, 0.0, η) ~ q(τ, 1.0, η),
        q(τ, ξ, 0.0) ~ q(τ, ξ, 1.0),
    ]
    domains_2d = [
        τ ∈ Interval(0.0, 0.05),
        ξ ∈ Interval(0.0, 1.0),
        η ∈ Interval(0.0, 1.0),
    ]
    @named periodic_2d = PDESystem(
        eq_2d, bcs_2d, domains_2d, [τ, ξ, η], [q(τ, ξ, η)]
    )
    discretization_2d = MOLFiniteDifference(
        [ξ => 0.2, η => 0.2], τ; advection_scheme = HermiteScheme()
    )

    prob_2d = discretize(periodic_2d, discretization_2d)
    sol_2d = solve(prob_2d; saveat = [0.05])
    reference_2d = initial_2d.(
        reshape(sol_2d[ξ] .- 0.05, :, 1),
        reshape(sol_2d[η] .- 0.05, 1, :),
    )

    @test successful_retcode(sol_2d)
    numerical_interior = @view sol_2d[q(τ, ξ, η)][end, 2:(end - 1), 2:(end - 1)]
    reference_interior = @view reference_2d[2:(end - 1), 2:(end - 1)]
    @test maximum(abs.(numerical_interior .- reference_interior)) < 5.0e-2
    @test length(discretization_2d.advection_scheme.deriv_var) == 2
end

@testset "HermiteScheme automatic Neumann and Robin boundaries" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Dx^2
    xmax = Float64(π)
    tfinal = 0.05
    exact_1d(x, t) = exp(-t) * cos(x)
    eq = Dt(u(t, x)) ~ Dxx(u(t, x))
    domains = [t ∈ Interval(0.0, tfinal), x ∈ Interval(0.0, xmax)]

    boundary_sets = (
        neumann_robin = [
            u(0, x) ~ exact_1d(x, 0.0),
            Dx(u(t, 0.0)) ~ 0.0,
            u(t, xmax) + Dx(u(t, xmax)) ~ -exp(-t),
        ],
        robin_robin = [
            u(0, x) ~ exact_1d(x, 0.0),
            u(t, 0.0) + Dx(u(t, 0.0)) ~ exp(-t),
            2u(t, xmax) - Dx(u(t, xmax)) ~ -2exp(-t),
        ],
        dirichlet_neumann = [
            u(0, x) ~ exact_1d(x, 0.0),
            u(t, 0.0) ~ exp(-t),
            Dx(u(t, xmax)) ~ 0.0,
        ],
    )

    for (name, bcs) in pairs(boundary_sets)
        pdesys = PDESystem(
            eq, bcs, domains, [t, x], [u(t, x)]; name = Symbol(:hermite_, name)
        )
        discretization = MOLFiniteDifference(
            [x => xmax / 20], t; advection_scheme = HermiteScheme()
        )
        prob = discretize(pdesys, discretization)
        sol = solve(prob; saveat = [tfinal])
        reference = exact_1d.(sol[x], tfinal)

        @test successful_retcode(sol)
        @test maximum(abs.(sol[u(t, x)][end, :] .- reference)) < 2.0e-3
        @test length(discretization.advection_scheme.deriv_var) == 1
    end

    @parameters τ ξ η
    @variables q(..)
    Dτ = Differential(τ)
    Dξ = Differential(ξ)
    Dη = Differential(η)
    exact_2d(ξ, η, τ) = exp(-2τ) * cos(ξ) * cos(η)
    eq_2d = Dτ(q(τ, ξ, η)) ~
        (Dξ^2)(q(τ, ξ, η)) + (Dη^2)(q(τ, ξ, η))
    bcs_2d = [
        q(0, ξ, η) ~ exact_2d(ξ, η, 0.0),
        q(τ, 0.0, η) ~ exp(-2τ) * cos(η),
        Dξ(q(τ, xmax, η)) ~ 0.0,
        q(τ, ξ, 0.0) + Dη(q(τ, ξ, 0.0)) ~ exp(-2τ) * cos(ξ),
        q(τ, ξ, xmax) ~ -exp(-2τ) * cos(ξ),
    ]
    domains_2d = [
        τ ∈ Interval(0.0, tfinal),
        ξ ∈ Interval(0.0, xmax),
        η ∈ Interval(0.0, xmax),
    ]
    @named mixed_2d = PDESystem(
        eq_2d, bcs_2d, domains_2d, [τ, ξ, η], [q(τ, ξ, η)]
    )
    discretization_2d = MOLFiniteDifference(
        [ξ => xmax / 8, η => xmax / 8], τ;
        advection_scheme = HermiteScheme()
    )
    prob_2d = discretize(mixed_2d, discretization_2d)
    sol_2d = solve(prob_2d; saveat = [tfinal])
    numerical_interior = @view sol_2d[q(τ, ξ, η)][end, 2:(end - 1), 2:(end - 1)]
    reference_interior = exact_2d.(
        reshape(sol_2d[ξ][2:(end - 1)], :, 1),
        reshape(sol_2d[η][2:(end - 1)], 1, :),
        tfinal,
    )

    @test successful_retcode(sol_2d)
    @test maximum(abs.(numerical_interior .- reference_interior)) < 5.0e-2
    @test length(discretization_2d.advection_scheme.deriv_var) == 2
end

@testset "HermiteScheme automatic mapping is reusable" begin
    @parameters t x
    @variables u(..) q(..)
    Dt = Differential(t)
    Dx = Differential(x)
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(-1.0, 1.0)]

    system_u = PDESystem(
        Dt(u(t, x)) ~ -Dx(u(t, x)),
        [u(0, x) ~ exp(-x^2), u(t, -1.0) ~ exp(-(-1.0 - t)^2),
            u(t, 1.0) ~ exp(-(1.0 - t)^2)],
        domains, [t, x], [u(t, x)]; name = :reusable_u
    )
    system_q = PDESystem(
        Dt(q(t, x)) ~ -Dx(q(t, x)),
        [q(0, x) ~ exp(-x^2), q(t, -1.0) ~ exp(-(-1.0 - t)^2),
            q(t, 1.0) ~ exp(-(1.0 - t)^2)],
        domains, [t, x], [q(t, x)]; name = :reusable_q
    )
    scheme = HermiteScheme()
    discretization = MOLFiniteDifference([x => 0.5], t; advection_scheme = scheme)

    symbolic_discretize(system_u, discretization)
    @test length(scheme.deriv_var) == 1
    @test isequal(only(keys(scheme.deriv_var)), operation(Symbolics.value(u(t, x))))

    symbolic_discretize(system_q, discretization)
    @test length(scheme.deriv_var) == 1
    @test isequal(only(keys(scheme.deriv_var)), operation(Symbolics.value(q(t, x))))
end
