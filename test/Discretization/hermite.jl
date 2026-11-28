using Test
using DomainSets
using ModelingToolkit
using Symbolics
using ForwardDiff
using MethodOfLines
using OrdinaryDiffEq
using SciMLBase: DAEProblem, successful_retcode

used_scheme(sys) = getmetadata(sys, ModelingToolkit.ProblemTypeCtx, nothing).disc.advection_scheme
used_scheme(pdesys, disc) = used_scheme(first(symbolic_discretize(pdesys, disc)))

@testset "HermiteScheme constructor" begin
    @parameters t x
    @variables u(..) v(..)

    scheme = @inferred HermiteScheme(deriv_var = u => v)
    @test scheme.deriv_var[u] === v
    @test !scheme.automatic
    @test (@inferred HermiteScheme()).automatic
    @test @inferred(MethodOfLines._same_hermite_variable(u, u(t, x)))
    @test MethodOfLines.extent(scheme, 2) == 0
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
        occursin(r"(?<![\wˍ])u\(t\)", partner_equation) && occursin("uˍx(t)", partner_equation)
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
    @test length(used_scheme(sys).deriv_var) == 2
    @test isempty(discretization.advection_scheme.deriv_var)
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
    @test length(used_scheme(periodic_1d, discretization_1d).deriv_var) == 1

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
    @test length(used_scheme(periodic_2d, discretization_2d).deriv_var) == 2
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
        @test length(used_scheme(pdesys, discretization).deriv_var) == 1
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
    @test length(used_scheme(mixed_2d, discretization_2d).deriv_var) == 2
end

@testset "HermiteScheme automatic mapping is reusable" begin
    @parameters t x
    @variables u(..) q(..)
    Dt = Differential(t)
    Dx = Differential(x)
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(-1.0, 1.0)]

    system_u = PDESystem(
        Dt(u(t, x)) ~ -Dx(u(t, x)),
        [
            u(0, x) ~ exp(-x^2), u(t, -1.0) ~ exp(-(-1.0 - t)^2),
            u(t, 1.0) ~ exp(-(1.0 - t)^2),
        ],
        domains, [t, x], [u(t, x)]; name = :reusable_u
    )
    system_q = PDESystem(
        Dt(q(t, x)) ~ -Dx(q(t, x)),
        [
            q(0, x) ~ exp(-x^2), q(t, -1.0) ~ exp(-(-1.0 - t)^2),
            q(t, 1.0) ~ exp(-(1.0 - t)^2),
        ],
        domains, [t, x], [q(t, x)]; name = :reusable_q
    )
    scheme = HermiteScheme()
    discretization = MOLFiniteDifference([x => 0.5], t; advection_scheme = scheme)

    mapping_u = used_scheme(first(symbolic_discretize(system_u, discretization))).deriv_var
    mapping_q = used_scheme(first(symbolic_discretize(system_q, discretization))).deriv_var
    @test isequal(only(keys(mapping_u)), operation(Symbolics.value(u(t, x))))
    @test isequal(only(keys(mapping_q)), operation(Symbolics.value(q(t, x))))
    @test isempty(scheme.deriv_var) && isempty(scheme.diffusive)
end

@testset "HermiteScheme with should_transform = false" begin
    @parameters t x
    @variables u(..) v(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Dx^2
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(-1.0, 1.0)]

    advection = PDESystem(
        Dt(u(t, x)) ~ -Dx(u(t, x)),
        [
            u(0, x) ~ exp(-x^2), u(t, -1.0) ~ exp(-(-1.0 - t)^2),
            u(t, 1.0) ~ exp(-(1.0 - t)^2),
        ],
        domains, [t, x], [u(t, x)]; name = :no_transform_auto
    )
    automatic = MOLFiniteDifference(
        [x => 0.5], t; advection_scheme = HermiteScheme(), should_transform = false
    )
    err = @test_throws ArgumentError symbolic_discretize(advection, automatic)
    @test occursin("should_transform = false", sprint(showerror, err.value))
    @test occursin("deriv_var", sprint(showerror, err.value))
    @test_throws ArgumentError discretize(advection, automatic)

    reused = HermiteScheme()
    sys_reused, _ = symbolic_discretize(advection, MOLFiniteDifference([x => 0.5], t; advection_scheme = reused))
    @test !isempty(used_scheme(sys_reused).deriv_var)
    @test isempty(reused.deriv_var)
    @test_throws ArgumentError symbolic_discretize(
        advection,
        MOLFiniteDifference([x => 0.5], t; advection_scheme = reused, should_transform = false)
    )

    coupled = PDESystem(
        [Dt(u(t, x)) ~ -v(t, x), Dt(v(t, x)) ~ -Dxx(u(t, x))],
        [
            u(0, x) ~ 0.0, v(0, x) ~ 0.0, u(t, -1.0) ~ 0.0, u(t, 1.0) ~ 0.0,
            v(t, -1.0) ~ 0.0, v(t, 1.0) ~ 0.0,
        ],
        domains, [t, x], [u(t, x), v(t, x)]; name = :no_transform_explicit
    )
    explicit(should_transform) = MOLFiniteDifference(
        [x => 0.25], t; advection_scheme = HermiteScheme(deriv_var = u => v),
        should_transform
    )
    sys_off, tspan = symbolic_discretize(coupled, explicit(false))
    @test tspan == (0.0, 0.1)
    eqs_off = string.(ModelingToolkit.equations(sys_off))
    sys_on, _ = symbolic_discretize(coupled, explicit(true))
    @test Set(eqs_off) == Set(string.(ModelingToolkit.equations(sys_on)))
    v2 = only(filter(e -> occursin(r"Differential\(t, 1\)\(\(v\(t\)\)\[2(:\d+)?\]\)", e), eqs_off))
    @test occursin(r"\(v\(t\)\)\[1(:\d+)?\]", v2) && occursin(r"\(v\(t\)\)\[3(:\d+)?\]", v2)
end

@testset "HermiteScheme automatic transformation picks the Dt(u) equation" begin
    @parameters t x
    @variables u(..) w(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    T = 0.1
    u_exact(x, t) = exp(-π^2 * t) * sin(π * x)
    w_exact(x, t) = t * exp(-π^2 * t) * sin(π * x)
    domains = [t ∈ Interval(0.0, T), x ∈ Interval(0.0, 1.0)]
    bcs = [
        u(0, x) ~ sin(π * x), w(0, x) ~ 0.0,
        u(t, 0) ~ 0.0, u(t, 1) ~ 0.0, w(t, 0) ~ 0.0, w(t, 1) ~ 0.0,
    ]
    eq_u = Dt(u(t, x)) ~ Dxx(u(t, x))
    eq_w = Dt(w(t, x)) ~ Dxx(w(t, x)) + u(t, x)
    hermite() = MOLFiniteDifference([x => 0.1], t; advection_scheme = HermiteScheme())

    for (eqs, dvs) in (
            ([eq_u, eq_w], [u(t, x), w(t, x)]), ([eq_w, eq_u], [u(t, x), w(t, x)]),
            ([eq_w, eq_u], [w(t, x), u(t, x)]), ([eq_u, eq_w], [w(t, x), u(t, x)]),
        )
        @named coupled = PDESystem(eqs, bcs, domains, [t, x], dvs)
        sol = solve(discretize(coupled, hermite()); abstol = 1.0e-10, reltol = 1.0e-10)
        @test successful_retcode(sol)
        @test maximum(abs, sol[u(t, x)][end, :] .- u_exact.(sol[x], T)) < 1.0e-4
        @test maximum(abs, sol[w(t, x)][end, :] .- w_exact.(sol[x], T)) < 1.0e-5
    end

    @variables v(..)
    eu = Dt(u(t, x)) ~ Dxx(u(t, x)) + v(t, x)
    ev = Dt(v(t, x)) ~ Dxx(v(t, x)) + u(t, x)
    bcs2 = [
        u(0, x) ~ sin(π * x), v(0, x) ~ sin(π * x),
        u(t, 0) ~ 0.0, u(t, 1) ~ 0.0, v(t, 0) ~ 0.0, v(t, 1) ~ 0.0,
    ]
    texts = map(((eu, ev), (ev, eu))) do order
        @named sys2 = PDESystem(collect(order), bcs2, domains, [t, x], [u(t, x), v(t, x)])
        sort(string.(equations(first(symbolic_discretize(sys2, hermite())))))
    end
    @test texts[1] == texts[2]

    @named wave0 = PDESystem(
        Dt(Dt(u(t, x))) ~ Dxx(u(t, x)),
        [u(0, x) ~ sin(π * x), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0],
        domains, [t, x], [u(t, x)]
    )
    err = @test_throws ArgumentError symbolic_discretize(wave0, hermite())
    @test occursin("initial condition for", sprint(showerror, err.value))

    @named twice = PDESystem(
        [eq_u, Dt(w(t, x)) ~ Dxx(w(t, x)) + Dt(u(t, x))], bcs, domains, [t, x],
        [u(t, x), w(t, x)]
    )
    err = @test_throws ArgumentError symbolic_discretize(twice, hermite())
    @test occursin("exactly one equation", sprint(showerror, err.value))

    @named nonlinear = PDESystem(
        Dt(u(t, x))^2 ~ Dxx(u(t, x)), bcs[[1, 3, 4]], domains, [t, x], [u(t, x)]
    )
    err = @test_throws ArgumentError symbolic_discretize(nonlinear, hermite())
    @test occursin("linearly", sprint(showerror, err.value))

    @named coupled_wall = PDESystem(
        [eq_u, Dt(w(t, x)) ~ Dxx(w(t, x))],
        [
            u(0, x) ~ 0.0, w(0, x) ~ sin(π * x), u(t, 0) ~ w(t, 0), u(t, 1) ~ 0.0,
            w(t, 0) ~ 0.0, w(t, 1) ~ 0.0,
        ],
        domains, [t, x], [u(t, x), w(t, x)]
    )
    err = @test_throws ArgumentError symbolic_discretize(coupled_wall, hermite())
    @test occursin("contain a dependent variable", sprint(showerror, err.value))
end

@testset "HermiteScheme wave equation with Dirichlet walls" begin
    @parameters t x
    @variables u(..) p(..)
    Dt = Differential(t)
    Dx = Differential(x)
    T = 0.75
    u_exact(x, t) = sin(2π * (x - t)) + 0.5 * cos(3π * (x + t))
    p_exact(x, t) = -2π * cos(2π * (x - t)) - 1.5π * sin(3π * (x + t))
    domains = [t ∈ Interval(0.0, T), x ∈ Interval(0.0, 1.0)]
    ubc = [u(t, 0.0) ~ u_exact(0.0, t), u(t, 1.0) ~ u_exact(1.0, t)]
    pbc = [p(t, 0.0) ~ p_exact(0.0, t), p(t, 1.0) ~ p_exact(1.0, t)]
    @named system = PDESystem(
        [Dt(u(t, x)) ~ p(t, x), Dt(p(t, x)) ~ Dx(Dx(u(t, x)))],
        [u(0, x) ~ u_exact(x, 0.0), p(0, x) ~ p_exact(x, 0.0), ubc..., pbc...],
        domains, [t, x], [u(t, x), p(t, x)]
    )
    @named direct = PDESystem(
        Dt(Dt(u(t, x))) ~ Dx(Dx(u(t, x))),
        [u(0, x) ~ u_exact(x, 0.0), Dt(u(0, x)) ~ p_exact(x, 0.0), ubc...],
        domains, [t, x], [u(t, x)]
    )
    function werr(sys, N, ao)
        disc = MOLFiniteDifference(
            [x => 1 / (N - 1)], t; approx_order = ao, advection_scheme = HermiteScheme()
        )
        sol = solve(discretize(sys, disc); saveat = [T], abstol = 1.0e-12, reltol = 1.0e-12)
        @test successful_retcode(sol)
        return maximum(abs, sol[u(t, x)][end, :] .- u_exact.(sol[x], T))
    end
    e21, e41 = werr(system, 21, 2), werr(system, 41, 2)
    @test e41 < 4.0e-5
    @test e21 / e41 > 12
    @test isapprox(werr(direct, 41, 2), e41; rtol = 1.0e-3)
    @test werr(direct, 41, 4) < 1.5e-6
end

@testset "HermiteScheme periodic wave equation as Dt(Dt(u))" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    T = 0.75
    @named wave = PDESystem(
        Dt(Dt(u(t, x))) ~ Dx(Dx(u(t, x))),
        [u(0, x) ~ sin(2π * x), Dt(u(0, x)) ~ -2π * cos(2π * x), u(t, 0.0) ~ u(t, 1.0)],
        [t ∈ Interval(0.0, T), x ∈ Interval(0.0, 1.0)], [t, x], [u(t, x)]
    )
    disc = MOLFiniteDifference([x => 1 / 40], t; advection_scheme = HermiteScheme())
    sol = solve(discretize(wave, disc); saveat = [T], abstol = 1.0e-10, reltol = 1.0e-10)
    @test successful_retcode(sol)
    @test maximum(abs, sol[u(t, x)][end, :] .- sin.(2π .* (sol[x] .- T))) < 1.0e-5
end

@testset "HermiteScheme 2D advection with a free outflow" begin
    @parameters t x y
    @variables u(..)
    Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)
    ua(x, y, t) = sin(2π * (x - 0.8t)) * sin(2π * (y - 0.6t))
    @named adv = PDESystem(
        Dt(u(t, x, y)) ~ -0.8 * Dx(u(t, x, y)) - 0.6 * Dy(u(t, x, y)),
        [u(0, x, y) ~ ua(x, y, 0.0), u(t, 0.0, y) ~ ua(0.0, y, t), u(t, x, 0.0) ~ ua(x, 0.0, t)],
        [t ∈ Interval(0.0, 0.5), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)],
        [t, x, y], [u(t, x, y)]
    )
    disc = MOLFiniteDifference(
        [x => 1 / 16, y => 1 / 16], t; approx_order = 4, advection_scheme = HermiteScheme()
    )
    sol = solve(discretize(adv, disc); saveat = [0.5], abstol = 1.0e-10, reltol = 1.0e-10)
    @test successful_retcode(sol)
    E = [ua(xi, yj, 0.5) for xi in sol[x], yj in sol[y]]
    @test maximum(abs, sol[u(t, x, y)][end, :, :] .- E) < 8.0e-5
end

@testset "HermiteScheme rejects staggered and edge-aligned grids" begin
    @parameters t x
    @variables u(..) ρ(..) ϕ(..)
    Dt, Dx = Differential(t), Differential(x)

    heat1 = PDESystem(
        Dt(u(t, x)) ~ Dx(Dx(u(t, x))),
        [u(0, x) ~ sin(π * x), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0],
        [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)], [t, x], [u(t, x)]; name = :edge1
    )
    edge1 = MOLFiniteDifference(
        [x => 0.1], t; advection_scheme = HermiteScheme(), grid_align = MethodOfLines.edge_align
    )
    err = @test_throws ArgumentError discretize(heat1, edge1)
    @test occursin("EdgeAlignedGrid", sprint(showerror, err.value))

    L = 8.0
    acoustic = PDESystem(
        [Dt(ρ(t, x)) + Dx(ϕ(t, x)) ~ 0, Dt(ϕ(t, x)) + 25.0 * Dx(ρ(t, x)) ~ 0],
        [ρ(0, x) ~ exp(-(x - L / 2)^2), ϕ(0.0, x) ~ 0.0, ρ(t, L) ~ ρ(t, -L), ϕ(t, -L) ~ ϕ(t, L)],
        [t ∈ Interval(0.0, 1.0), x ∈ Interval(-L, L)], [t, x], [ρ(t, x), ϕ(t, x)]; name = :stag
    )
    stag = MOLFiniteDifference(
        [x => 0.125], t; advection_scheme = HermiteScheme(),
        grid_align = MethodOfLines.StaggeredGrid(), edge_aligned_var = ϕ(t, x)
    )
    err = @test_throws ArgumentError symbolic_discretize(acoustic, stag)
    @test occursin("StaggeredGrid", sprint(showerror, err.value))
end

@testset "HermiteScheme mixed derivative u_xy with Dirichlet walls" begin
    @parameters t x y
    @variables u(..)
    Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)
    k, T = 0.1, 0.5
    uexact(x, y, t) = exp(-t) * sin(π * x + 0.5) * (2 + cos(2π * y))
    U, E = u(t, x, y), uexact(x, y, t)
    f = expand_derivatives(Dt(E) - k * (Dx(Dx(E)) + Dy(Dy(E)) + Dx(Dy(E))))
    eq = Dt(U) ~ k * (Dx(Dx(U)) + Dy(Dy(U)) + Dx(Dy(U))) + f
    dom = [t ∈ Interval(0.0, T), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]
    xwalls = [u(t, 0.0, y) ~ uexact(0.0, y, t), u(t, 1.0, y) ~ uexact(1.0, y, t)]
    for (name, ywalls) in (
            (:xy_periodic, [u(t, x, 0.0) ~ u(t, x, 1.0)]),
            (:xy_dirichlet, [u(t, x, 0.0) ~ uexact(x, 0.0, t), u(t, x, 1.0) ~ uexact(x, 1.0, t)]),
        )
        errs = map((9, 17)) do N
            sys = PDESystem(
                [eq], [u(0, x, y) ~ uexact(x, y, 0.0); xwalls; ywalls], dom, [t, x, y], [U]; name
            )
            disc = MOLFiniteDifference(
                [x => 1 / (N - 1), y => 1 / (N - 1)], t; approx_order = 2,
                advection_scheme = HermiteScheme()
            )
            sol = solve(discretize(sys, disc); saveat = [T], abstol = 1.0e-10, reltol = 1.0e-10)
            @test successful_retcode(sol)
            e = abs.(sol[U][end, :, :] .- [uexact(a, b, T) for a in sol[x], b in sol[y]])
            name === :xy_periodic && (e[1, 1] = e[1, end] = e[end, 1] = e[end, end] = 0)
            maximum(e)
        end
        @test errs[2] < 2.0e-4
        @test log2(errs[1] / errs[2]) > 3.5
    end
end

@testset "HermiteScheme advection-diffusion with a Dirichlet outflow" begin
    @parameters t x
    @variables u(..)
    Dt, Dx = Differential(t), Differential(x)
    T = 0.5
    uexact(x, t) = exp(-0.04π^2 * t) * sin(2π * (x - t))
    function aderr(N, ao)
        @named advdiff = PDESystem(
            [Dt(u(t, x)) ~ -Dx(u(t, x)) + 0.01 * Dx(Dx(u(t, x)))],
            [u(0, x) ~ uexact(x, 0.0), u(t, 0.0) ~ uexact(0.0, t), u(t, 1.0) ~ uexact(1.0, t)],
            [t ∈ Interval(0.0, T), x ∈ Interval(0.0, 1.0)], [t, x], [u(t, x)]
        )
        disc = MOLFiniteDifference(
            [x => 1 / (N - 1)], t; approx_order = ao, advection_scheme = HermiteScheme()
        )
        sol = solve(discretize(advdiff, disc); saveat = [T], abstol = 1.0e-10, reltol = 1.0e-10)
        @test successful_retcode(sol)
        return maximum(abs, sol[u(t, x)][end, :] .- uexact.(sol[x], T))
    end
    for (ao, Ns) in ((2, (13, 14)), (4, (17, 18)))
        @test maximum(N -> aderr(N, ao), Ns) < 2.0e-3
        @test aderr(41, ao) < 1.0e-6
    end
    @named adv4 = PDESystem(
        [Dt(u(t, x)) ~ -Dx(u(t, x))],
        [u(0, x) ~ sin(2π * x), u(t, 0.0) ~ sin(-2π * t), u(t, 1.0) ~ sin(2π * (1 - t))],
        [t ∈ Interval(0.0, T), x ∈ Interval(0.0, 1.0)], [t, x], [u(t, x)]
    )
    disc4 = MOLFiniteDifference([x => 1 / 3], t; approx_order = 2, advection_scheme = HermiteScheme())
    @test discretize(adv4, disc4) isa DAEProblem
end

@testset "HermiteScheme damped wave equation as Dt(Dt(u))" begin
    @parameters t x
    @variables u(..) p(..)
    Dt, Dx = Differential(t), Differential(x)
    T = 1.0
    uexact(x, t) = exp(-t) * sin(π * x)
    domains = [t ∈ Interval(0.0, T), x ∈ Interval(0.0, 1.0)]
    walls = [u(t, 0.0) ~ 0.0, u(t, 1.0) ~ 0.0]
    @named direct = PDESystem(
        Dt(Dt(u(t, x))) ~ Dx(Dx(u(t, x))) / π^2 - 2 * Dt(u(t, x)),
        [u(0, x) ~ uexact(x, 0.0), Dt(u(0, x)) ~ -uexact(x, 0.0), walls...],
        domains, [t, x], [u(t, x)]
    )
    @named system = PDESystem(
        [Dt(u(t, x)) ~ p(t, x), Dt(p(t, x)) ~ Dx(Dx(u(t, x))) / π^2 - 2 * p(t, x)],
        [u(0, x) ~ uexact(x, 0.0), p(0, x) ~ -uexact(x, 0.0), walls..., p(t, 0.0) ~ 0.0, p(t, 1.0) ~ 0.0],
        domains, [t, x], [u(t, x), p(t, x)]
    )
    function werr(sys, N)
        disc = MOLFiniteDifference(
            [x => 1 / (N - 1)], t; approx_order = 2, advection_scheme = HermiteScheme()
        )
        sol = solve(discretize(sys, disc); saveat = [T], abstol = 1.0e-12, reltol = 1.0e-12)
        @test successful_retcode(sol)
        return maximum(abs, sol[u(t, x)][end, :] .- uexact.(sol[x], T))
    end
    e21, e41 = werr(direct, 21), werr(direct, 41)
    @test e41 < 4.0e-5
    @test e21 / e41 > 12
    @test isapprox(werr(system, 41), e41; rtol = 1.0e-3)
end

struct PausingGrid <: AbstractVector{Float64}
    v::Vector{Float64}
    armed::Base.RefValue{Bool}
    paused::Channel{Nothing}
    resume::Channel{Nothing}
end
Base.size(g::PausingGrid) = size(g.v)
function Base.getindex(g::PausingGrid, i::Int)
    if g.armed[]
        g.armed[] = false
        put!(g.paused, nothing)
        take!(g.resume)
    end
    return g.v[i]
end

@testset "HermiteScheme() discretizations interleaved on one configuration" begin
    @parameters t x
    @variables u(..) q(..)
    Dt, Dx = Differential(t), Differential(x)
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)]
    @named sysu = PDESystem(
        Dt(u(t, x)) ~ Dx(Dx(u(t, x))), [u(0, x) ~ sin(π * x), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0],
        domains, [t, x], [u(t, x)]
    )
    @named sysq = PDESystem(
        Dt(q(t, x)) ~ Dx(Dx(q(t, x))), [q(0, x) ~ sin(2π * x), q(t, 0) ~ 0.0, q(t, 1) ~ 0.0],
        domains, [t, x], [q(t, x)]
    )
    grid = PausingGrid(collect(range(0.0, 1.0; length = 17)), Ref(false), Channel{Nothing}(1), Channel{Nothing}(1))
    disc = MOLFiniteDifference([x => grid], t; advection_scheme = HermiteScheme())
    uexact(t, x) = exp(-π^2 * t) * sin(π * x)
    function final_error(prob)
        sol = solve(prob; saveat = [0.1], abstol = 1.0e-10, reltol = 1.0e-10)
        @test successful_retcode(sol)
        return maximum(abs, sol[u(t, x)][end, :] .- uexact.(0.1, sol[x]))
    end
    alone = final_error(discretize(sysu, disc))
    function await_pause(task)
        timedwait(() -> isready(grid.paused) || istaskdone(task), 600.0)
        istaskdone(task) && fetch(task)
        isready(grid.paused) || error("the first discretization did not reach the grid")
        return take!(grid.paused)
    end
    grid.armed[] = true
    first_call = @async discretize(sysu, disc)
    await_pause(first_call)
    discretize(sysq, disc)
    put!(grid.resume, nothing)
    @test isapprox(final_error(fetch(first_call)), alone; rtol = 1.0e-6)
    grid.armed[] = true
    first_sys = @async symbolic_discretize(sysu, disc)
    await_pause(first_sys)
    symbolic_discretize(sysq, disc)
    put!(grid.resume, nothing)
    sys, _ = fetch(first_sys)
    @test any(eq -> occursin("uˍx", string(eq.rhs)) && occursin("u(t", string(eq.lhs)), ModelingToolkit.equations(sys))
    @test isempty(disc.advection_scheme.deriv_var)
    @test any(k -> occursin("uˍx", string(k)), keys(get_discrete(sysu, disc)))
    @test isempty(disc.advection_scheme.deriv_var)
end

struct BranchReal <: Real
    v::Float64
end
Base.:+(a::BranchReal, b::Int) = a.v < 2 ? Float32(a.v + b) : a.v + b

@testset "FusedStencil matches broadcast in value and element type" begin
    SU = Symbolics.SymbolicUtils
    @variables A[1:3]
    fused(g) = SU.operation(Symbolics.unwrap(MethodOfLines.fuse_stencils(Symbolics.unwrap(g(A)))))
    c = Float32(2^25)
    b = big"1.0000000000000000000000001"
    dual = [ForwardDiff.Dual{Nothing}(1.0, 1.0) for _ in 1:3]
    for (a, g) in (
            (ones(Float32, 3), A -> (A .+ c) .- c),
            (ones(Float32, 3), A -> (A .+ fill(c, 3)) .- fill(c, 3)),
            (Float32[1, 2, 3], A -> 2 .* A .+ 1),
            (ones(3), A -> (A .+ c) .- c),
            (ones(3), A -> (A .+ fill(2.0^60, 3)) .- 2.0^60),
            (dual, A -> (A .+ 2.0^60) .- 2.0^60),
            (ones(BigFloat, 3), A -> (A .+ b) .- 1),
            (ones(BigFloat, 3), A -> (A .+ ones(BigFloat, 3)) .- 1),
        )
        op = fused(g)
        @test op isa MethodOfLines.FusedStencil
        r = @inferred op(a)
        @test isequal(r, g(a))
        @test eltype(r) == eltype(g(a))
    end
    a = Real[1.0f0, 2.0f0, 3.0f0]
    r = fused(A -> 2 .* A .+ 1)(a)
    @test isequal(r, 2 .* a .+ 1) && eltype(r) == eltype(2 .* a .+ 1)
    mixed = Real[1, big"2.1", 1 // 3]
    for a in (ones(Int, 3), ones(Float32, 3))
        r = fused(A -> A .+ mixed)(a)
        @test isequal(r, a .+ mixed) && eltype(r) == eltype(a .+ mixed)
    end
    branch = [BranchReal(0.25), BranchReal(2.1234567890123452), BranchReal(3.123456789012345)]
    r = fused(A -> A .+ 1)(branch)
    @test isequal(r, branch .+ 1) && eltype(r) == eltype(branch .+ 1)
    @test_throws ArgumentError MethodOfLines._fused_node!(MethodOfLines._FusedStencilBuilder((3,)), 1 + 2im)
end

@testset "HermiteScheme rejects invalid explicit mappings" begin
    @parameters t x
    @variables u(..) v(..) w(..) z(..) q(..)
    Dt, Dx = Differential(t), Differential(x)
    @named sys = PDESystem(
        [
            Dt(u(t, x)) ~ -v(t, x), Dt(v(t, x)) ~ -Dx(Dx(u(t, x))),
            Dt(w(t, x)) ~ Dx(Dx(w(t, x))), Dt(z(t)) ~ -z(t),
        ],
        [
            u(0, x) ~ 0.0, v(0, x) ~ 0.0, w(0, x) ~ 0.0, z(0) ~ 1.0,
            u(t, 0.0) ~ 0.0, u(t, 1.0) ~ 0.0, v(t, 0.0) ~ 0.0, v(t, 1.0) ~ 0.0,
            w(t, 0.0) ~ 0.0, w(t, 1.0) ~ 0.0,
        ],
        [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)], [t, x], [u(t, x), v(t, x), w(t, x), z(t)]
    )
    function message(deriv_var)
        disc = MOLFiniteDifference([x => 0.25], t; advection_scheme = HermiteScheme(; deriv_var))
        try
            symbolic_discretize(sys, disc)
            return ""
        catch e
            e isa ArgumentError || rethrow(e)
            return sprint(showerror, e)
        end
    end
    @test occursin("dependent variables of the PDE system", message(u => q))
    @test occursin("a directional key must be (u, x)", message([(u, t) => v]))
    @test occursin("spatial variables of", message(u => z))
    @test occursin("partner of more than one", message([u => v, w => v]))
    @test occursin("more than one partner", message([u => v, (u, x) => w]))
    @test occursin("different variable", message(u => u))
    @test_throws ArgumentError HermiteScheme(deriv_var = [u => v, u => w])
end

@testset "HermiteScheme coupled first-order system" begin
    @parameters t x
    @variables ρ(..) ϕ(..)
    Dt, Dx = Differential(t), Differential(x)
    T = 0.2
    ρe(t, x) = sin(2π * x) * cos(2π * t)
    eqs = [Dt(ρ(t, x)) ~ -Dx(ϕ(t, x)), Dt(ϕ(t, x)) ~ -Dx(ρ(t, x))]
    bcs = [ρ(0, x) ~ ρe(0, x), ϕ(0, x) ~ 0.0, ρ(t, 0) ~ ρ(t, 1), ϕ(t, 0) ~ ϕ(t, 1)]
    domains = [t ∈ Interval(0.0, T), x ∈ Interval(0.0, 1.0)]
    for dvs in ([ρ(t, x), ϕ(t, x)], [ϕ(t, x), ρ(t, x)])
        sys = PDESystem(eqs, bcs, domains, [t, x], dvs; name = :coupled)
        e16, e32 = map((16, 32)) do N
            disc = MOLFiniteDifference([x => 1 / N], t; approx_order = 2, advection_scheme = HermiteScheme())
            sol = solve(discretize(sys, disc); saveat = [T], abstol = 1.0e-11, reltol = 1.0e-11)
            @test successful_retcode(sol)
            maximum(abs, sol[ρ(t, x)][end, :] .- ρe.(T, sol[x]))
        end
        @test e32 < 5.0e-6
        @test log2(e16 / e32) > 3.5
    end
end

@testset "HermiteScheme diffusion in flux form" begin
    @parameters t x r
    @variables u(..) w(..)
    Dt, Dx, Dr = Differential(t), Differential(x), Differential(r)
    T = 0.05
    hermite(dxs) = MOLFiniteDifference(dxs, t; approx_order = 2, advection_scheme = HermiteScheme())
    final(sys, dxs) = solve(discretize(sys, hermite(dxs)); saveat = [T], abstol = 1.0e-11, reltol = 1.0e-11)

    U = u(t, x)
    bcs = [u(0, x) ~ 1 + 0.2 * sin(2π * x), u(t, 0) ~ u(t, 1)]
    domains = [t ∈ Interval(0.0, T), x ∈ Interval(0.0, 1.0)]
    flux = PDESystem(Dt(U) ~ Dx((1 + U) * Dx(U)), bcs, domains, [t, x], [U]; name = :flux)
    expanded = PDESystem(Dt(U) ~ Dx(U)^2 + (1 + U) * Dx(Dx(U)), bcs, domains, [t, x], [U]; name = :expanded)
    uf, ux = (final(sys, [x => 1 / 32])[U][end, :] for sys in (flux, expanded))
    @test maximum(abs, uf .- ux) < 1.0e-8

    we(t, r) = exp(-π^2 * t) * sin(π * r) / r
    W = w(t, r)
    spherical = PDESystem(
        Dt(W) ~ 1 / r^2 * Dr(r^2 * Dr(W)), [w(0, r) ~ we(0, r), w(t, 0.5) ~ we(t, 0.5), w(t, 1.0) ~ we(t, 1.0)],
        [t ∈ Interval(0.0, T), r ∈ Interval(0.5, 1.0)], [t, r], [W]; name = :spherical
    )
    e11, e21 = map((11, 21)) do N
        sol = final(spherical, [r => 0.5 / (N - 1)])
        @test successful_retcode(sol)
        maximum(abs, sol[W][end, :] .- we.(T, sol[r]))
    end
    @test log2(e11 / e21) > 3.5
end

@testset "HermiteScheme 3D heat with Dirichlet edges" begin
    @parameters t x y z
    @variables u(..)
    Dt, Dx, Dy, Dz = Differential(t), Differential(x), Differential(y), Differential(z)
    g(x, y, z) = 1 + x + y + z
    U = u(t, x, y, z)
    bcs = [
        u(0, x, y, z) ~ g(x, y, z),
        u(t, 0, y, z) ~ g(0, y, z), u(t, 1, y, z) ~ g(1, y, z),
        u(t, x, 0, z) ~ g(x, 0, z), u(t, x, 1, z) ~ g(x, 1, z),
        u(t, x, y, 0) ~ g(x, y, 0), u(t, x, y, 1) ~ g(x, y, 1),
    ]
    domains = [t ∈ Interval(0.0, 0.05), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0), z ∈ Interval(0.0, 1.0)]
    sys = PDESystem(Dt(U) ~ Dx(Dx(U)) + Dy(Dy(U)) + Dz(Dz(U)), bcs, domains, [t, x, y, z], [U]; name = :heat3d)
    disc = MOLFiniteDifference([x => 0.2, y => 0.2, z => 0.2], t; approx_order = 2, advection_scheme = HermiteScheme())
    sol = solve(discretize(sys, disc); saveat = [0.05], abstol = 1.0e-9, reltol = 1.0e-9)
    @test successful_retcode(sol)
    @test maximum(abs, sol[U][end, :, :, :] .- [g(a, b, c) for a in sol[x], b in sol[y], c in sol[z]]) < 1.0e-8
end

@testset "HermiteScheme rejects nonlinear Dirichlet data, stationary systems and fourth derivatives" begin
    @parameters t x
    @variables u(..)
    Dt, Dx = Differential(t), Differential(x)
    nonlinear = PDESystem(
        Dt(u(t, x)) ~ Dx(Dx(u(t, x))), [u(0, x) ~ 1 + 0.5 * sin(π * x), u(t, 0)^2 ~ 1, u(t, 1) ~ 1.0],
        [t ∈ Interval(0.0, 0.05), x ∈ Interval(0.0, 1.0)], [t, x], [u(t, x)]; name = :nonlinear_wall
    )
    err = @test_throws ArgumentError discretize(nonlinear, MOLFiniteDifference([x => 0.1], t; advection_scheme = HermiteScheme()))
    @test occursin("linear in", sprint(showerror, err.value))
    poisson = PDESystem(
        Dx(Dx(u(x))) ~ -π^2 * sin(π * x), [u(0) ~ 0.0, u(1) ~ 0.0], [x ∈ Interval(0.0, 1.0)], [x], [u(x)];
        name = :poisson
    )
    err = @test_throws ArgumentError discretize(poisson, MOLFiniteDifference([x => 0.1], nothing; advection_scheme = HermiteScheme()))
    @test occursin("time-dependent", sprint(showerror, err.value))
    D2(f) = Dx(Dx(f))
    for rhs in (-D2(D2(u(t, x))), -Dx((1 + x) * Dx(D2(u(t, x)))), -D2((1 + t) * D2(u(t, x))))
        beam = PDESystem(
            Dt(u(t, x)) ~ rhs,
            [u(0, x) ~ sin(π * x), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0, Dx(Dx(u(t, 0))) ~ 0.0, Dx(Dx(u(t, 1))) ~ 0.0],
            [t ∈ Interval(0.0, 0.01), x ∈ Interval(0.0, 1.0)], [t, x], [u(t, x)]; name = :beam
        )
        err = @test_throws ArgumentError discretize(beam, MOLFiniteDifference([x => 0.1], t; advection_scheme = HermiteScheme()))
        @test occursin("up to third order", sprint(showerror, err.value))
    end
end

@testset "HermiteScheme results do not depend on the written form of the PDE" begin
    @parameters t x
    @variables u(..)
    Dt, Dx = Differential(t), Differential(x)
    D2(f) = Dx(Dx(f))
    domains = [t ∈ Interval(0.0, 0.05), x ∈ Interval(0.0, 1.0)]
    walls = [u(0, x) ~ sin(π * x), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0]
    function solved(rhs, ao)
        @named sys = PDESystem(Dt(u(t, x)) ~ rhs, walls, domains, [t, x], [u(t, x)])
        disc = MOLFiniteDifference([x => 1 / 16], t; approx_order = ao, advection_scheme = HermiteScheme())
        sol = solve(discretize(sys, disc); saveat = [0.05], abstol = 1.0e-13, reltol = 1.0e-13)
        @test successful_retcode(sol)
        return sol, collect(used_scheme(sys, disc).diffusive)
    end
    forced(t, x) = (π^2 + (1 - π^2) * exp(-π^2 * t)) * sin(π * x)
    for rhs in (D2(u(t, x)) + D2(D2(sin(π * x))), D2(u(t, x)) + π^4 * sin(π * x))
        sol, _ = solved(rhs, 2)
        @test maximum(abs, sol[u(t, x)][end, :] .- forced.(0.05, sol[x])) < 1.0e-5
    end
    uexact(t, x) = exp(-π^2 * (t + t^2 / 2)) * sin(π * x)
    (fsol, ftag), (esol, etag) = solved(Dx((1 + t) * Dx(u(t, x))), 6), solved((1 + t) * D2(u(t, x)), 6)
    @test isequal(ftag, etag)
    ferr = maximum(abs, fsol[u(t, x)][end, :] .- uexact.(0.05, fsol[x]))
    eerr = maximum(abs, esol[u(t, x)][end, :] .- uexact.(0.05, esol[x]))
    @test max(ferr, eerr) < 1.0e-10
    @test isapprox(ferr, eerr; rtol = 1.0e-3)
end

@testset "HermiteScheme explicit 2D partners take Dirichlet corners" begin
    @parameters t x y
    @variables u(..) v(..) w(..)
    Dt, Dy = Differential(t), Differential(y)
    walls(f, c) = [f(t, 0, y) ~ c, f(t, 1, y) ~ c, f(t, x, 0) ~ c, f(t, x, 1) ~ c]
    @named sys = PDESystem(
        [Dt(u(t, x, y)) ~ Dy(v(t, x, y)), Dt(v(t, x, y)) ~ 0, Dt(w(t, x, y)) ~ 0],
        [u(0, x, y) ~ 1; v(0, x, y) ~ 0; w(0, x, y) ~ 0; walls(u, 1); walls(v, 0); walls(w, 0)],
        [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)],
        [t, x, y], [u(t, x, y), v(t, x, y), w(t, x, y)]
    )
    disc = MOLFiniteDifference(
        [x => 0.125, y => 0.125], t; advection_scheme = HermiteScheme(deriv_var = [(u, x) => v, (u, y) => w])
    )
    sol = solve(discretize(sys, disc); saveat = [0.1], abstol = 1.0e-10, reltol = 1.0e-10)
    @test successful_retcode(sol)
    @test maximum(abs, sol[u(t, x, y)][end, :, :] .- 1) < 1.0e-10
end

@testset "HermiteScheme wall conditions with derivatives of another variable" begin
    @parameters t x
    @variables u(..) p(..)
    Dt, Dx = Differential(t), Differential(x)
    pexact(t, x) = exp(-π^2 * t) * cos(π * x)
    ueq = Dt(u(t, x)) ~ Dx(Dx(u(t, x))) + Dx(Dx(p(t, x))) + π^2 * exp(-π^2 * t) * cos(π * x)
    peq = Dt(p(t, x)) ~ Dx(Dx(p(t, x)))
    bcs = [
        u(0, x) ~ 0, p(0, x) ~ cos(π * x), u(t, 0) ~ 0, u(t, 1) ~ 0,
        Dx(p(t, 0)) ~ 0, Dx(p(t, 1)) ~ 0,
    ]
    domains = [t ∈ Interval(0.0, 0.05), x ∈ Interval(0.0, 1.0)]
    disc() = MOLFiniteDifference([x => 1 / 16], t; advection_scheme = HermiteScheme())
    for (eqs, dvs) in (([ueq, peq], [u(t, x), p(t, x)]), ([peq, ueq], [p(t, x), u(t, x)]))
        @named sys = PDESystem(eqs, bcs, domains, [t, x], dvs)
        sol = solve(discretize(sys, disc()); saveat = [0.05], abstol = 1.0e-10, reltol = 1.0e-10)
        @test successful_retcode(sol)
        @test maximum(abs, sol[u(t, x)][end, :]) < 1.0e-5
        @test maximum(abs, sol[p(t, x)][end, :] .- pexact.(0.05, sol[x])) < 1.0e-5
    end
    pbcs = [u(0, x) ~ 0, p(0, x) ~ cos(2π * x), u(t, 0) ~ 0, u(t, 1) ~ 0, p(t, 0) ~ p(t, 1)]
    @named periodic = PDESystem([ueq, peq], pbcs, domains, [t, x], [u(t, x), p(t, x)])
    err = @test_throws ArgumentError discretize(periodic, disc())
    @test occursin("no condition on that wall", sprint(showerror, err.value))
end

@testset "HermiteScheme() third derivatives only along periodic axes" begin
    @parameters t x
    @variables u(..)
    Dt, Dx = Differential(t), Differential(x)
    uexact(t, x) = exp(-t) * (1 + x + x^2 + x^3)
    @named wall = PDESystem(
        Dt(u(t, x)) ~ -Dx(Dx(Dx(u(t, x)))) + exp(-t) * (5 - x - x^2 - x^3),
        [u(0, x) ~ uexact(0, x), u(t, 0) ~ exp(-t), u(t, 1) ~ 4 * exp(-t), Dx(u(t, 1)) ~ 6 * exp(-t)],
        [t ∈ Interval(0.0, 0.05), x ∈ Interval(0.0, 1.0)], [t, x], [u(t, x)]
    )
    err = @test_throws ArgumentError discretize(wall, MOLFiniteDifference([x => 0.05], t; advection_scheme = HermiteScheme()))
    @test occursin("only along periodic axes", sprint(showerror, err.value))
    @named periodic = PDESystem(
        Dt(u(t, x)) ~ -Dx(Dx(Dx(u(t, x)))), [u(0, x) ~ sin(x), u(t, 0) ~ u(t, 2π)],
        [t ∈ Interval(0.0, 0.05), x ∈ Interval(0.0, 2π)], [t, x], [u(t, x)]
    )
    errs = map((16, 32, 64)) do n
        disc = MOLFiniteDifference([x => 2π / n], t; advection_scheme = HermiteScheme())
        sol = solve(discretize(periodic, disc); saveat = [0.05], abstol = 1.0e-11, reltol = 1.0e-11)
        @test successful_retcode(sol)
        maximum(abs, sol[u(t, x)][end, :] .- sin.(sol[x] .+ 0.05))
    end
    @test errs[1] > errs[2] > errs[3]
    @test errs[3] < 1.0e-9
end

@testset "HermiteScheme() generated names avoid user variables" begin
    @parameters t x
    @variables u(..)
    w = only(@variables $(Symbol("uˍx"))(..))
    z = only(@variables $(Symbol("uˍt"))(..))
    Dt, Dx = Differential(t), Differential(x)
    T = 0.05
    domains = [t ∈ Interval(0.0, T), x ∈ Interval(0.0, 1.0)]
    walls(f) = [f(t, 0) ~ 0.0, f(t, 1) ~ 0.0]
    heat(t, x) = exp(-π^2 * t) * sin(π * x)
    disc() = MOLFiniteDifference([x => 1 / 16], t; advection_scheme = HermiteScheme())
    function check(sys, fields)
        partners = collect(values(used_scheme(sys, disc()).deriv_var))
        declared = [operation(Symbolics.value(f)) for (f, _) in fields]
        @test !any(p -> any(d -> isequal(p, d), declared), partners)
        sol = solve(discretize(sys, disc()); saveat = [T], abstol = 1.0e-10, reltol = 1.0e-10)
        @test successful_retcode(sol)
        for (f, g) in fields
            @test maximum(abs, sol[f][end, :] .- g.(T, sol[x])) < 1.0e-5
        end
    end
    @named partner = PDESystem(
        [Dt(u(t, x)) ~ Dx(Dx(u(t, x))), Dt(w(t, x)) ~ Dx(Dx(w(t, x)))],
        [u(0, x) ~ sin(π * x); w(0, x) ~ 2 * sin(π * x); walls(u); walls(w)], domains, [t, x], [u(t, x), w(t, x)]
    )
    check(partner, [(u(t, x), heat), (w(t, x), (t, x) -> 2 * heat(t, x))])
    @named velocity = PDESystem(
        [Dt(Dt(u(t, x))) ~ Dx(Dx(u(t, x))) / π^2 - 2 * Dt(u(t, x)), Dt(z(t, x)) ~ Dx(Dx(z(t, x)))],
        [u(0, x) ~ sin(π * x); Dt(u(0, x)) ~ -sin(π * x); z(0, x) ~ sin(π * x); walls(u); walls(z)],
        domains, [t, x], [u(t, x), z(t, x)]
    )
    check(velocity, [(u(t, x), (t, x) -> exp(-t) * sin(π * x)), (z(t, x), heat)])
end
