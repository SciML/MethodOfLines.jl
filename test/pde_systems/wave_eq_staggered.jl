using ModelingToolkit, MethodOfLines, DomainSets, Test, Symbolics, SymbolicUtils,
    LinearAlgebra
using OrdinaryDiffEq

@testset "1D wave equation, staggered grid, Mixed BC" begin
    @parameters t x
    @variables ρ(..) ϕ(..)
    Dt = Differential(t)
    Dx = Differential(x)

    a = 5.0
    L = 8.0
    dx = 0.125
    dt = (dx / a)^2
    tmax = 10.0

    initialFunction(x) = exp(-(x)^2)
    eq = [
        Dt(ρ(t, x)) + Dx(ϕ(t, x)) ~ 0,
        Dt(ϕ(t, x)) + a^2 * Dx(ρ(t, x)) ~ 0,
    ]
    bcs = [
        ρ(0, x) ~ initialFunction(x),
        ϕ(0.0, x) ~ 0.0,
        Dx(ρ(t, L)) ~ 0.0,
        ϕ(t, -L) ~ 0.0,
    ] #-a^2*Dx(ρ(t,L))];

    domains = [
        t in Interval(0.0, tmax),
        x in Interval(-L, L),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [ρ(t, x), ϕ(t, x)])

    discretization = MOLFiniteDifference(
        [x => dx], t, grid_align = MethodOfLines.StaggeredGrid(),
        edge_aligned_var = ϕ(t, x)
    )
    prob = discretize(pdesys, discretization)

    sol = solve(prob, SplitEuler(), dt = dt)

    test_ind = floor(Int, (2(L - dx) / a) / (dt))
    @test_broken maximum(sol[1:128, 1] .- sol[1:128, test_ind]) < max(dx^2, dt)
    @test_broken maximum(sol[1:128, 1] .- sol[1:128, 2 * test_ind]) < 10 * max(dx^2, dt)
end

# Neumann BC test: staggered discretization produces boundary terms (e.g. ϕ(t, 8.0))
# that aren't in the discretized unknowns. MTK v11's mtkcompile is stricter about this
# validation than the old structural_simplify. This is a pre-existing staggered grid
# boundary handling issue now surfaced by MTK v11.
@testset "1D wave equation, staggered grid, Neumann BC" begin
    @test_broken begin
        @parameters t x
        @variables ρ(..) ϕ(..)
        Dt = Differential(t)
        Dx = Differential(x)

        a = 5.0
        L = 8.0
        dx = 0.125
        dt = dx / a
        tmax = 10.0

        initialFunction(x) = exp(-(x)^2)
        eq = [
            Dt(ρ(t, x)) + Dx(ϕ(t, x)) ~ 0,
            Dt(ϕ(t, x)) + a^2 * Dx(ρ(t, x)) ~ 0,
        ]
        bcs = [
            ρ(0, x) ~ initialFunction(x),
            ϕ(0.0, x) ~ 0.0,
            Dt(ρ(t, L)) - (1 / a) * Dt(ϕ(t, L)) ~ 0.0,
            Dt(ρ(t, -L)) + (1 / a) * Dt(ϕ(t, -L)) ~ 0.0,
        ]

        domains = [
            t in Interval(0.0, tmax),
            x in Interval(-L, L),
        ]

        @named pdesys = PDESystem(eq, bcs, domains, [t, x], [ρ(t, x), ϕ(t, x)])

        discretization = MOLFiniteDifference(
            [x => dx], t, grid_align = MethodOfLines.StaggeredGrid(),
            edge_aligned_var = ϕ(t, x)
        )
        prob = discretize(pdesys, discretization)
        sol = solve(prob, SplitEuler(), dt = (dx / a)^2)
        maximum(sol[:, end]) < 1.0e-3
    end
end

@testset "1D wave equation, staggered grid, Periodic BC" begin
    @parameters t x
    @variables ρ(..) ϕ(..)
    Dt = Differential(t)
    Dx = Differential(x)

    a = 5.0
    L = 8.0
    dx = 0.125
    dt = (dx / a)^2
    tmax = 10.0

    initialFunction(x) = exp(-(x - L / 2)^2)
    eq = [
        Dt(ρ(t, x)) + Dx(ϕ(t, x)) ~ 0,
        Dt(ϕ(t, x)) + a^2 * Dx(ρ(t, x)) ~ 0,
    ]
    bcs = [
        ρ(0, x) ~ initialFunction(x),
        ϕ(0.0, x) ~ 0.0,
        ρ(t, L) ~ ρ(t, -L),
        ϕ(t, -L) ~ ϕ(t, L),
    ]

    domains = [
        t in Interval(0.0, tmax),
        x in Interval(-L, L),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [ρ(t, x), ϕ(t, x)])

    discretization = MOLFiniteDifference(
        [x => dx], t, grid_align = MethodOfLines.StaggeredGrid(),
        edge_aligned_var = ϕ(t, x)
    )
    prob = discretize(pdesys, discretization)

    sol = solve(prob, SplitEuler(), dt = dt)

    test_ind = round(Int, ((2 * L - dx) / a) / (dt))
    @test_broken maximum(sol[1:128, 1] .- sol[1:128, test_ind]) < max(dx^2, dt)
    @test_broken maximum(sol[1:128, 1] .- sol[1:128, 2 * test_ind]) < 10 * max(dx^2, dt)
end
