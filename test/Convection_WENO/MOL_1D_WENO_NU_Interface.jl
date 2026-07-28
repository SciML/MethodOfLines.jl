# NU WENO: periodic and two-domain interface BCs.

using ModelingToolkit, MethodOfLines, DomainSets, LinearAlgebra, Test
using OrdinaryDiffEq
using SciMLBase

# Self-similar stretching: refinement isolates asymptotic MMS order.
stretched_grid(a, b, n; amp = 0.15) = [
    let ξ = a + (b - a) * (i - 1) / (n - 1)
            ξ + amp * sinpi(2 * (ξ - a) / (b - a))
    end
        for i in 1:n
]

@testset "Periodic non-uniform WENO: traveling wave accuracy" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    T_END = 0.5
    u_exact(x, t) = sinpi(x - t)

    eq = Dt(u(t, x)) ~ -Dx(u(t, x))
    bcs = [
        u(0, x) ~ sinpi(x),
        u(t, 0.0) ~ u(t, 2.0),
    ]
    domains = [t ∈ Interval(0.0, T_END), x ∈ Interval(0.0, 2.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    errs = map((41, 81, 161)) do n
        g = stretched_grid(0.0, 2.0, n; amp = 0.05)
        @assert all(diff(g) .> 0)
        disc = MOLFiniteDifference([x => g], t; advection_scheme = WENOScheme())
        prob = discretize(pdesys, disc)
        sol = solve(prob, Tsit5(); abstol = 1.0e-10, reltol = 1.0e-10, saveat = [T_END])
        @test SciMLBase.successful_retcode(sol)
        usol = sol[u(t, x)][end, :]
        @test all(isfinite, usol)
        maximum(abs.(usol .- u_exact.(sol[x], T_END)))
    end

    @test errs[1] < 5.0e-3
    orders = [log2(errs[k] / errs[k + 1]) for k in 1:(length(errs) - 1)]
    # NU WENO-5: formally 4th order in smooth regions.
    @test all(>(3.0), orders)
    @test orders[end] > 3.5
end

@testset "Two-domain interface, non-uniform grids: co-refined EOC" begin
    @parameters t x1 x2
    @variables u1(..) u2(..)
    Dt = Differential(t)
    Dx1 = Differential(x1)
    Dx2 = Differential(x2)

    T_END = 0.5
    pulse(x, t) = exp(-((x - t) - 0.7)^2 / (2 * 0.1^2))

    eqs = [
        Dt(u1(t, x1)) ~ -Dx1(u1(t, x1)),
        Dt(u2(t, x2)) ~ -Dx2(u2(t, x2)),
    ]
    bcs = [
        u1(0, x1) ~ pulse(x1, 0.0),
        u2(0, x2) ~ pulse(x2, 0.0),
        u1(t, 0.0) ~ pulse(0.0, t),
        u1(t, 1.0) ~ u2(t, 1.0),
        Dx2(u2(t, 2.0)) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, T_END),
        x1 ∈ Interval(0.0, 1.0),
        x2 ∈ Interval(1.0, 2.0),
    ]
    @named pdesys = PDESystem(
        eqs, bcs, domains, [t, x1, x2], [u1(t, x1), u2(t, x2)]
    )

    # Co-refinement: interval counts double each level, mismatch ratio preserved,
    # so the seam crossing is exercised at every resolution.
    levels = ((41, 61), (81, 121), (161, 241))
    errs = map(levels) do (n1, n2)
        # Deliberately mismatched NU grids across the interface.
        g1 = stretched_grid(0.0, 1.0, n1; amp = 0.03)
        g2 = stretched_grid(1.0, 2.0, n2; amp = 0.04)
        @assert all(diff(g1) .> 0) && all(diff(g2) .> 0)

        disc = MOLFiniteDifference(
            [x1 => g1, x2 => g2], t; advection_scheme = WENOScheme()
        )
        prob = discretize(pdesys, disc)
        sol = solve(prob, Tsit5(); abstol = 1.0e-10, reltol = 1.0e-10, saveat = [T_END])
        @test SciMLBase.successful_retcode(sol)

        u1sol = sol[u1(t, x1)][end, :]
        u2sol = sol[u2(t, x2)][end, :]
        @test all(isfinite, u1sol) && all(isfinite, u2sol)

        # Seam continuity: interface identification is algebraic.
        @test abs(u1sol[end] - u2sol[1]) < 1.0e-8

        # Pulse straddles the interface at T_END.
        e1 = maximum(abs.(u1sol .- pulse.(sol[x1], T_END)))
        e2 = maximum(abs.(u2sol .- pulse.(sol[x2], T_END)))
        max(e1, e2)
    end

    @test errs[2] < 5.0e-3
    orders = [log2(errs[k] / errs[k + 1]) for k in 1:(length(errs) - 1)]
    # NU WENO-5: formally 4th order in smooth regions; no order loss across the seam.
    @test all(>(3.0), orders)
    @test orders[end] > 3.3
end

@testset "Mismatched NU interface rejects higher-order derivatives" begin
    @parameters t x1 x2
    @variables c1(..) c2(..)
    Dt = Differential(t)
    Dx1 = Differential(x1)
    Dx2 = Differential(x2)

    eqs = [
        Dt(c1(t, x1)) ~ Dx1(Dx1(c1(t, x1))),
        Dt(c2(t, x2)) ~ Dx2(Dx2(c2(t, x2))),
    ]
    bcs = [
        c1(0, x1) ~ 1 + cospi(2 * x1),
        c2(0, x2) ~ 1 + cospi(2 * x2),
        Dx1(c1(t, 0.0)) ~ 0.0,
        c1(t, 0.5) ~ c2(t, 0.5),
        Dx1(c1(t, 0.5)) ~ Dx2(c2(t, 0.5)),
        Dx2(c2(t, 1.0)) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.1),
        x1 ∈ Interval(0.0, 0.5),
        x2 ∈ Interval(0.5, 1.0),
    ]
    @named pdesys = PDESystem(
        eqs, bcs, domains, [t, x1, x2], [c1(t, x1), c2(t, x2)]
    )

    g1 = stretched_grid(0.0, 0.5, 21; amp = 0.01)
    g2 = stretched_grid(0.5, 1.0, 31; amp = 0.012)
    disc = MOLFiniteDifference(
        [x1 => g1, x2 => g2], t; advection_scheme = WENOScheme()
    )
    # Dxx crossing a mismatched NU interface is not coordinate-aware; must be rejected.
    @test_throws ArgumentError discretize(pdesys, disc)
end
