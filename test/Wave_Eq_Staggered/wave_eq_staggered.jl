using ModelingToolkit, MethodOfLines, DomainSets, Test, Symbolics, SymbolicUtils,
    LinearAlgebra
using SciMLBase: successful_retcode
using OrdinaryDiffEq
using OrdinaryDiffEqLowOrderRK: SplitEuler

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
    @test successful_retcode(sol)

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

# Interior equations carry the time derivative; count those emitted in array (slice) form
# rather than pointwise, to record which interior path a test exercises.
function narrayeqs_interior(sys)
    isarr(x) = let u = Symbolics.unwrap(x)
        !(u isa AbstractArray) && SymbolicUtils.symtype(u) <: AbstractArray
    end
    return count(ModelingToolkit.get_eqs(sys)) do eq
        occursin("Differential(t", string(eq)) && (isarr(eq.lhs) || isarr(eq.rhs))
    end
end

# `discretize` returns a SplitODEProblem without symbolic indexing. Its states are the
# unknowns of the compiled system, in order; return them as `name => grid index`.
function split_states(pdesys, disc)
    sys, _ = symbolic_discretize(pdesys, disc)
    return map(ModelingToolkit.get_unknowns(mtkcompile(sys))) do u
        arr, i = SymbolicUtils.arguments(u)
        nameof(SymbolicUtils.operation(arr)) => SymbolicUtils.unwrap_const(i)
    end
end

# 1D Maxwell's equations with a Drude current J, which has no spatial derivative and lives
# on E's nodes: ε ∂E/∂t = -∂H/∂x - J, μ ∂H/∂t = -∂E/∂x, ∂J/∂t = ωp² E.
# `bcs` gives the initial and boundary conditions in terms of (E, H, J, t, x).
function maxwell_drude(bcs, dx; ε = 1.0, μ = 1.0, ωp = 2.0, L = 1.0, tmax = 1.0)
    t = ModelingToolkit.t_nounits
    @parameters x
    @variables E(..) H(..) J(..)
    Dt = Differential(t)
    Dx = Differential(x)
    eqs = [
        ε * Dt(E(t, x)) ~ -Dx(H(t, x)) - J(t, x),
        μ * Dt(H(t, x)) ~ -Dx(E(t, x)),
        Dt(J(t, x)) ~ ωp^2 * E(t, x),
    ]
    domains = [t ∈ Interval(0.0, tmax), x ∈ Interval(0.0, L)]
    @named pdesys = PDESystem(
        eqs, bcs(E, H, J, t, x, L), domains, [t, x], [E(t, x), H(t, x), J(t, x)]
    )
    disc = MOLFiniteDifference(
        [x => dx], t; grid_align = MethodOfLines.StaggeredGrid(),
        edge_aligned_var = H(t, x)
    )
    return pdesys, disc, (E = E(t, x), H = H(t, x), J = J(t, x))
end

# The reproducer of issue #712: a pulse between PEC walls.
pec_pulse(E, H, J, t, x, L) = [
    E(0, x) ~ exp(-(x - 0.5)^2 / 0.01), H(0, x) ~ 0, J(0, x) ~ 0,
    E(t, 0) ~ 0, E(t, L) ~ 0,
]

# One periodic Fourier mode of E. With H = J = 0 initially this is the standing wave
# E = cos(kx) cos(ωt), which oscillates at the single frequency ω² = ωp²/ε + k²/(εμ).
periodic_mode(k) = (E, H, J, t, x, L) -> [
    E(0, x) ~ cos(k * x), H(0, x) ~ 0.0, J(0, x) ~ 0.0,
    E(t, 0) ~ E(t, L), H(t, 0) ~ H(t, L),
]

# Angular frequency of the oscillation of state `i`, from its zero crossings, which for
# cos(ωt) fall at tₙ = (n + 1/2)π/ω: each is bisected on the dense output, and ω is π over
# the least-squares slope of tₙ against n.
function measured_frequency(sol, i)
    f(τ) = sol(τ)[i]
    crossings = Float64[]
    for n in 1:(length(sol.t) - 1)
        a, b = sol.t[n], sol.t[n + 1]
        sign(f(a)) == sign(f(b)) && continue
        for _ in 1:60
            m = (a + b) / 2
            sign(f(m)) == sign(f(a)) ? (a = m) : (b = m)
        end
        push!(crossings, (a + b) / 2)
    end
    n = 0:(length(crossings) - 1)
    n̄, t̄ = sum(n) / length(n), sum(crossings) / length(crossings)
    slope = sum((n .- n̄) .* (crossings .- t̄)) / sum((n .- n̄) .^ 2)
    return π / slope
end

@testset "Three variables (#712): the reproducer solves on both entry points" begin
    pdesys, disc, vars = maxwell_drude(pec_pulse, 0.05)
    sys, _ = symbolic_discretize(pdesys, disc)
    # every interior collapses to an array equation: this covers the slice-form path
    @test narrayeqs_interior(sys) == 3

    sol = solve(discretize(pdesys, disc), SplitEuler(), dt = 1.0e-3)
    @test successful_retcode(sol)

    dsol = solve(DAEProblem(pdesys, disc))
    @test successful_retcode(dsol)
    @test size(dsol[vars.J]) == (length(dsol.t), 21)
end

@testset "Three variables: the SplitODEProblem keeps every variable's coupling" begin
    pdesys, disc, vars = maxwell_drude(pec_pulse, 0.05)
    prob = discretize(pdesys, disc)
    sys, tspan = symbolic_discretize(pdesys, disc)
    full = ODEProblem(mtkcompile(sys), nothing, tspan)
    states = split_states(pdesys, disc)
    isedge = first.(states) .== :H

    # f1 holds the edge aligned rows, f2 every other row, and together they are the full
    # right hand side; the two-group split zeroed every row that depends only on the
    # variable left out of both groups
    u = sin.(1:length(prob.u0)) .+ 0.5
    du1, du2 = similar(u), similar(u)
    prob.f.f1(du1, u, prob.p, 0.0)
    prob.f.f2(du2, u, prob.p, 0.0)
    @test all(iszero, du1[.!isedge])
    @test all(iszero, du2[isedge])
    @test maximum(abs, du1 .+ du2 .- full.f(u, full.p, 0.0)) < 1.0e-12

    # the split solution matches the DAEProblem, which never splits
    ssol = solve(prob, Tsit5(); abstol = 1.0e-10, reltol = 1.0e-10)
    dsol = solve(DAEProblem(pdesys, disc); abstol = 1.0e-10, reltol = 1.0e-10)
    for name in (:E, :H, :J)
        idx = findall(s -> first(s) == name, states)
        dae_end = dsol[vars[name]][end, last.(states[idx])]
        @test maximum(abs, dae_end) > 0.05
        @test maximum(abs, ssol.u[end][idx] .- dae_end) < 1.0e-6
    end
end

@testset "Three variables: Maxwell–Drude dispersion converges at second order" begin
    ε, μ, ωp, k = 1.0, 0.5, 2.0, 2π
    ω_exact = sqrt(ωp^2 / ε + k^2 / (ε * μ))
    dxs = [1 / 20, 1 / 40, 1 / 80]
    errs = map(dxs) do dx
        pdesys, disc, _ = maxwell_drude(periodic_mode(k), dx; ε, μ, ωp, tmax = 10.0)
        prob = discretize(pdesys, disc)
        sol = solve(prob, Tsit5(); abstol = 1.0e-11, reltol = 1.0e-11)
        states = split_states(pdesys, disc)
        i = argmax([first(s) == :E ? abs(prob.u0[j]) : -1.0 for (j, s) in enumerate(states)])
        ω = measured_frequency(sol, i)
        # the semi-discrete staggered scheme replaces k by its modified wavenumber
        k_mod = 2 / dx * sin(k * dx / 2)
        @test ω ≈ sqrt(ωp^2 / ε + k_mod^2 / (ε * μ)) rtol = 1.0e-8
        abs(ω - ω_exact)
    end
    @test errs[1] < 0.05
    orders = log2.(errs[1:(end - 1)] ./ errs[2:end])
    @test all(o -> 1.9 < o < 2.1, orders)

    # J has no periodic condition, so the E and J interiors fall back to the pointwise
    # path (the one that threw the KeyError) while H stays in array form
    pdesys, disc, _ = maxwell_drude(periodic_mode(k), dxs[1]; ε, μ, ωp)
    @test narrayeqs_interior(first(symbolic_discretize(pdesys, disc))) == 1
end

@testset "Two edge aligned variables" begin
    @parameters t x
    @variables ρ1(..) ϕ1(..) ρ2(..) ϕ2(..)
    Dt = Differential(t)
    Dx = Differential(x)
    a, L, dx = 5.0, 8.0, 0.25
    domains = [t in Interval(0.0, 1.0), x in Interval(-L, L)]
    pair(ρ, ϕ, amp) = (
        [Dt(ρ(t, x)) + Dx(ϕ(t, x)) ~ 0, Dt(ϕ(t, x)) + a^2 * Dx(ρ(t, x)) ~ 0],
        [
            ρ(0, x) ~ amp * exp(-x^2), ϕ(0.0, x) ~ 0.0,
            ρ(t, L) ~ ρ(t, -L), ϕ(t, -L) ~ ϕ(t, L),
        ],
    )
    function solve_final(eqs, bcs, depvars, edge)
        @named pdesys = PDESystem(eqs, bcs, domains, [t, x], depvars)
        disc = MOLFiniteDifference(
            [x => dx], t; grid_align = MethodOfLines.StaggeredGrid(),
            edge_aligned_var = edge
        )
        sol = solve(discretize(pdesys, disc), Tsit5(); abstol = 1.0e-10, reltol = 1.0e-10)
        return Dict(zip(split_states(pdesys, disc), sol.u[end]))
    end

    # two decoupled wave equations, the second with twice the amplitude of the first
    eq1, bc1 = pair(ρ1, ϕ1, 1.0)
    eq2, bc2 = pair(ρ2, ϕ2, 2.0)
    both = solve_final(
        [eq1; eq2], [bc1; bc2], [ρ1(t, x), ϕ1(t, x), ρ2(t, x), ϕ2(t, x)],
        [ϕ1(t, x), ϕ2(t, x)]
    )
    alone = solve_final(eq1, bc1, [ρ1(t, x), ϕ1(t, x)], ϕ1(t, x))
    for ((name, i), v) in alone
        @test both[name => i] ≈ v atol = 1.0e-8
        other = Symbol(replace(string(name), "1" => "2"))
        @test both[other => i] ≈ 2v atol = 1.0e-8
    end
end

@testset "Time dependent boundary values reach the SplitODEProblem" begin
    @parameters t x
    @variables ρ(..) ϕ(..)
    Dt = Differential(t)
    Dx = Differential(x)
    a, L, dx = 5.0, 8.0, 0.5
    eq = [Dt(ρ(t, x)) + Dx(ϕ(t, x)) ~ 0, Dt(ϕ(t, x)) + a^2 * Dx(ρ(t, x)) ~ 0]
    bcs = [
        ρ(0, x) ~ exp(-x^2), ϕ(0.0, x) ~ 0.0,
        Dx(ρ(t, L)) ~ 0.0, ϕ(t, -L) ~ sin(t),
    ]
    domains = [t in Interval(0.0, 1.0), x in Interval(-L, L)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [ρ(t, x), ϕ(t, x)])
    disc = MOLFiniteDifference(
        [x => dx], t, grid_align = MethodOfLines.StaggeredGrid(),
        edge_aligned_var = ϕ(t, x)
    )
    prob = discretize(pdesys, disc)
    sys, tspan = symbolic_discretize(pdesys, disc)
    full = ODEProblem(mtkcompile(sys), nothing, tspan)

    u = cos.(1:length(prob.u0))
    du1, du2 = similar(u), similar(u)
    for τ in (0.0, 0.5, 1.0)
        prob.f.f1(du1, u, prob.p, τ)
        prob.f.f2(du2, u, prob.p, τ)
        @test maximum(abs, du1 .+ du2 .- full.f(u, full.p, τ)) < 1.0e-12
    end
end
