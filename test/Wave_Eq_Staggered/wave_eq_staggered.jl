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
# over more than one point, to record which interior path a test exercises. A one-element
# slice equation is the slice spelling of a pointwise equation.
function narrayeqs_interior(sys)
    isarr(x) = let u = Symbolics.unwrap(x)
        !(u isa AbstractArray) && SymbolicUtils.symtype(u) <: AbstractArray &&
            prod(size(u)) > 1
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
    isf1 = in((:H, :J)).(first.(states))

    # f1 holds the rows of H and J, which are updated from E, f2 those of E, and together
    # they are the full right hand side; the two-group split zeroed every row that depends
    # only on the variable left out of both groups
    u = sin.(1:length(prob.u0)) .+ 0.5
    du1, du2 = similar(u), similar(u)
    prob.f.f1(du1, u, prob.p, 0.0)
    prob.f.f2(du2, u, prob.p, 0.0)
    @test all(iszero, du1[.!isf1])
    @test all(iszero, du2[isf1])
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

# A split integrator advances f1 and f2 in turn. Explicit Euler on each in turn is
# symplectic Euler, Yee's scheme to first order in time, when neither half's rows read its
# own variables. Grouping by staggering put J with E, which J is updated from and which is
# updated from J, so that pair took a forward Euler step and the energy grew every step.
@testset "Three variables: a split step keeps the Maxwell–Drude energy bounded" begin
    ωp, dx = 4.0, 0.05
    pdesys, disc, _ = maxwell_drude(pec_pulse, dx; ωp)
    prob = discretize(pdesys, disc)
    name = first.(split_states(pdesys, disc))
    energy(u) = sum(abs2, u[name .!= :J]) + sum(abs2, u[name .== :J]) / ωp^2
    function split_euler_energy(f1, f2; dt = dx / 2, nsteps = 800)
        u, du = copy(prob.u0), similar(prob.u0)
        return map(1:nsteps) do n
            f1(du, u, prob.p, n * dt)
            u .+= dt .* du
            f2(du, u, prob.p, n * dt)
            u .+= dt .* du
            energy(u)
        end
    end

    sys, tspan = symbolic_discretize(pdesys, disc)
    full = ODEProblem(mtkcompile(sys), nothing, tspan)
    isH = name .== :H
    parity1(du, u, p, t) = (full.f(du, u, full.p, t); du[.!isH] .= 0; nothing)
    parity2(du, u, p, t) = (full.f(du, u, full.p, t); du[isH] .= 0; nothing)

    W0 = energy(prob.u0)
    @test maximum(split_euler_energy(prob.f.f1, prob.f.f2)) < 1.5 * W0
    @test last(split_euler_energy(parity1, parity2)) > 100 * W0
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

@testset "Edge aligned points sit half a step past the grid points" begin
    pdesys, disc, vars = maxwell_drude(
        (E, H, J, t, x, L) -> [
            E(0, x) ~ sin(π * x), H(0, x) ~ x, J(0, x) ~ 0,
            E(t, 0) ~ 0, E(t, L) ~ 0,
        ], 0.1; tmax = 0.1
    )
    sol = solve(DAEProblem(pdesys, disc))
    @test successful_retcode(sol)
    grid = collect(0.0:0.1:1.0)
    # initial conditions are evaluated at each variable's own points
    @test sol[vars.H][1, :] ≈ grid .+ 0.05
    @test sol[vars.E][1, :] ≈ sin.(π .* grid) atol = 1.0e-12
    # and the interpolation knows them: halfway between grid points is an H point
    for i in (1, 5, 10)
        @test sol(sol.t[end], grid[i] + 0.05; dv = vars.H) ≈ sol[vars.H][end, i]
        @test sol(sol.t[end], grid[i]; dv = vars.E) ≈ sol[vars.E][end, i]
    end
    # evaluating on the grid reaches half a step outside the first H point
    @test length(sol(sol.t[end], :; dv = vars.H)) == length(grid)
end

# Cavity modes of Maxwell's equations (ε = μ = 1) in the unit square or cube with
# perfectly conducting walls, on the Yee layout: each E component is staggered along
# itself, each H component across itself, so tangential E sits on the walls.
function cavity_mode_frequency(prob, states, field, k̃, ndims)
    sol = solve(prob, Tsit5(); abstol = 1.0e-10, reltol = 1.0e-10)
    sel = findall(s -> first(s) == field, states)
    i = sel[argmax(abs.(prob.u0[sel]))]
    # a single mode keeps its shape: every point stays in proportion to point i
    drift = maximum(abs, sol.u[end][sel] ./ sol.u[end][i] .- prob.u0[sel] ./ prob.u0[i])
    return measured_frequency(sol, i), drift, sqrt(ndims) * k̃
end

@testset "2D Maxwell cavity on a Yee layout converges at second order" begin
    @parameters t x y
    @variables Hz(..) Ex(..) Ey(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dy = Differential(y)
    X = (t, x, y)
    eqs = [
        Dt(Hz(X...)) ~ Dy(Ex(X...)) - Dx(Ey(X...)),
        Dt(Ex(X...)) ~ Dy(Hz(X...)),
        Dt(Ey(X...)) ~ -Dx(Hz(X...)),
    ]
    walls = [Ex(t, x, 0) ~ 0, Ex(t, x, 1) ~ 0, Ey(t, 0, y) ~ 0, Ey(t, 1, y) ~ 0]
    # the (1, 1) mode, started from its H field or from its E field
    starts = Dict(
        :Hz => [
            Hz(0, x, y) ~ cos(π * x) * cos(π * y), Ex(0, x, y) ~ 0.0, Ey(0, x, y) ~ 0.0,
        ],
        :Ex => [
            Hz(0, x, y) ~ 0.0, Ex(0, x, y) ~ -cos(π * x) * sin(π * y),
            Ey(0, x, y) ~ sin(π * x) * cos(π * y),
        ],
    )
    domains = [t ∈ Interval(0.0, 3.0), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]
    for (field, ics) in starts
        errs = map([1 / 10, 1 / 20]) do dx
            @named pdesys = PDESystem(
                eqs, [ics; walls], domains, [t, x, y], [Hz(X...), Ex(X...), Ey(X...)]
            )
            disc = MOLFiniteDifference(
                [x => dx, y => dx], t; grid_align = MethodOfLines.StaggeredGrid(),
                edge_aligned_var = [Hz(X...), Ex(X...) => x, Ey(X...) => y]
            )
            prob = discretize(pdesys, disc)
            states = split_states(pdesys, disc)

            # Ex and Ey are staggered in one direction, Hz in two: the split separates E
            # (f1) from H (f2)
            u = sin.(1:length(prob.u0))
            du1, du2 = similar(u), similar(u)
            prob.f.f1(du1, u, prob.p, 0.0)
            prob.f.f2(du2, u, prob.p, 0.0)
            isH = first.(states) .== :Hz
            @test all(iszero, du1[isH]) && all(iszero, du2[.!isH])

            ω, drift, ω̃ = cavity_mode_frequency(prob, states, field, 2 / dx * sin(π * dx / 2), 2)
            @test ω ≈ ω̃ rtol = 1.0e-8
            @test drift < 1.0e-10
            abs(ω - sqrt(2) * π)
        end
        @test 1.9 < log2(errs[1] / errs[2]) < 2.1
    end
end

@testset "3D Maxwell cavity on a Yee layout" begin
    @parameters t x y z
    @variables Ex(..) Ey(..) Ez(..) Hx(..) Hy(..) Hz(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dy = Differential(y)
    Dz = Differential(z)
    X = (t, x, y, z)
    eqs = [
        Dt(Ex(X...)) ~ Dy(Hz(X...)) - Dz(Hy(X...)),
        Dt(Ey(X...)) ~ Dz(Hx(X...)) - Dx(Hz(X...)),
        Dt(Ez(X...)) ~ Dx(Hy(X...)) - Dy(Hx(X...)),
        Dt(Hx(X...)) ~ Dz(Ey(X...)) - Dy(Ez(X...)),
        Dt(Hy(X...)) ~ Dx(Ez(X...)) - Dz(Ex(X...)),
        Dt(Hz(X...)) ~ Dy(Ex(X...)) - Dx(Ey(X...)),
    ]
    # the (1, 1, 1) mode, started from a divergence free E field
    ics = [
        Ex(0, x, y, z) ~ cos(π * x) * sin(π * y) * sin(π * z),
        Ey(0, x, y, z) ~ -sin(π * x) * cos(π * y) * sin(π * z),
        Ez(0, x, y, z) ~ 0.0, Hx(0, x, y, z) ~ 0.0, Hy(0, x, y, z) ~ 0.0,
        Hz(0, x, y, z) ~ 0.0,
    ]
    # tangential E vanishes on every wall
    walls = [
        Ex(t, x, 0, z) ~ 0, Ex(t, x, 1, z) ~ 0, Ex(t, x, y, 0) ~ 0, Ex(t, x, y, 1) ~ 0,
        Ey(t, 0, y, z) ~ 0, Ey(t, 1, y, z) ~ 0, Ey(t, x, y, 0) ~ 0, Ey(t, x, y, 1) ~ 0,
        Ez(t, 0, y, z) ~ 0, Ez(t, 1, y, z) ~ 0, Ez(t, x, 0, z) ~ 0, Ez(t, x, 1, z) ~ 0,
    ]
    domains = [
        t ∈ Interval(0.0, 2.0), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0),
        z ∈ Interval(0.0, 1.0),
    ]
    @named pdesys = PDESystem(
        eqs, [ics; walls], domains, [t, x, y, z],
        [Ex(X...), Ey(X...), Ez(X...), Hx(X...), Hy(X...), Hz(X...)]
    )
    dx = 1 / 5
    disc = MOLFiniteDifference(
        [x => dx, y => dx, z => dx], t; grid_align = MethodOfLines.StaggeredGrid(),
        edge_aligned_var = [
            Ex(X...) => x, Ey(X...) => y, Ez(X...) => z,
            Hx(X...) => (y, z), Hy(X...) => (x, z), Hz(X...) => (x, y),
        ]
    )
    prob = discretize(pdesys, disc)
    states = split_states(pdesys, disc)
    ω, drift, ω̃ = cavity_mode_frequency(prob, states, :Ex, 2 / dx * sin(π * dx / 2), 3)
    @test ω ≈ ω̃ rtol = 1.0e-8
    @test drift < 1.0e-10
end
