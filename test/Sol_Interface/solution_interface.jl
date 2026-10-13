# Packages and inclusions
using ModelingToolkit, MethodOfLines, LinearAlgebra, Test, OrdinaryDiffEq, NonlinearSolve,
    DomainSets
using ModelingToolkit: Differential
using SciMLBase
include(joinpath(@__DIR__, "..", "shared", "ode_discretize.jl"))

@testset "Test 00a: Test solution interface, time dependent" begin
    # Parameters, variables, and derivatives
    @parameters t x y
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dy = Differential(y)

    # 1D PDE and boundary conditions
    eq = Dt(u(t, x, y)) ~ -Dx(u(t, x, y)) + Dy(u(t, x, y))

    asf(x, y) = (0.5 / (0.2 * sqrt(2.0 * 3.1415))) * exp(-(x + y - 1.0)^2 / (2.0 * 0.2^2))
    bcs = [
        u(0, x, y) ~ asf(x, y),
        u(t, 0, y) ~ u(t, 2, y),
        u(t, x, 0) ~ u(t, x, 2),
    ]

    # Space and time domains
    domains = [
        t ∈ Interval(0.0, 2.0),
        x ∈ Interval(0.0, 2.0),
        y ∈ Interval(0.0, 2.0),
    ]

    # PDE system
    @named pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])

    # Method of lines discretization
    dx = dy = 2 / 8
    order = 1
    discretization = MOLFiniteDifference(
        [x => dx, y => dy], t; advection_scheme = WENOScheme()
    )
    # explicitly specify upwind order

    prob = ode_discretize(pdesys, discretization)

    using OrdinaryDiffEq
    sol = solve(prob, FBDF(), saveat = 0.1)

    grid = get_discrete(pdesys, discretization)

    traditional_sol = zeros(Float64, length(sol.t), 9, 9)

    solu = sol.original_sol[grid[u(t, x, y)]]

    for t_i in 1:length(sol.t)
        for x_i in 2:9
            for y_i in 2:9
                traditional_sol[t_i, x_i, y_i] = solu[t_i][x_i, y_i]
            end
        end
    end
    #periodic BCs are unknown to ODE, and thus to sol.original_sol
    # so replace here
    traditional_sol[:, :, 1] = traditional_sol[:, :, end]
    traditional_sol[:, 1, :] = traditional_sol[:, end, :]
    traditional_sol[:, 1, 1] .= 0.0 #corners pinned to zero

    @test sol[u(t, x, y)] == traditional_sol
    @test sol[u(t, x, y), 2, :, :] == traditional_sol[2, :, :]
    @test sol[x, 2:3] == sol[x][2:3]
    # `end` would ask the solution for the field's axes.
    @test_throws "cannot be resolved" sol[u(t, x, y), end, :, :]

    @test sol(pi / 2, pi / 2, pi / 2; dv = u(t, x, y)) isa Float64
    @test sol(pi / 2, pi / 2, :) isa Vector{Vector{Float64}}
    @test sol(pi / 2, :, :)[1] isa Matrix{Float64}

    @test sol[t] == sol.t
    @test (sol[x] == sol[y])
    @test (sol[y] isa StepRangeLen)
end

@testset "Test 00b: Test solution interface, time independent" begin
    @parameters x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ 0
    dx = 1 / 8
    dy = 1 / 8

    bcs = [
        u(0, y) ~ x * y,
        u(1, y) ~ x * y,
        u(x, 0) ~ x * y,
        u(x, 1) ~ x * y,
    ]

    # Space and time domains
    domains = [
        x ∈ Interval(0.0, 1.0),
        y ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem([eq], bcs, domains, [x, y], [u(x, y)])

    # Note that we pass in `nothing` for the time variable `t` here since we
    # are creating a stationary problem without a dependence on time, only space.
    discretization = MOLFiniteDifference([x => dx, y => dy], nothing, approx_order = 2)

    prob = discretize(pdesys, discretization)
    sol = NonlinearSolve.solve(prob, NewtonRaphson())

    grid = get_discrete(pdesys, discretization)

    solu = map(d -> sol.original_sol[d], grid[u(x, y)])

    @test sol[u(x, y)] == solu

    @test sol(pi / 4, pi / 4, dv = u(x, y)) isa Float64
    @test sol(pi / 4, :)[1] isa Vector{Float64}

    @test (sol[x] == sol[y])
    @test (sol[y] isa StepRangeLen)
end

@testset "Test 00c: Colon arguments follow the variable's argument order" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    # Time last, and a time domain longer than the space domain.
    eq = Dt(u(x, t)) ~ Dxx(u(x, t))
    bcs = [u(x, 0) ~ cos(x), u(0, t) ~ exp(-t), u(1, t) ~ exp(-t) * cos(1)]
    domains = [t ∈ Interval(0.0, 2.0), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(x, t)])
    sol = solve(discretize(pdesys, MOLFiniteDifference([x => 0.1], t)); saveat = 0.1)

    @test sol(:, :; dv = u(x, t)) == sol[u(x, t)]
    @test sol(:, 0.5; dv = u(x, t)) == sol(sol[x], 0.5; dv = u(x, t))
    @test sol(0.5, :)[1] == sol(sol[x], 0.5; dv = u(x, t))
end

include("array_observables.jl")

# Every stored grid value of the wrapper, and the interpolant at the grid points, equal the
# raw solution's value for that discrete variable, whether it is an unknown or observed.
function wrapper_matches_raw(prob; solve_kwargs...)
    raw = solve(prob; wrap = Val(false), solve_kwargs...)
    sol = SciMLBase.wrap_sol(raw)
    s = sol.disc_data.discretespace
    for u in s.ū
        discu = s.discvars[u]
        stored = sol[Num(u)]
        for I in CartesianIndices(discu)
            expected = raw[discu[I]]
            ndims(discu) == 0 || @test stored[:, I] ≈ expected atol = 1.0e-12
        end
    end
    return sol
end

@testset "Test 10: wrapped values equal the raw solution" begin
    @parameters t x y
    @variables u(..) Ez(..) Hx(..) Hy(..)
    Dt, Dx, Dy = Differential(t), Differential(x), Differential(y)
    heat_eqs = [Dt(u(t, x, y)) ~ Dx(Dx(u(t, x, y))) + Dy(Dy(u(t, x, y)))]
    heat_bcs = [
        u(0, x, y) ~ sin(pi * x) * sin(pi * y), u(t, 0, y) ~ 0, u(t, 1, y) ~ 0,
        u(t, x, 0) ~ 0, u(t, x, 1) ~ y * t,
    ]
    domains = [t ∈ Interval(0.0, 0.01), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]
    @named heat = PDESystem(heat_eqs, heat_bcs, domains, [t, x, y], [u(t, x, y)])
    disc = MOLFiniteDifference([x => 0.125, y => 0.125], t)

    # every grid point an unknown
    sol = wrapper_matches_raw(DAEProblem(heat, disc); saveat = 0.002)
    xs, ys = sol[x], sol[y]
    @test sol(sol.t[end], xs, ys; dv = u(t, x, y)) ≈ sol[u(t, x, y)][end, :, :]
    # boundary points eliminated into observed equations
    sys, tspan = symbolic_discretize(heat, disc)
    wrapper_matches_raw(ODEProblem(mtkcompile(sys), nothing, tspan), saveat = 0.002)

    # staggered, per-direction alignment
    X = (t, x, y)
    tm_eqs = [
        Dt(Ez(X...)) ~ Dx(Hy(X...)) - Dy(Hx(X...)), Dt(Hx(X...)) ~ -Dy(Ez(X...)),
        Dt(Hy(X...)) ~ Dx(Ez(X...)),
    ]
    tm_bcs = [
        Ez(0, x, y) ~ sin(pi * x) * sin(pi * y), Hx(0, x, y) ~ 0, Hy(0, x, y) ~ 0,
        Ez(t, 0, y) ~ 0, Ez(t, 1, y) ~ 0, Ez(t, x, 0) ~ 0, Ez(t, x, 1) ~ 0,
    ]
    @named tm = PDESystem(tm_eqs, tm_bcs, domains, [t, x, y], [Ez(X...), Hx(X...), Hy(X...)])
    tmdisc = MOLFiniteDifference(
        [x => 0.125, y => 0.125], t; grid_align = MethodOfLines.StaggeredGrid(),
        edge_aligned_var = [Hx(X...) => y, Hy(X...) => x]
    )
    wrapper_matches_raw(DAEProblem(tm, tmdisc); saveat = 0.002)
end

@testset "Test 11: wrapped steady state values equal the raw solution" begin
    @parameters x y
    @variables u(..)
    Dx, Dy = Differential(x), Differential(y)
    eqs = [Dx(Dx(u(x, y))) + Dy(Dy(u(x, y))) ~ -sin(pi * x) * sin(pi * y)]
    bcs = [u(0, y) ~ 0, u(1, y) ~ 0, u(x, 0) ~ 0, u(x, 1) ~ x]
    @named pdesys = PDESystem(
        eqs, bcs, [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)], [x, y], [u(x, y)]
    )
    prob = discretize(pdesys, MOLFiniteDifference([x => 0.125, y => 0.125]))
    raw = solve(prob, NewtonRaphson(); wrap = Val(false))
    sol = SciMLBase.wrap_sol(raw)
    discu = sol.disc_data.discretespace.discvars[sol.disc_data.discretespace.ū[1]]
    @test [raw[discu[I]] for I in CartesianIndices(discu)] ≈ sol[u(x, y)] atol = 1.0e-12
end
