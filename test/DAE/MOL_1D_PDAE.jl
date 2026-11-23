# 1D diffusion problem

# Packages and inclusions
using ModelingToolkit, MethodOfLines, LinearAlgebra, Test, OrdinaryDiffEq, DomainSets
using ModelingToolkit: Differential

# Tests
@testset "Dt(u(t,x)) ~ Dxx(u(t,x)), 0 ~ Dxx(v(t,x)) + sin(x), Dirichlet BCs" begin
    # Method of Manufactured Solutions
    u_exact = (x, t) -> exp.(-t) * cos.(x)
    v_exact = (x, t) -> exp.(-t) * sin.(x)

    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..) v(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Dx^2

    # 1D PDE and boundary conditions
    eqs = [
        Dt(u(t, x)) ~ Dxx(u(t, x)),
        0 ~ Dxx(v(t, x)) + exp(-t) * sin(x),
    ]
    bcs = [
        u(0, x) ~ cos(x),
        v(0, x) ~ sin(x),
        u(t, 0) ~ exp(-t),
        Dx(u(t, 1)) ~ -exp(-t) * sin(1),
        Dx(v(t, 0)) ~ exp(-t),
        v(t, 1) ~ exp(-t) * sin(1),
    ]

    # Space and time domains
    domains = [
        t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, 1.0),
    ]

    # PDE system
    @named pdesys = PDESystem(eqs, bcs, domains, [t, x], [u(t, x), v(t, x)])

    # Method of lines discretization
    l = 20
    dx = range(0.0, 1.0, length = l)
    dx_ = dx[2] - dx[1]
    order = 2
    discretization = MOLFiniteDifference([x => dx_], t)

    prob = discretize(pdesys, discretization)
    sol = solve(prob; saveat = 0.1)

    x_sol = sol[x]
    t_sol = sol[t]

    solu = sol[u(t, x)]
    solv = sol[v(t, x)]

    # Test against exact solution
    for i in 1:length(t_sol)
        @test all(isapprox.(u_exact(x_sol, t_sol[i]), solu[i, :], atol = 0.01))
        @test all(isapprox.(v_exact(x_sol, t_sol[i]), solv[i, :], atol = 0.01))
    end
end

@testset "remake with a parameter in the initial condition" begin
    using SymbolicIndexingInterface: setp_oop
    import SciMLBase
    @parameters t x α
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    bcs = [
        u(0, x) ~ α * cos(x), u(t, 0) ~ α * exp(-t), u(t, 1) ~ α * exp(-t) * cos(1),
    ]
    domains = [t ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0)]
    build(α0) = PDESystem(
        Dt(u(t, x)) ~ Dxx(u(t, x)), bcs, domains, [t, x], [u(t, x)], [α => α0];
        name = :heat
    )
    disc = MOLFiniteDifference([x => 0.1], t)
    fresh = discretize(build(2.4), disc; fallback = false)
    newp(prob) = setp_oop(prob, [α])(prob, [2.4])

    # opt-in: remake(p) recomputes the initial condition. The oldest allowed
    # ModelingToolkitBase (1.73, the Downgrade job's floor) cannot compile the
    # initialization system ("different independent variable"); skip only on that error.
    prob = try
        discretize(build(1.2), disc; fallback = false, build_initializeprob = true)
    catch err
        (
            occursin("different independent variable", sprint(showerror, err)) &&
                pkgversion(MethodOfLines.ModelingToolkitBase) < v"1.74"
        ) || rethrow()
        nothing
    end
    if prob === nothing
        @test_skip false
    else
        @test prob isa SciMLBase.DAEProblem
        sol2 = solve(remake(prob; p = newp(prob)); saveat = 0.1)
        sol_fresh = solve(fresh; saveat = 0.1)
        # `prob2.u0` is refreshed by the initialization at solve time
        @test sol2[u(t, x)][1, :] ≈ 2.4 .* cos.(sol2[x])
        @test sol2[u(t, x)] ≈ sol_fresh[u(t, x)] atol = 1.0e-6
    end

    # default: a stale parameter-dependent initial condition is warned about, once per
    # discretize, naming the opt-in; an explicit `build_initializeprob = false` and an
    # initial condition without a parameter stay quiet
    @test_logs (:warn, r"build_initializeprob = true") discretize(
        build(1.2), disc; fallback = false
    )
    @test_logs discretize(build(1.2), disc; fallback = false, build_initializeprob = false)
    if pkgversion(MethodOfLines.ModelingToolkitBase) >= v"1.74"
        disc_init = MOLFiniteDifference([x => 0.1], t; build_initializeprob = true)
        logs, _ = Test.collect_test_logs() do
            discretize(build(1.2), disc_init; fallback = false)
        end
        @test !any(l -> occursin("build_initializeprob = true", l.message), logs)
    end
    if pkgversion(MethodOfLines.ModelingToolkitBase) >= v"1.74"
        # a per-call keyword wins over the one stored on the discretization
        disc_off = MOLFiniteDifference([x => 0.1], t; build_initializeprob = false)
        probo = discretize(
            build(1.2), disc_off; fallback = false, build_initializeprob = true
        )
        @test SciMLBase.has_initialization_data(probo.f)
    end
    bcs_const = [u(0, x) ~ cos(x), u(t, 0) ~ α * exp(-t), u(t, 1) ~ α * exp(-t) * cos(1)]
    @test_logs discretize(
        PDESystem(
            Dt(u(t, x)) ~ Dxx(u(t, x)), bcs_const, domains, [t, x], [u(t, x)], [α => 1.2];
            name = :heat_const
        ), disc; fallback = false
    )

    # default: remake(u0) is honoured
    prob = discretize(build(1.2), disc; fallback = false)
    @test prob isa SciMLBase.DAEProblem
    prob3 = remake(prob; u0 = fill(4.0, length(prob.u0)))
    @test solve(prob3; saveat = 0.1)[u(t, x)][1, 2:(end - 1)] ≈ fill(4.0, 9)

    # default: fully_determined still builds
    @test discretize(build(1.2), disc; fallback = false, fully_determined = true) isa
        SciMLBase.DAEProblem
end
