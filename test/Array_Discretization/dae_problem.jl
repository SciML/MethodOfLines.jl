# Tests for the DAEProblem path, which builds an implicit-DAE problem from the residuals
# MethodOfLines emits without running `mtkcompile`, so the array (slice-form) equations
# survive into the generated code. Every case is checked against the compiled
# `ODEProblem` path, which this must reproduce.

using MethodOfLines, ModelingToolkit, OrdinaryDiffEq, DomainSets, Symbolics
using SciMLBase
using DiffEqBase: BrownFullBasicInit, ShampineCollocationInit
using OrdinaryDiffEqRosenbrock: Rodas4
using ModelingToolkit: get_eqs
using SymbolicUtils: symtype
using Test

function isarrayeq(eq)
    function isarr(x)
        u = Symbolics.unwrap(x)
        return !(u isa AbstractArray) && symtype(u) <: AbstractArray
    end
    return isarr(eq.lhs) || isarr(eq.rhs)
end

array_disc(dxs, t; kwargs...) = MOLFiniteDifference(dxs, t; kwargs...)

# Solve the same system both ways and return the discretized values of `u` at the final
# time, indexed identically, plus the DAE problem for structural checks.
function solve_both(pdesys, dxs, t, u; disc_kwargs = (;), tol = 1.0e-10)
    disc = array_disc(dxs, t; disc_kwargs...)
    prob_dae = discretize(pdesys, disc)
    @test prob_dae isa SciMLBase.DAEProblem
    sol_dae = solve(prob_dae; reltol = tol, abstol = tol)
    sys, tspan = symbolic_discretize(pdesys, disc)
    prob_ode = ODEProblem(mtkcompile(sys), nothing, tspan)
    sol_ode = solve(prob_ode, Rodas4(); reltol = tol, abstol = tol)
    nx = ndims(get_discrete(pdesys, disc)[u])
    final(sol) = sol[u][end, ntuple(_ -> Colon(), nx)...]
    return prob_dae, sol_dae, final(sol_dae), final(sol_ode)
end

@testset "1D heat, Dirichlet BCs" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    n = 21

    eq = Dt(u(t, x)) ~ Dxx(u(t, x))
    bcs = [u(0, x) ~ sinpi(x), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0]
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    disc = array_disc([x => n], t)
    sys, _ = symbolic_discretize(pdesys, disc)
    @test isempty(MethodOfLines.brown_init_offenders(complete(sys)))
    # the guard is not vacuous: this system does carry initialization equations
    @test !isempty(ModelingToolkit.initialization_equations(complete(sys)))

    prob, sol, dae_vals, ode_vals = solve_both(pdesys, [x => n], t, u(t, x))
    @test SciMLBase.successful_retcode(sol)
    # the solution is wrapped, so it is indexed by `u(t, x)` and interpolates like the
    # `discretize` path rather than being indexed by the discretized variables
    @test sol isa SciMLBase.PDETimeSeriesSolution
    @test sol(0.1, 0.5)[1] ≈ sinpi(0.5) * exp(-pi^2 * 0.1) rtol = 1.0e-2
    @test any(isarrayeq, get_eqs(prob.f.sys))
    @test prob.kwargs[:initializealg] isa BrownFullBasicInit
    # interior points are differential, the two boundaries algebraic
    @test count(prob.differential_vars) == n - 2

    xs = range(0.0, 1.0, length = n)
    exact = @. sinpi(xs) * exp(-pi^2 * 0.1)
    @test maximum(abs, dae_vals .- exact) < 1.0e-3
    @test dae_vals ≈ ode_vals rtol = 1.0e-6
end

@testset "1D advection" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    n = 41

    eq = Dt(u(t, x)) + Dx(u(t, x)) ~ 0
    bcs = [u(0, x) ~ exp(-100 * (x - 0.3)^2), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0]
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    disc = array_disc([x => n], t)
    @test isempty(MethodOfLines.brown_init_offenders(complete(first(symbolic_discretize(pdesys, disc)))))

    _, sol, dae_vals, ode_vals = solve_both(pdesys, [x => n], t, u(t, x))
    @test SciMLBase.successful_retcode(sol)
    @test dae_vals ≈ ode_vals rtol = 1.0e-6
end

@testset "2D diffusion" begin
    @parameters t x y
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    n = 11

    eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y))
    bcs = [
        u(0, x, y) ~ sinpi(x) * sinpi(y),
        u(t, 0, y) ~ 0.0, u(t, 1, y) ~ 0.0,
        u(t, x, 0) ~ 0.0, u(t, x, 1) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.05), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0),
    ]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])

    disc = array_disc([x => n, y => n], t)
    @test isempty(MethodOfLines.brown_init_offenders(complete(first(symbolic_discretize(pdesys, disc)))))

    _, sol, dae_vals, ode_vals = solve_both(
        pdesys, [x => n, y => n], t, u(t, x, y); tol = 1.0e-8
    )
    @test SciMLBase.successful_retcode(sol)
    @test dae_vals ≈ ode_vals rtol = 1.0e-5
end

@testset "Brusselator, periodic BCs" begin
    @parameters t x y
    @variables u(..) v(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    n = 8
    α = 10.0
    brusselator_f(x, y) = (((x - 0.3)^2 + (y - 0.6)^2) <= 0.1^2) * 5.0

    eqs = [
        Dt(u(t, x, y)) ~ 1.0 + v(t, x, y) * u(t, x, y)^2 - 4.4 * u(t, x, y) +
            α * (Dxx(u(t, x, y)) + Dyy(u(t, x, y))) + brusselator_f(x, y),
        Dt(v(t, x, y)) ~ 3.4 * u(t, x, y) - v(t, x, y) * u(t, x, y)^2 +
            α * (Dxx(v(t, x, y)) + Dyy(v(t, x, y))),
    ]
    bcs = [
        u(0, x, y) ~ 22 * (y * (1 - y))^(3 / 2),
        v(0, x, y) ~ 27 * (x * (1 - x))^(3 / 2),
        u(t, 0, y) ~ u(t, 1, y), u(t, x, 0) ~ u(t, x, 1),
        v(t, 0, y) ~ v(t, 1, y), v(t, x, 0) ~ v(t, x, 1),
    ]
    domains = [
        t ∈ Interval(0.0, 0.05), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0),
    ]
    @named pdesys = PDESystem(eqs, bcs, domains, [t, x, y], [u(t, x, y), v(t, x, y)])

    disc = array_disc([x => n, y => n], t)
    @test isempty(MethodOfLines.brown_init_offenders(complete(first(symbolic_discretize(pdesys, disc)))))

    _, sol, dae_vals, ode_vals = solve_both(
        pdesys, [x => n, y => n], t, u(t, x, y); tol = 1.0e-8
    )
    @test SciMLBase.successful_retcode(sol)
    @test dae_vals ≈ ode_vals rtol = 1.0e-5
end

@testset "PDAE with an algebraic variable" begin
    @parameters t x
    @variables u(..) v(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    n = 11

    eqs = [
        Dt(u(t, x)) ~ Dxx(u(t, x)) + v(t, x),
        v(t, x) ~ u(t, x)^2,
    ]
    bcs = [
        u(0, x) ~ sinpi(x), v(0, x) ~ sinpi(x)^2,
        u(t, 0) ~ 0.0, u(t, 1) ~ 0.0,
        v(t, 0) ~ 0.0, v(t, 1) ~ 0.0,
    ]
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eqs, bcs, domains, [t, x], [u(t, x), v(t, x)])

    disc = array_disc([x => n], t)
    sys = complete(first(symbolic_discretize(pdesys, disc)))
    # `v` is algebraic: its initial condition is a guess for the consistent-initialization
    # solve on either path, not a constraint, so no initialization equation is emitted for
    # it and BrownFullBasicInit remains applicable.
    @test isempty(MethodOfLines.brown_init_offenders(sys))

    prob, sol, dae_vals, ode_vals = solve_both(pdesys, [x => n], t, u(t, x))
    @test SciMLBase.successful_retcode(sol)
    @test dae_vals ≈ ode_vals rtol = 1.0e-6
end

@testset "wave equation with a Dt initial condition is rejected" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dtt = Differential(t)^2
    Dxx = Differential(x)^2
    n = 11

    eq = Dtt(u(t, x)) ~ Dxx(u(t, x))
    bcs = [
        u(0, x) ~ sinpi(x), Dt(u(0, x)) ~ 0.0,
        u(t, 0) ~ 0.0, u(t, 1) ~ 0.0,
    ]
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    disc = array_disc([x => n], t)
    offenders = MethodOfLines.brown_init_offenders(
        complete(first(symbolic_discretize(pdesys, disc)))
    )
    @test !isempty(offenders)
    @test all(o -> occursin("time derivative", last(o)), offenders)

    err = try
        DAEProblem(pdesys, disc)
        nothing
    catch e
        e
    end
    @test err isa MethodOfLines.BrownFullBasicInitUnsafeError
    msg = sprint(showerror, err)
    @test occursin("discretize(pdesys, disc)", msg)
    @test occursin("BrownFullBasicInit", msg)
    @test occursin(string(first(first(offenders))), msg)

    fallback_prob = discretize(pdesys, disc)
    @test fallback_prob isa SciMLBase.ODEProblem
    @test SciMLBase.successful_retcode(
        solve(fallback_prob, Rodas4(); reltol = 1.0e-8, abstol = 1.0e-8)
    )
    @test_throws MethodOfLines.BrownFullBasicInitUnsafeError discretize(
        pdesys, disc; fallback = false
    )

    # an explicit `initializealg` overrides the safety gate; the system is then rejected
    # for being second order in time, which no initialization algorithm can fix
    err2 = try
        DAEProblem(pdesys, disc; initializealg = ShampineCollocationInit())
        nothing
    catch e
        e
    end
    @test err2 isa ArgumentError
    @test occursin("order 2", err2.msg)
end

@testset "user-supplied initializealg wins" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    n = 21

    eq = Dt(u(t, x)) ~ Dxx(u(t, x))
    bcs = [u(0, x) ~ sinpi(x), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0]
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    disc = array_disc([x => n], t)
    prob = DAEProblem(pdesys, disc; initializealg = ShampineCollocationInit())
    @test prob.kwargs[:initializealg] isa ShampineCollocationInit
    @test SciMLBase.successful_retcode(
        solve(prob; reltol = 1.0e-8, abstol = 1.0e-8)
    )
end

@testset "explicit compiled ODE path" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq = Dt(u(t, x)) ~ Dxx(u(t, x))
    bcs = [u(0, x) ~ sinpi(x), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0]
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    @test_throws ArgumentError MOLFiniteDifference([x => 11], t; use_ODAE = true)
    disc = MOLFiniteDifference([x => 11], t)
    sys, tspan = symbolic_discretize(pdesys, disc)
    prob = ODEProblem(mtkcompile(sys), nothing, tspan)
    @test prob isa SciMLBase.ODEProblem
    @test SciMLBase.successful_retcode(solve(prob, Rodas4()))
end

# The predicate's algebraic-variable and coupled-unknown branches are unreachable from what
# MethodOfLines currently emits: PDEBase only writes an initialization equation for a
# variable whose time-derivative order in the system exceeds the order of its initial
# condition, which makes the variable differential by construction. They are checked here
# on hand-built systems so the guard is known to discriminate rather than only to accept.
@testset "safety predicate branches" begin
    @independent_variables τ
    @variables a(τ) b(τ)
    D = Differential(τ)
    eqs = [D(a) ~ -a + b, b ~ 2a]
    build(init) = complete(
        System(eqs, τ, [a, b], []; initialization_eqs = init, name = :sys)
    )

    @test isempty(MethodOfLines.brown_init_offenders(build([a ~ 1.0])))

    algebraic = MethodOfLines.brown_init_offenders(build([b ~ 1.0]))
    @test length(algebraic) == 1
    @test occursin("algebraic rather than differential", last(only(algebraic)))

    coupled = MethodOfLines.brown_init_offenders(build([a ~ b]))
    @test length(coupled) == 1
    @test occursin("relates", last(only(coupled)))

    derivative = MethodOfLines.brown_init_offenders(build([D(a) ~ 0.0]))
    @test length(derivative) == 1
    @test occursin("time derivative", last(only(derivative)))
end
