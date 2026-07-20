# Tests for the ArrayDiscretization strategy (issue #428): the interior of each PDE is
# represented as a single symbolic array equation over slices of the array variables.
# Every case is checked for agreement against ScalarizedDiscretization, which the array
# strategy must reproduce exactly (it uses the same stencils, and falls back to the
# scalar path for unsupported patterns).

using MethodOfLines, ModelingToolkit, OrdinaryDiffEq, DomainSets, Symbolics
using OrdinaryDiffEqRosenbrock: Rodas4
using OrdinaryDiffEqSSPRK: SSPRK22
using NonlinearSolve: NewtonRaphson
using ModelingToolkit: get_eqs
using SymbolicUtils: symtype
using Test

# Solve pdesys with both strategies and return (array_sol, scalar_sol, array_sys)
function solve_both(pdesys, dxs, t; disc_kwargs = (;), solver = Rodas4(), kwsolve = (;))
    disc_arr = MOLFiniteDifference(
        dxs, t; discretization_strategy = ArrayDiscretization(), disc_kwargs...
    )
    disc_scal = MOLFiniteDifference(
        dxs, t; discretization_strategy = ScalarizedDiscretization(), disc_kwargs...
    )
    sys_arr, _ = symbolic_discretize(pdesys, disc_arr)
    prob_arr = discretize(pdesys, disc_arr)
    prob_scal = discretize(pdesys, disc_scal)
    sol_arr = solve(prob_arr, solver; reltol = 1.0e-10, abstol = 1.0e-10, kwsolve...)
    sol_scal = solve(prob_scal, solver; reltol = 1.0e-10, abstol = 1.0e-10, kwsolve...)
    return sol_arr, sol_scal, sys_arr
end

# The number of equations whose left/right hand side is an unscalarized symbolic array.
# symtype falls back to typeof for non-symbolic values, and literal-array sides (like the
# zeros rhs of a slice-form equation) only count when the other side is symbolic, so this
# counts exactly the array-form equations.
function narrayeqs(sys)
    function isarr(x)
        u = Symbolics.unwrap(x)
        return !(u isa AbstractArray) && symtype(u) <: AbstractArray
    end
    return count(eq -> isarr(eq.lhs) || isarr(eq.rhs), get_eqs(sys))
end

@testset "1D linear diffusion, Dirichlet BCs" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq = Dt(u(t, x)) ~ Dxx(u(t, x))
    bcs = [u(0, x) ~ sinpi(x), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0]
    domains = [t ∈ Interval(0.0, 0.2), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    sol_arr, sol_scal, sys_arr = solve_both(pdesys, [x => 0.05], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    # The interior must be a single array equation
    @test narrayeqs(sys_arr) == 1
    @test sol_arr[u(t, x)] ≈ sol_scal[u(t, x)] rtol = 1.0e-6

    # Against the analytic solution
    xdisc = sol_arr[x]
    tdisc = sol_arr[t]
    exact = [exp(-pi^2 * ti) * sinpi(xi) for ti in tdisc, xi in xdisc]
    @test maximum(abs.(sol_arr[u(t, x)] .- exact)) < 1.0e-2
end

@testset "1D diffusion, Neumann and Robin BCs" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    eq = Dt(u(t, x)) ~ Dxx(u(t, x))
    bcs = [
        u(0, x) ~ cospi(x),
        Dx(u(t, 0)) ~ 0.0,
        u(t, 1) + Dx(u(t, 1)) ~ -exp(-pi^2 * t),
    ]
    domains = [t ∈ Interval(0.0, 0.2), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    sol_arr, sol_scal, sys_arr = solve_both(pdesys, [x => 0.05], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs(sys_arr) == 1
    @test sol_arr[u(t, x)] ≈ sol_scal[u(t, x)] rtol = 1.0e-6
end

@testset "1D advection-diffusion, constant coefficient (winding rules)" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    eq = Dt(u(t, x)) ~ -2.0 * Dx(u(t, x)) + 0.1 * Dxx(u(t, x))
    bcs = [u(0, x) ~ exp(-100 * (x - 0.3)^2), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0]
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    sol_arr, sol_scal, sys_arr = solve_both(pdesys, [x => 0.02], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs(sys_arr) == 1
    @test sol_arr[u(t, x)] ≈ sol_scal[u(t, x)] rtol = 1.0e-6
end

@testset "1D nonlinear advection (Burgers-type, coefficient depends on u)" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    eq = Dt(u(t, x)) ~ -u(t, x) * Dx(u(t, x)) + 0.05 * Dxx(u(t, x))
    bcs = [u(0, x) ~ sinpi(2x) + 1.0, u(t, 0) ~ 1.0, u(t, 1) ~ 1.0]
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    sol_arr, sol_scal, sys_arr = solve_both(pdesys, [x => 0.02], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs(sys_arr) == 1
    @test sol_arr[u(t, x)] ≈ sol_scal[u(t, x)] rtol = 1.0e-6
end

@testset "1D diffusion with space and time dependent coefficient" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq = Dt(u(t, x)) ~ (1.1 + sinpi(x)) * Dxx(u(t, x)) + (1 + t) * u(t, x)
    bcs = [u(0, x) ~ sinpi(x), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0]
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    sol_arr, sol_scal, sys_arr = solve_both(pdesys, [x => 0.05], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs(sys_arr) == 1
    @test sol_arr[u(t, x)] ≈ sol_scal[u(t, x)] rtol = 1.0e-6
end

@testset "1D diffusion, fourth order approximation (frame points)" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq = Dt(u(t, x)) ~ Dxx(u(t, x))
    bcs = [u(0, x) ~ sinpi(x), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0]
    domains = [t ∈ Interval(0.0, 0.2), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    sol_arr, sol_scal, sys_arr = solve_both(
        pdesys, [x => 0.05], t; disc_kwargs = (; approx_order = 4)
    )
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    # One array equation for the core, plus scalar frame equations near the boundaries
    @test narrayeqs(sys_arr) == 1
    @test sol_arr[u(t, x)] ≈ sol_scal[u(t, x)] rtol = 1.0e-6
end

@testset "1D diffusion on a nonuniform grid" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq = Dt(u(t, x)) ~ Dxx(u(t, x))
    bcs = [u(0, x) ~ sinpi(x), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0]
    domains = [t ∈ Interval(0.0, 0.2), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    # A deterministic, smoothly stretched nonuniform grid
    gridvec = [0.5 * (1 - cospi(i / 20)) for i in 0:20]
    sol_arr, sol_scal, sys_arr = solve_both(pdesys, [x => gridvec], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs(sys_arr) == 1
    @test sol_arr[u(t, x)] ≈ sol_scal[u(t, x)] rtol = 1.0e-6
end

@testset "2D linear diffusion" begin
    @parameters t x y
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

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

    sol_arr, sol_scal, sys_arr = solve_both(pdesys, [x => 0.1, y => 0.1], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs(sys_arr) == 1
    @test sol_arr[u(t, x, y)] ≈ sol_scal[u(t, x, y)] rtol = 1.0e-6
end

@testset "Coupled system of two variables" begin
    @parameters t x
    @variables u(..) v(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eqs = [
        Dt(u(t, x)) ~ Dxx(u(t, x)) + v(t, x),
        Dt(v(t, x)) ~ Dxx(v(t, x)) - u(t, x),
    ]
    bcs = [
        u(0, x) ~ sinpi(x), v(0, x) ~ 0.0,
        u(t, 0) ~ 0.0, u(t, 1) ~ 0.0,
        v(t, 0) ~ 0.0, v(t, 1) ~ 0.0,
    ]
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eqs, bcs, domains, [t, x], [u(t, x), v(t, x)])

    sol_arr, sol_scal, sys_arr = solve_both(pdesys, [x => 0.05], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs(sys_arr) == 2
    @test sol_arr[u(t, x)] ≈ sol_scal[u(t, x)] rtol = 1.0e-6
    @test sol_arr[v(t, x)] ≈ sol_scal[v(t, x)] rtol = 1.0e-6
end

@testset "Fallback: periodic BCs still match the scalar path" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    eq = Dt(u(t, x)) ~ Dxx(u(t, x))
    bcs = [u(0, x) ~ sinpi(2x), u(t, 0) ~ u(t, 1)]
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    sol_arr, sol_scal, sys_arr = solve_both(pdesys, [x => 0.05], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    # Periodic BCs are not representable as slices; the whole equation falls back
    @test narrayeqs(sys_arr) == 0
    @test sol_arr[u(t, x)] ≈ sol_scal[u(t, x)] rtol = 1.0e-6
end

@testset "Fallback: nonlinear laplacian still matches the scalar path" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    eq = Dt(u(t, x)) ~ Dx(u(t, x) * Dx(u(t, x)))
    bcs = [u(0, x) ~ 1.0 + sinpi(x) / 2, u(t, 0) ~ 1.0, u(t, 1) ~ 1.0]
    domains = [t ∈ Interval(0.0, 0.05), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    sol_arr, sol_scal, sys_arr = solve_both(pdesys, [x => 0.05], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs(sys_arr) == 0
    @test sol_arr[u(t, x)] ≈ sol_scal[u(t, x)] rtol = 1.0e-6
end

@testset "Fallback: WENO scheme still matches the scalar path" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    eq = Dt(u(t, x)) ~ -Dx(u(t, x))
    bcs = [u(0, x) ~ exp(-100 * (x - 0.3)^2), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0]
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    sol_arr, sol_scal, sys_arr = solve_both(
        pdesys, [x => 0.02], t;
        disc_kwargs = (; advection_scheme = WENOScheme()), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-3)
    )
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs(sys_arr) == 0
    @test sol_arr[u(t, x)] ≈ sol_scal[u(t, x)] rtol = 1.0e-6
end

@testset "Stationary (NonlinearProblem) still works with ArrayDiscretization" begin
    @parameters x
    @variables u(..)
    Dxx = Differential(x)^2

    eq = Dxx(u(x)) ~ -sinpi(x) * pi^2
    bcs = [u(0) ~ 0.0, u(1) ~ 0.0]
    domains = [x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem([eq], bcs, domains, [x], [u(x)])

    disc_arr = MOLFiniteDifference(
        [x => 0.05]; discretization_strategy = ArrayDiscretization()
    )
    disc_scal = MOLFiniteDifference(
        [x => 0.05]; discretization_strategy = ScalarizedDiscretization()
    )
    prob = discretize(pdesys, disc_arr)
    prob_scal = discretize(pdesys, disc_scal)
    # `mtkcompile` tears this linear system down to a single unknown, for which the
    # solver's progress-based stall criterion can trip while the returned solution is
    # fully converged: locally the scalar path reports `Stalled` under `TrustRegion` and
    # `Success` under `NewtonRaphson`, with the two solutions agreeing to 1.1e-14, and
    # the array path reports `Success` under all three. The retcode therefore tests the
    # solver's convergence reporting on a degenerate system rather than this strategy, so
    # assert the solution itself (a stronger check: a non-converged solve fails these).
    sol = solve(prob, NewtonRaphson())
    sol_scal = solve(prob_scal, NewtonRaphson())
    xs = sol[x]
    @test sol[u(x)] ≈ sol_scal[u(x)] rtol = 1.0e-8
    @test sol[u(x)] ≈ sinpi.(xs) atol = 1.0e-2
    @test all(isfinite, sol[u(x)])
end
