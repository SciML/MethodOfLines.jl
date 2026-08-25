# Tests for equation discretization: the interior of each supported PDE is represented as
# symbolic array equations over slices of the array variables.
# Unsupported patterns fall back to pointwise equations automatically.

using MethodOfLines, ModelingToolkit, OrdinaryDiffEq, DomainSets, Symbolics
using SciMLBase
using SciMLBase: successful_retcode
using OrdinaryDiffEqRosenbrock: Rodas4
using OrdinaryDiffEqSSPRK: SSPRK22
using OrdinaryDiffEqLowOrderRK: SplitEuler
using NonlinearSolve: NewtonRaphson
using ModelingToolkit: get_eqs
using SymbolicUtils
using SymbolicUtils: symtype
using Test
include(joinpath(@__DIR__, "..", "shared", "ode_discretize.jl"))

# Solve the compiled ODE form used by explicit-solver tests and return its symbolic
# array system for structural checks.
function solve_discretized(pdesys, dxs, t; disc_kwargs = (;), solver = Rodas4(), kwsolve = (;))
    disc = MOLFiniteDifference(dxs, t; disc_kwargs...)
    sys_arr, _ = symbolic_discretize(pdesys, disc)
    prob_arr = ode_discretize(pdesys, disc)
    sol_arr = solve(prob_arr, solver; reltol = 1.0e-10, abstol = 1.0e-10, kwsolve...)
    return sol_arr, sys_arr
end

# The number of equations whose left/right hand side is an unscalarized symbolic array.
# symtype falls back to typeof for non-symbolic values, and literal-array sides (like the
# zeros rhs of a slice-form equation) only count when the other side is symbolic, so this
# counts exactly the array-form equations.
function narrayeqs(sys)
    return count(isarrayeq, get_eqs(sys))
end

function isarrayeq(eq)
    function isarr(x)
        u = Symbolics.unwrap(x)
        return !(u isa AbstractArray) && symtype(u) <: AbstractArray
    end
    return isarr(eq.lhs) || isarr(eq.rhs)
end

# Interior equations carry the time derivative; boundary equations do not. Boundary
# conditions on a face are emitted in array form too, so counting only the interior
# expresses "the interior collapsed to a single equation" unambiguously.
isinterioreq(eq) = occursin("Differential(t", string(eq))
narrayeqs_interior(sys) = count(
    eq -> isinterioreq(eq) && isarrayeq(eq), get_eqs(sys)
)

# Incompatible two-domain layouts must raise, not `BoundsError`, and must not
# depend on which equation is discretized first (upper faces skip BC emit).
function throws_incompatible_layout(pdesys, dxs, t)
    thrown = try
        symbolic_discretize(pdesys, MOLFiniteDifference(dxs, t))
        nothing
    catch e
        e
    end
    return thrown isa ArgumentError &&
        occursin("incompatible layout", sprint(showerror, thrown))
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

    sol_arr, sys_arr = solve_discretized(pdesys, [x => 0.05], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    # The interior must be a single array equation
    @test narrayeqs_interior(sys_arr) == 1

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

    sol_arr, sys_arr = solve_discretized(pdesys, [x => 0.05], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
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

    sol_arr, sys_arr = solve_discretized(pdesys, [x => 0.02], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
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

    sol_arr, sys_arr = solve_discretized(pdesys, [x => 0.02], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
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

    sol_arr, sys_arr = solve_discretized(pdesys, [x => 0.05], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
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

    sol_arr, sys_arr = solve_discretized(
        pdesys, [x => 0.05], t; disc_kwargs = (; approx_order = 4)
    )
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    # One array equation for the core, plus scalar frame equations near the boundaries
    @test narrayeqs_interior(sys_arr) == 1
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
    sol_arr, sys_arr = solve_discretized(pdesys, [x => gridvec], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
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

    sol_arr, sys_arr = solve_discretized(pdesys, [x => 0.1, y => 0.1], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
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

    sol_arr, sys_arr = solve_discretized(pdesys, [x => 0.05], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 2
end

@testset "1D periodic BCs" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq = Dt(u(t, x)) ~ Dxx(u(t, x))
    bcs = [u(0, x) ~ sinpi(2x), u(t, 0) ~ u(t, 1)]
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    sol_arr, sys_arr = solve_discretized(pdesys, [x => 0.05], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    # The stencil is translation invariant across the whole periodic direction, so the
    # interior is one array equation plus the points whose stencils wrap over the seam,
    # which in 1D are single points and stay scalar.
    @test narrayeqs_interior(sys_arr) == 1

    # The periodic solution is a decaying sine wave
    xdisc = sol_arr[x]
    tdisc = sol_arr[t]
    exact = [exp(-4 * pi^2 * ti) * sinpi(2xi) for ti in tdisc, xi in xdisc]
    @test maximum(abs.(sol_arr[u(t, x)] .- exact)) < 1.0e-2
end

@testset "1D periodic advection" begin
    # The winding (upwind) stencils wrap over the seam as well, and asymmetrically: only
    # one end of the interior needs the extra equations for each wind direction.
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    eq = Dt(u(t, x)) ~ -1.5 * Dx(u(t, x)) + 0.02 * Dxx(u(t, x))
    bcs = [u(0, x) ~ sinpi(2x), u(t, 0) ~ u(t, 1)]
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    sol_arr, sys_arr = solve_discretized(pdesys, [x => 0.02], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
end

@testset "2D periodic BCs in both directions" begin
    @parameters t x y
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y))
    bcs = [
        u(0, x, y) ~ sinpi(2x) * sinpi(2y),
        u(t, 0, y) ~ u(t, 1, y),
        u(t, x, 0) ~ u(t, x, 1),
    ]
    domains = [
        t ∈ Interval(0.0, 0.02), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0),
    ]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])

    sol_arr, sys_arr = solve_discretized(pdesys, [x => 0.1, y => 0.1], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    # One equation for the points whose stencils do not wrap, one for each of the four
    # slabs along the seams, and a scalar one for each of the four points where two seams
    # meet: the slabs are the array equations that make this scale.
    @test narrayeqs_interior(sys_arr) == 5

    # and the equation count does not grow with the grid
    counts = map([8, 16]) do n
        disc = MOLFiniteDifference(
            [x => 1 / (n - 1), y => 1 / (n - 1)], t
        )
        sys, _ = symbolic_discretize(pdesys, disc)
        length(get_eqs(sys))
    end
    @test counts[1] == counts[2]
end

@testset "2D periodic in one direction, Dirichlet in the other" begin
    @parameters t x y
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y))
    bcs = [
        u(0, x, y) ~ sinpi(2x) * sinpi(y),
        u(t, 0, y) ~ u(t, 1, y),
        u(t, x, 0) ~ 0.0, u(t, x, 1) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.02), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0),
    ]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])

    sol_arr, sys_arr = solve_discretized(pdesys, [x => 0.1, y => 0.1], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    # the core plus one slab per seam in x; y contributes no wrapping
    @test narrayeqs_interior(sys_arr) == 3
end

@testset "Brusselator: coupled 2D system, periodic in both directions" begin
    @parameters x y t
    @variables u(..) v(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    ∇²(w) = Dxx(w) + Dyy(w)

    brusselator_f(x, y, t) = (((x - 0.3)^2 + (y - 0.6)^2) <= 0.1^2) * (t >= 1.1) * 5.0
    α = 10.0
    u0(x, y, t) = 22(y * (1 - y))^(3 / 2)
    v0(x, y, t) = 27(x * (1 - x))^(3 / 2)

    eqs = [
        Dt(u(x, y, t)) ~ 1.0 + v(x, y, t) * u(x, y, t)^2 - 4.4 * u(x, y, t) +
            α * ∇²(u(x, y, t)) + brusselator_f(x, y, t),
        Dt(v(x, y, t)) ~ 3.4 * u(x, y, t) - v(x, y, t) * u(x, y, t)^2 +
            α * ∇²(v(x, y, t)),
    ]
    domains = [
        x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0), t ∈ Interval(0.0, 2.0),
    ]
    bcs = [
        u(x, y, 0) ~ u0(x, y, 0),
        u(0, y, t) ~ u(1, y, t),
        u(x, 0, t) ~ u(x, 1, t),
        v(x, y, 0) ~ v0(x, y, 0),
        v(0, y, t) ~ v(1, y, t),
        v(x, 0, t) ~ v(x, 1, t),
    ]
    @named pdesys = PDESystem(eqs, bcs, domains, [x, y, t], [u(x, y, t), v(x, y, t)])

    # five interior equations per variable, as in the single variable 2D case
    counts = map([8, 16]) do n
        disc = MOLFiniteDifference(
            [x => 1 / n, y => 1 / n], t
        )
        sys, _ = symbolic_discretize(pdesys, disc)
        (; narr = narrayeqs_interior(sys), n = length(get_eqs(sys)))
    end
    @test counts[1].narr == 10
    @test counts[1].n == counts[2].n

    disc = MOLFiniteDifference([x => 1 / 8, y => 1 / 8], t)
    sol = solve(
        ode_discretize(pdesys, disc), Rodas4();
        reltol = 1.0e-10, abstol = 1.0e-10, saveat = 0.5
    )
    @test successful_retcode(sol)
end

@testset "1D two-domain interface, linear diffusion" begin
    # Taps that leave one domain land in the other variable's array. Each domain's
    # interior is one array equation plus the (stencil-wide) wrap points, a count
    # that does not grow with the grid.
    @parameters t x1 x2
    @variables c1(..) c2(..)
    Dt = Differential(t)
    Dx1 = Differential(x1)
    Dx2 = Differential(x2)
    Dxx1 = Dx1^2
    Dxx2 = Dx2^2

    eqs = [Dt(c1(t, x1)) ~ Dxx1(c1(t, x1)), Dt(c2(t, x2)) ~ Dxx2(c2(t, x2))]
    bcs = [
        c1(0, x1) ~ -x1 * (x1 - 1) * sin(x1),
        c2(0, x2) ~ -x2 * (x2 - 1) * sin(x2),
        c1(t, 0) ~ 0.0,
        c1(t, 0.5) ~ c2(t, 0.5),
        -Dx1(c1(t, 0.5)) ~ -Dx2(c2(t, 0.5)),
        c2(t, 1) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.1), x1 ∈ Interval(0.0, 0.5), x2 ∈ Interval(0.5, 1.0),
    ]
    @named pdesys = PDESystem(eqs, bcs, domains, [t, x1, x2], [c1(t, x1), c2(t, x2)])

    sol_arr, sys_arr = solve_discretized(pdesys, [x1 => 0.05, x2 => 0.05], t)
    @test SciMLBase.successful_retcode(sol_arr)
    @test narrayeqs_interior(sys_arr) == 2

    counts = map([0.05, 0.025]) do dx
        disc = MOLFiniteDifference([x1 => dx, x2 => dx], t)
        sys, _ = symbolic_discretize(pdesys, disc)
        length(get_eqs(sys))
    end
    @test counts[1] == counts[2]
end

@testset "1D nonlinear laplacian (u coefficient)" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    eq = Dt(u(t, x)) ~ Dx(u(t, x) * Dx(u(t, x)))
    bcs = [u(0, x) ~ 1.0 + sinpi(x) / 2, u(t, 0) ~ 1.0, u(t, 1) ~ 1.0]
    domains = [t ∈ Interval(0.0, 0.05), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    sol_arr, sys_arr = solve_discretized(pdesys, [x => 0.05], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    # The half-offset coefficient expression is translation invariant over the core, so
    # the interior collapses to one array equation (issue #623).
    @test narrayeqs_interior(sys_arr) == 1

    # and the equation count does not grow with the grid
    counts = map([21, 41]) do n
        disc = MOLFiniteDifference(
            [x => 1 / (n - 1)], t
        )
        sys, _ = symbolic_discretize(pdesys, disc)
        length(get_eqs(sys))
    end
    @test counts[1] == counts[2]
end

@testset "1D nonlinear laplacian against an analytic solution" begin
    # test/Nonlinear_Diffusion Test 00 (doi:10.1016/j.camwa.2006.12.077):
    # Dt(u) ~ Dx(u^-1 Dx(u)), exact solution 2(1 + t)/(1 + x)^2.
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    analytic(t, x) = 2.0 * (1.0 + t) / (1.0 + x)^2
    eq = Dt(u(t, x)) ~ Dx(u(t, x)^(-1) * Dx(u(t, x)))
    bcs = [
        u(0, x) ~ analytic(0.0, x),
        u(t, 0) ~ analytic(t, 0.0),
        u(t, 2) ~ analytic(t, 2.0),
    ]
    domains = [t ∈ Interval(0.0, 2.0), x ∈ Interval(0.0, 2.0)]
    @named pdesys = PDESystem([eq], bcs, domains, [t, x], [u(t, x)])

    sol_arr, sys_arr = solve_discretized(pdesys, [x => 0.04], t)
    @test SciMLBase.successful_retcode(sol_arr)
    @test narrayeqs_interior(sys_arr) == 1

    xdisc = sol_arr[x]
    exact = [analytic(sol_arr[t][end], xi) for xi in xdisc]
    @test sol_arr[u(t, x)][end, :] ≈ exact atol = 0.1
end

@testset "1D nonlinear laplacian, coefficient depending on x" begin
    # An independent variable in the coefficient becomes numerically interpolated grid
    # values at the half-offset points.
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    eq = Dt(u(t, x)) ~ Dx((1 + x) * u(t, x) * Dx(u(t, x)))
    bcs = [u(0, x) ~ 1.0 + sinpi(x) / 2, u(t, 0) ~ 1.0, u(t, 1) ~ 1.0]
    domains = [t ∈ Interval(0.0, 0.05), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    sol_arr, sys_arr = solve_discretized(pdesys, [x => 0.05], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
end

@testset "1D nonlinear laplacian, division forms" begin
    @parameters t x p
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    bcs = [u(0, x) ~ 1.0 + sinpi(x) / 2, u(t, 0) ~ 1.0, u(t, 1) ~ 1.0]
    domains = [t ∈ Interval(0.0, 0.05), x ∈ Interval(0.0, 1.0)]

    # A divided coefficient, Dx(Dx(u)/a(u))
    eq = Dt(u(t, x)) ~ Dx(Dx(u(t, x)) / u(t, x))
    @named pdesys_div = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
    sol_arr, sys_arr = solve_discretized(pdesys_div, [x => 0.05], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1

    # A grid-constant prefactor and a parameter divisor on the whole laplacian
    eq = Dt(u(t, x)) ~ 3.0 * Dx(u(t, x) * Dx(u(t, x))) / p
    @named pdesys_pre = PDESystem(eq, bcs, domains, [t, x], [u(t, x)], [p => 2.0])
    sol_arr, sys_arr = solve_discretized(pdesys_pre, [x => 0.05], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
end

@testset "1D nonlinear laplacian, grid-varying factor multiplying the laplacian" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    bcs = [u(0, x) ~ 1.0 + sinpi(x) / 2, u(t, 0) ~ 1.0, u(t, 1) ~ 1.0]
    domains = [t ∈ Interval(0.0, 0.05), x ∈ Interval(0.0, 1.0)]

    eqs = [
        # An independent variable factor
        Dt(u(t, x)) ~ x * Dx(u(t, x)^2 * Dx(u(t, x))),
        # A dependent variable factor
        Dt(u(t, x)) ~ u(t, x) * Dx(u(t, x)^2 * Dx(u(t, x))),
        # A factor on the divided-coefficient form
        Dt(u(t, x)) ~ x * Dx(Dx(u(t, x)) / u(t, x)),
        # A grid-varying factor together with a grid-varying divisor
        Dt(u(t, x)) ~ (1 + x) * Dx(u(t, x)^2 * Dx(u(t, x))) / (2 + x),
    ]
    for eq in eqs
        @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
        sol_arr, _ = solve_discretized(pdesys, [x => 0.05], t)
        @test SciMLBase.successful_retcode(sol_arr)
    end
end

@testset "1D nonlinear laplacian, fourth order approximation" begin
    # Band regression guard: at order 4 the half-offset operators are wider than the
    # central second difference `d_orders` reports, so the core must shrink accordingly.
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    eq = Dt(u(t, x)) ~ Dx(u(t, x) * Dx(u(t, x)))
    bcs = [u(0, x) ~ 1.0 + sinpi(x) / 2, u(t, 0) ~ 1.0, u(t, 1) ~ 1.0]
    domains = [t ∈ Interval(0.0, 0.05), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    sol_arr, sys_arr = solve_discretized(
        pdesys, [x => 0.05], t; disc_kwargs = (; approx_order = 4)
    )
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
end

@testset "1D nonlinear laplacian, periodic BCs" begin
    # In a periodic direction the core spans the whole interior except the points whose
    # slices would straddle the seam.
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    eq = Dt(u(t, x)) ~ Dx(u(t, x) * Dx(u(t, x)))
    bcs = [u(0, x) ~ 1.5 + sinpi(2x) / 2, u(t, 0) ~ u(t, 1)]
    domains = [t ∈ Interval(0.0, 0.05), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    sol_arr, sys_arr = solve_discretized(pdesys, [x => 0.05], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
end

@testset "1D nonlinear laplacian on a nonuniform grid" begin
    # Half-offset interior weights vary per point and enter as numeric coefficient
    # vectors.
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    eq = Dt(u(t, x)) ~ Dx(u(t, x) * Dx(u(t, x)))
    bcs = [u(0, x) ~ 1.0 + sinpi(x) / 2, u(t, 0) ~ 1.0, u(t, 1) ~ 1.0]
    domains = [t ∈ Interval(0.0, 0.05), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    gridvec = [0.5 * (1 - cospi(i / 20)) for i in 0:20]
    sol_arr, sys_arr = solve_discretized(pdesys, [x => gridvec], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
end

@testset "2D nonlinear laplacian" begin
    @parameters t x y
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dy = Differential(y)

    eq = Dt(u(t, x, y)) ~ Dx(u(t, x, y) * Dx(u(t, x, y))) +
        Dy(u(t, x, y) * Dy(u(t, x, y)))
    bcs = [
        u(0, x, y) ~ 1.0 + sinpi(x) * sinpi(y) / 2,
        u(t, 0, y) ~ 1.0, u(t, 1, y) ~ 1.0,
        u(t, x, 0) ~ 1.0, u(t, x, 1) ~ 1.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.025), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0),
    ]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])

    sol_arr, sys_arr = solve_discretized(pdesys, [x => 0.1, y => 0.1], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
end

@testset "1D spherical laplacian" begin
    # Cardinalization rewrites Dt(u) ~ Dr(r^2*Dr(u))/r^2 with a Mul numerator, the shape
    # the pointwise path discretizes through the nonlinear laplacian rules: r^2 enters at
    # the half-offset points, the outer r^-2 as a broadcast division by the grid values.
    @parameters t r
    @variables u(..)
    Dt = Differential(t)
    Dr = Differential(r)

    # MMS as in Diffusion Test 07: u = exp(-t)sin(r)/r, satisfies Dr(u(t, 0)) = 0.
    u_exact = (r, t) -> exp(-t) * sin(r) / r
    eq = Dt(u(t, r)) ~ Dr(r^2 * Dr(u(t, r))) / r^2
    bcs = [
        u(0, r) ~ sin(r) / r,
        Dr(u(t, 0)) ~ 0.0,
        u(t, 1) ~ exp(-t) * sin(1.0),
    ]
    domains = [t ∈ Interval(0.0, 1.0), r ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, r], [u(t, r)])

    sol_arr, sys_arr = solve_discretized(pdesys, [r => 0.1], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1

    rdisc = sol_arr[r][2:(end - 1)]
    for (i, ti) in enumerate(sol_arr[t])
        @test sol_arr[u(t, r)][i, 2:(end - 1)] ≈ u_exact.(rdisc, ti) atol = 0.05
    end

    # and the equation count does not grow with the grid
    counts = map([21, 41]) do n
        disc = MOLFiniteDifference(
            [r => 1 / (n - 1)], t
        )
        sys, _ = symbolic_discretize(pdesys, disc)
        length(get_eqs(sys))
    end
    @test counts[1] == counts[2]
end

@testset "1D spherical laplacian, fourth order approximation" begin
    @parameters t r
    @variables u(..)
    Dt = Differential(t)
    Dr = Differential(r)

    eq = Dt(u(t, r)) ~ Dr(r^2 * Dr(u(t, r))) / r^2
    bcs = [
        u(0, r) ~ sin(r) / r,
        Dr(u(t, 0)) ~ 0.0,
        u(t, 1) ~ exp(-t) * sin(1.0),
    ]
    domains = [t ∈ Interval(0.0, 1.0), r ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, r], [u(t, r)])

    sol_arr, sys_arr = solve_discretized(
        pdesys, [r => 0.1], t; disc_kwargs = (; approx_order = 4)
    )
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
end

@testset "1D spherical laplacian with a constant prefactor" begin
    # Diffusion Test 08: 4/r^2 * Dr(r^2 * Dr(u)), the prefactor rides along as a
    # grid-constant factor of the divided nonlinear laplacian.
    @parameters t r
    @variables u(..)
    Dt = Differential(t)
    Dr = Differential(r)

    u_exact = (r, t) -> exp(-4t) * sin(r) / r
    eq = Dt(u(t, r)) ~ 4 / r^2 * Dr(r^2 * Dr(u(t, r)))
    bcs = [
        u(0, r) ~ sin(r) / r,
        Dr(u(t, 0)) ~ 0.0,
        u(t, 1) ~ exp(-4t) * sin(1.0),
    ]
    domains = [t ∈ Interval(0.0, 1.0), r ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, r], [u(t, r)])

    sol_arr, sys_arr = solve_discretized(pdesys, [r => 0.1], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1

    rdisc = sol_arr[r][2:(end - 1)]
    for (i, ti) in enumerate(sol_arr[t])
        @test sol_arr[u(t, r)][i, 2:(end - 1)] ≈ u_exact.(rdisc, ti) atol = 0.05
    end
end

@testset "1D spherical laplacian, bare divided term uses the spherical scheme" begin
    # With the laplacian written on the lhs the cardinalized numerator stays bare (no
    # Mul), the one shape where the scalar spherical scheme wins over the nonlinear
    # laplacian rules; the slice form mirrors `spherical_diffusion`.
    @parameters t r
    @variables u(..)
    Dt = Differential(t)
    Dr = Differential(r)

    eq = Dr(r^2 * Dr(u(t, r))) / r^2 ~ Dt(u(t, r))
    bcs = [
        u(0, r) ~ sin(r) / r,
        Dr(u(t, 0)) ~ 0.0,
        u(t, 1) ~ exp(-t) * sin(1.0),
    ]
    domains = [t ∈ Interval(0.0, 1.0), r ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, r], [u(t, r)])

    sol_arr, sys_arr = solve_discretized(pdesys, [r => 0.1], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
end

@testset "1D spherical laplacian, grid-varying factor multiplying the laplacian" begin
    @parameters t r
    @variables u(..)
    Dt = Differential(t)
    Dr = Differential(r)

    bcs = [
        u(0, r) ~ sin(r) / r,
        Dr(u(t, 0)) ~ 0.0,
        u(t, 1) ~ exp(-t) * sin(1.0),
    ]
    domains = [t ∈ Interval(0.0, 1.0), r ∈ Interval(0.0, 1.0)]

    eqs = [
        # An independent variable factor
        Dt(u(t, r)) ~ (1 + r) * Dr(r^2 * Dr(u(t, r))) / r^2,
        # A dependent variable factor
        Dt(u(t, r)) ~ u(t, r) * Dr(r^2 * Dr(u(t, r))) / r^2,
    ]
    for eq in eqs
        @named pdesys = PDESystem(eq, bcs, domains, [t, r], [u(t, r)])
        sol_arr, _ = solve_discretized(pdesys, [r => 0.1], t)
        @test SciMLBase.successful_retcode(sol_arr)
    end
end

@testset "1D spherical laplacian on a nonuniform grid" begin
    # Bare form: the spherical scheme's centered first derivative takes per-point
    # numeric weights on nonuniform grids.
    @parameters t r
    @variables u(..)
    Dt = Differential(t)
    Dr = Differential(r)

    eq = Dr(r^2 * Dr(u(t, r))) / r^2 ~ Dt(u(t, r))
    bcs = [
        u(0, r) ~ sin(r) / r,
        Dr(u(t, 0)) ~ 0.0,
        u(t, 1) ~ exp(-t) * sin(1.0),
    ]
    domains = [t ∈ Interval(0.0, 1.0), r ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, r], [u(t, r)])

    gridvec = [0.5 * (1 - cospi(i / 20)) for i in 0:20]
    sol_arr, sys_arr = solve_discretized(pdesys, [r => gridvec], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
end

@testset "Fallback: grid-varying factor multiplying a nonlinear laplacian" begin
    # The pointwise path leaves such factors undiscretized (a pre-existing pointwise-path
    # bug), so no slice form can reproduce it; the equation stays pointwise for parity.
    # Symbolic level only: the scalar result does not simulate.
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    eq = Dt(u(t, x)) ~ x * Dx(u(t, x)^2 * Dx(u(t, x)))
    bcs = [u(0, x) ~ 1.0 + sinpi(x) / 2, u(t, 0) ~ 1.0, u(t, 1) ~ 1.0]
    domains = [t ∈ Interval(0.0, 0.05), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    lenient = MOLFiniteDifference(
        [x => 0.05], t
    )
    sys, _ = symbolic_discretize(pdesys, lenient)
    @test narrayeqs_interior(sys) == 0
end

@testset "WENO advection on a uniform grid" begin
    # The taps are fixed offsets, so the scheme traces once; the solution-dependent
    # weights broadcast elementwise.
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    eq = Dt(u(t, x)) ~ -Dx(u(t, x))
    bcs = [u(0, x) ~ exp(-100 * (x - 0.3)^2), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0]
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    sol_arr, sys_arr = solve_discretized(
        pdesys, [x => 0.02], t;
        disc_kwargs = (; advection_scheme = WENOScheme()), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-3)
    )
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1

    # the scheme is traced once, so the equation count is resolution independent
    counts = map([51, 101]) do n
        disc = MOLFiniteDifference(
            [x => 1 / (n - 1)], t; advection_scheme = WENOScheme(),

        )
        sys, _ = symbolic_discretize(pdesys, disc)
        length(get_eqs(sys))
    end
    @test counts[1] == counts[2]
end

@testset "WENO advection with a coefficient and diffusion" begin
    # The scheme replaces the bare `Dx(u)`; coefficients and other derivatives
    # broadcast on.
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    eq = Dt(u(t, x)) ~ -(1.0 + x) * Dx(u(t, x)) + 0.05 * Dxx(u(t, x))
    bcs = [u(0, x) ~ exp(-100 * (x - 0.3)^2), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0]
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    sol_arr, sys_arr = solve_discretized(
        pdesys, [x => 0.02], t;
        disc_kwargs = (; advection_scheme = WENOScheme()), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-4)
    )
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
end

@testset "WENO advection with periodic boundaries" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    eq = Dt(u(t, x)) ~ -Dx(u(t, x))
    bcs = [u(0, x) ~ sinpi(2x), u(t, 0) ~ u(t, 1)]
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    sol_arr, sys_arr = solve_discretized(
        pdesys, [x => 0.02], t;
        disc_kwargs = (; advection_scheme = WENOScheme()), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-3)
    )
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    # one array equation for the points whose taps do not wrap, the rest pointwise
    @test narrayeqs_interior(sys_arr) == 1

    # the number of wrap points is fixed by the stencil, so the equation count is
    # resolution independent
    counts = map([51, 101]) do n
        disc = MOLFiniteDifference(
            [x => 1 / (n - 1)], t; advection_scheme = WENOScheme(),

        )
        sys, _ = symbolic_discretize(pdesys, disc)
        length(get_eqs(sys))
    end
    @test counts[1] == counts[2]

    # a periodic nonuniform direction goes through the coefficient split: seam windows
    # take the periodically shifted coordinates `bcoord` produces, so parity is bitwise
    gridvec = [0.5 * (1 - cospi(i / 50)) for i in 0:50]
    solp_arr, sysp_arr = solve_discretized(
        pdesys, [x => gridvec], t;
        disc_kwargs = (; advection_scheme = WENOScheme()), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-3)
    )
    @test solp_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sysp_arr) == 1

    # resolution independence holds on the periodic nonuniform path too
    pcounts = map([40, 80]) do n
        disc = MOLFiniteDifference(
            [x => [0.5 * (1 - cospi(i / n)) for i in 0:n]], t;
            advection_scheme = WENOScheme()
        )
        sys, _ = symbolic_discretize(pdesys, disc)
        length(get_eqs(sys))
    end
    @test pcounts[1] == pcounts[2]
end

@testset "WENO advection on a nonuniform grid (coefficient split)" begin
    # Coordinate arithmetic sits in numeric per-point slots (`array_scheme_split`), so
    # the kernel traces once and the solutions agree bitwise, like the uniform case.
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    eq = Dt(u(t, x)) ~ -Dx(u(t, x))
    bcs = [u(0, x) ~ exp(-100 * (x - 0.3)^2), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0]
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    gridvec = [0.5 * (1 - cospi(i / 40)) for i in 0:40]
    sol_arr, sys_arr = solve_discretized(
        pdesys, [x => gridvec], t;
        disc_kwargs = (; advection_scheme = WENOScheme()), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-4)
    )
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1

    # the kernel is traced once, so the equation count is resolution independent
    counts = map([40, 80]) do n
        disc = MOLFiniteDifference(
            [x => [0.5 * (1 - cospi(i / n)) for i in 0:n]], t;
            advection_scheme = WENOScheme(),

        )
        sys, _ = symbolic_discretize(pdesys, disc)
        length(get_eqs(sys))
    end
    @test counts[1] == counts[2]
end

@testset "Nonuniform WENO coefficient split reproduces the scalar trace" begin
    # Acceptance criterion of the split: substituting the numeric slots into the traced
    # kernel must be symbolically identical (`isequal`) to the scalar core's trace on
    # the same window — identical expressions compile to identical code, hence bitwise
    # parity. Breaks visibly if SymbolicUtils canonicalization ever changes.
    ε = 1.0e-6
    usyms = [Symbolics.variable(:u, i) for i in 1:5]
    split = MethodOfLines.array_scheme_split(WENOScheme())
    @test split !== nothing
    csyms = [Symbolics.variable(:c, i) for i in 1:split.nslots]
    traced = split.apply(usyms, [ε], 0.0, csyms)

    # Deterministic grid-window families: smooth cosine stretching, geometric
    # stretching, jittered spacing, abrupt step-size changes.
    function window(family, trial)
        h = 0.03
        steps = if family == 1
            full = [0.5 * (1 - cospi(i / 24)) for i in 0:24]
            return full[trial:(trial + 4)]
        elseif family == 2
            r = 1.05 + 0.1 * trial
            [h * r^k for k in 0:3]
        elseif family == 3
            [h * (1 + 0.4 * sin(3.7 * k + trial)) for k in 0:3]
        else
            [isodd(k + trial) ? h : (1 + trial / 2) * h for k in 0:3]
        end
        return vcat(0.0, cumsum(steps)) .+ 0.01 * trial
    end
    for family in 1:4, trial in 1:10
        x = window(family, trial)
        @assert all(>(0), diff(x))
        c = split.coeffs(x)
        applied = Symbolics.substitute(traced, Dict(csyms .=> c))
        core = MethodOfLines._weno_f_nonuniform_core(usyms, ε, x, Val(3))
        @test isequal(Symbolics.unwrap(applied), Symbolics.unwrap(core))

        # and the kernels evaluate numerically like the core
        uvals = [sin(1.3 * k + family) + 0.01 * trial for k in 1:5]
        a = split.apply(uvals, [ε], 0.0, c)
        b = MethodOfLines._weno_f_nonuniform_core(uvals, ε, x, Val(3))
        @test isapprox(a, b; rtol = 1.0e-13)
    end

    # seam windows of a periodic direction: exactly the coordinates
    # `array_periodic_coord` hands to `coeffs` near either end
    n = 17
    pgrid = [0.5 * (1 - cospi(i / (n - 1))) for i in 0:(n - 1)] .+ 0.25
    for center in (1, 2, 3, n - 1, n)
        x = [MethodOfLines.array_periodic_coord(pgrid, center + k, n) for k in -2:2]
        @assert all(>(0), diff(x))
        c = split.coeffs(x)
        applied = Symbolics.substitute(traced, Dict(csyms .=> c))
        core = MethodOfLines._weno_f_nonuniform_core(usyms, ε, x, Val(3))
        @test isequal(Symbolics.unwrap(applied), Symbolics.unwrap(core))
    end
end

@testset "Nonuniform WENO: diffusion, coupling, minimal grid" begin
    @parameters t x
    @variables u(..) v(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)]
    cosgrid(n) = [0.5 * (1 - cospi(i / n)) for i in 0:n]

    # WENO and a nonuniform central difference in the same equation
    @named advdiff = PDESystem(
        Dt(u(t, x)) ~ -Dx(u(t, x)) + 0.05 * Dxx(u(t, x)),
        [u(0, x) ~ exp(-100 * (x - 0.3)^2), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0],
        domains, [t, x], [u(t, x)]
    )
    sol_arr, sys_arr = solve_discretized(
        advdiff, [x => cosgrid(40)], t;
        disc_kwargs = (; advection_scheme = WENOScheme()), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-4)
    )
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1

    # coupled system: one trace per equation; coupling terms reorder one addition, so
    # parity is to reassociation level (measured ~1e-18) rather than bitwise
    @named coupled = PDESystem(
        [
            Dt(u(t, x)) ~ -Dx(u(t, x)) + 0.1 * v(t, x),
            Dt(v(t, x)) ~ -Dx(v(t, x)) - 0.1 * u(t, x),
        ],
        [
            u(0, x) ~ exp(-100 * (x - 0.3)^2), v(0, x) ~ exp(-100 * (x - 0.6)^2),
            u(t, 0) ~ 0.0, u(t, 1) ~ 0.0, v(t, 0) ~ 0.0, v(t, 1) ~ 0.0,
        ],
        domains, [t, x], [u(t, x), v(t, x)]
    )
    solc_arr, sysc_arr = solve_discretized(
        coupled, [x => cosgrid(40)], t;
        disc_kwargs = (; advection_scheme = WENOScheme()), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-4)
    )
    @test solc_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sysc_arr) == 2

    # smallest representable grid (n = 7: three-point core, frame at 2 and 6)
    @named adv = PDESystem(
        Dt(u(t, x)) ~ -Dx(u(t, x)),
        [u(0, x) ~ exp(-100 * (x - 0.3)^2), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0],
        domains, [t, x], [u(t, x)]
    )
    sol7_arr, sys7_arr = solve_discretized(
        adv, [x => cosgrid(6)], t;
        disc_kwargs = (; advection_scheme = WENOScheme()), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-4)
    )
    @test sol7_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys7_arr) == 1
end

@testset "Periodic nonuniform WENO: minimal grid, guards" begin
    @parameters t x y
    @variables u(..) w(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2
    Dy = Differential(y)
    cosgrid(n) = [0.5 * (1 - cospi(i / n)) for i in 0:n]
    dom1 = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)]

    # Multi-dimensional systems eagerly build mixed-derivative stencils in the scalar
    # path, which reject nonuniform interfaces; the array path builds mixed rules only
    # for the mixed terms present in the equation, so it covers this system while the
    # pointwise path has no reference to match. Accuracy is checked against the exact
    # advection solution instead.
    dom2 = [t ∈ Interval(0.0, 0.05), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]
    bcs2 = [
        w(0, x, y) ~ sinpi(2x) * exp(-100 * (y - 0.4)^2),
        w(t, 0, y) ~ w(t, 1, y), w(t, x, 0) ~ 0.0, w(t, x, 1) ~ 0.0,
    ]
    @named pdesys2 = PDESystem(
        Dt(w(t, x, y)) ~ -Dx(w(t, x, y)) - Dy(w(t, x, y)),
        bcs2, dom2, [t, x, y], [w(t, x, y)]
    )
    sol2, sys2 = solve_discretized(
        pdesys2, [x => cosgrid(20), y => 0.05], t;
        disc_kwargs = (; advection_scheme = WENOScheme()), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-4)
    )
    interior2 = filter(isinterioreq, get_eqs(sys2))
    @test !isempty(interior2) && all(isarrayeq, interior2)
    @test sol2.retcode == SciMLBase.ReturnCode.Success
    xs2, ys2, T2 = sol2[x], sol2[y], sol2[t][end]
    exact2 = [
        sinpi(2 * (xi - T2)) * exp(-100 * (yi - T2 - 0.4)^2) for xi in xs2, yi in ys2
    ]
    @test maximum(abs.(sol2[w(t, x, y)][end, :, :] .- exact2)) < 0.1

    # a mixed term reaching along the periodic nonuniform direction has no slice form
    @named pdesys2m = PDESystem(
        Dt(w(t, x, y)) ~ -Dx(w(t, x, y)) - Dy(w(t, x, y)) + 0.01 * Dx(Dy(w(t, x, y))),
        bcs2, dom2, [t, x, y], [w(t, x, y)]
    )
    @test_throws AssertionError symbolic_discretize(
        pdesys2m,
        MOLFiniteDifference(
            [x => cosgrid(20), y => 0.05], t; advection_scheme = WENOScheme()
        )
    )

    # smallest grid the boundary extrapolator admits (7 points); every window wraps
    @named adv = PDESystem(
        Dt(u(t, x)) ~ -Dx(u(t, x)),
        [u(0, x) ~ sinpi(2x), u(t, 0) ~ u(t, 1)],
        dom1, [t, x], [u(t, x)]
    )
    sol7_arr, sys7_arr = solve_discretized(
        adv, [x => cosgrid(6)], t;
        disc_kwargs = (; advection_scheme = WENOScheme()), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-4)
    )
    @test sol7_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys7_arr) == 1

    # linear stencils still have no seam form on a nonuniform grid
    @named advdiffp = PDESystem(
        Dt(u(t, x)) ~ -Dx(u(t, x)) + 0.05 * Dxx(u(t, x)),
        [u(0, x) ~ sinpi(2x), u(t, 0) ~ u(t, 1)],
        dom1, [t, x], [u(t, x)]
    )
    @test_throws AssertionError symbolic_discretize(
        advdiffp,
        MOLFiniteDifference(
            [x => cosgrid(20)], t; advection_scheme = WENOScheme()
        )
    )
end

@testset "WENO advection in multi-dimensional equations" begin
    # Each direction is traced once (`array_advection_rules`), so a 2D equation carries
    # two traces in one array equation.
    @parameters t x y
    @variables w(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dy = Differential(y)
    dom2 = [t ∈ Interval(0.0, 0.05), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]
    bcs2 = [
        w(0, x, y) ~ exp(-100 * ((x - 0.4)^2 + (y - 0.4)^2)),
        w(t, 0, y) ~ 0.0, w(t, 1, y) ~ 0.0, w(t, x, 0) ~ 0.0, w(t, x, 1) ~ 0.0,
    ]
    @named pdesys = PDESystem(
        Dt(w(t, x, y)) ~ -Dx(w(t, x, y)) - Dy(w(t, x, y)),
        bcs2, dom2, [t, x, y], [w(t, x, y)]
    )
    sol_arr, sys_arr = solve_discretized(
        pdesys, [x => 0.05, y => 0.05], t;
        disc_kwargs = (; advection_scheme = WENOScheme()), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-3)
    )
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1

    # mixed grid: x traces the uniform kernel, y goes through the coefficient split
    gridvec = [0.5 * (1 - cospi(i / 20)) for i in 0:20]
    solm_arr, sysm_arr = solve_discretized(
        pdesys, [x => 0.05, y => gridvec], t;
        disc_kwargs = (; advection_scheme = WENOScheme()), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-3)
    )
    @test solm_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sysm_arr) == 1

    # a trace and central differences in one 2D equation
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    @named advdiff2d = PDESystem(
        Dt(w(t, x, y)) ~ -Dx(w(t, x, y)) - Dy(w(t, x, y)) +
            0.05 * (Dxx(w(t, x, y)) + Dyy(w(t, x, y))),
        bcs2, dom2, [t, x, y], [w(t, x, y)]
    )
    sold_arr, sysd_arr = solve_discretized(
        advdiff2d, [x => 0.05, y => 0.05], t;
        disc_kwargs = (; advection_scheme = WENOScheme()), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-3)
    )
    @test sold_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sysd_arr) == 1

end

@testset "User defined functional advection schemes" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    eq = Dt(u(t, x)) ~ -Dx(u(t, x))
    bcs = [u(0, x) ~ exp(-100 * (x - 0.3)^2), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0]
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    # A scheme of the taps alone traces into one array expression like WENO does.
    central3(u, p, t, x, dx) = (u[3] - u[1]) / (2dx)
    scheme = FunctionalScheme{3, 1}(
        central3, [nothing], [nothing], false, []; name = "central3"
    )
    sol_arr, sys_arr = solve_discretized(
        pdesys, [x => 0.02], t;
        disc_kwargs = (; advection_scheme = scheme), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-3)
    )
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1

    # A scheme that reads the grid coordinate falls back: a trace cannot reproduce the
    # pointwise path's Float64 coordinate folds digit for digit.
    coord3(u, p, t, x, dx) = (u[3] - u[1]) / (x[3] - x[1])
    xscheme = FunctionalScheme{3, 1}(
        coord3, [nothing], [nothing], false, []; name = "coord3"
    )
    sol_arr2, sys_arr2 = solve_discretized(
        pdesys, [x => 0.02], t;
        disc_kwargs = (; advection_scheme = xscheme), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-3)
    )
    @test sol_arr2.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr2) == 0

    # A nonuniform scheme without an `array_scheme_split` falls back; the pointwise
    # result is untouched.
    nu3(u, p, t, x, dx) = (u[3] - u[1]) / (x[3] - x[1])
    nuscheme = FunctionalScheme{3, 1}(
        nu3, [nothing], [nothing], true, []; name = "nu3"
    )
    gridvec = [0.5 * (1 - cospi(i / 40)) for i in 0:40]
    sol_arr3, sys_arr3 = solve_discretized(
        pdesys, [x => gridvec], t;
        disc_kwargs = (; advection_scheme = nuscheme), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-4)
    )
    @test sol_arr3.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr3) == 0

    # A trace failure (hard branch on coordinates, which are symbols while tracing)
    # falls back, not errors.
    xbranch(u, p, t, x, dx) = x[3] > 0.5 ? (u[3] - u[2]) / dx : (u[2] - u[1]) / dx
    bscheme = FunctionalScheme{3, 1}(
        xbranch, [nothing], [nothing], false, []; name = "xbranch"
    )
    sol_arr4, sys_arr4 = solve_discretized(
        pdesys, [x => 0.02], t;
        disc_kwargs = (; advection_scheme = bscheme), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-3)
    )
    @test sol_arr4.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr4) == 0

    # A time-dependent flux traces fine: `t` stays symbolic in the array equation.
    tflux(u, p, t, x, dx) = (1 + 0.1 * t) * (u[3] - u[1]) / (2 * dx)
    tscheme = FunctionalScheme{3, 1}(
        tflux, [nothing], [nothing], false, []; name = "tflux"
    )
    sol_arr5, sys_arr5 = solve_discretized(
        pdesys, [x => 0.02], t;
        disc_kwargs = (; advection_scheme = tscheme), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-3)
    )
    @test sol_arr5.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr5) == 1
end

@testset "WENO advection of a nonlinear flux (Burgers)" begin
    # The scheme replaces the bare `Dx(u)`; the multiplying `u` broadcasts on. The
    # product reorders one multiplication, so parity is to one ulp rather than bitwise.
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    eq = Dt(u(t, x)) ~ -u(t, x) * Dx(u(t, x)) + 0.05 * Dxx(u(t, x))
    bcs = [u(0, x) ~ exp(-100 * (x - 0.3)^2), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0]
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    sol_arr, sys_arr = solve_discretized(
        pdesys, [x => 0.02], t;
        disc_kwargs = (; advection_scheme = WENOScheme()), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-4)
    )
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
end

@testset "Stationary systems still produce a NonlinearProblem" begin
    @parameters x
    @variables u(..)
    Dxx = Differential(x)^2

    eq = Dxx(u(x)) ~ -sinpi(x) * pi^2
    bcs = [u(0) ~ 0.0, u(1) ~ 0.0]
    domains = [x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem([eq], bcs, domains, [x], [u(x)])

    disc = MOLFiniteDifference([x => 0.05])
    prob = discretize(pdesys, disc)
    sol = solve(prob, NewtonRaphson())
    xs = sol[x]
    @test sol[u(x)] ≈ sinpi.(xs) atol = 1.0e-2
    @test all(isfinite, sol[u(x)])
end

@testset "Fallback: array-valued dependent variables" begin
    # `@variables u(..)[1:n]` discretizes to a nested getindex, whose immediate parent is
    # a component rather than a grid-shaped array. This must fall back to the pointwise
    # path rather than erroring: the array strategy may never turn a system the scalar
    # path can discretize into a failure.
    n_comp = 2
    @parameters t x p[1:n_comp] q[1:n_comp]
    @variables u(..)[1:n_comp]
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eqs = [Dt(u(t, x)[i]) ~ p[i] * Dxx(u(t, x)[i]) for i in 1:n_comp]
    bcs = reduce(
        vcat,
        [
            [
                    u(0, x)[i] ~ q[i] * cos(x),
                    u(t, 0)[i] ~ sin(t),
                    u(t, 1)[i] ~ exp(-t) * cos(1),
                ] for i in 1:n_comp
        ]
    )
    domains = [t ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(
        eqs, bcs, domains, [t, x], [u(t, x)[i] for i in 1:n_comp], [p, q];
        initial_conditions = Dict(p => [1.5, 2.0], q => [1.2, 1.8])
    )

    disc = MOLFiniteDifference([x => 0.1], t)
    prob_arr = ode_discretize(pdesys, disc)
    sol_arr = solve(prob_arr, Rodas4(); reltol = 1.0e-10, abstol = 1.0e-10, saveat = 0.2)
    @test SciMLBase.successful_retcode(sol_arr)
    for i in 1:n_comp
        @test all(isfinite, sol_arr[u(t, x)[i]])
    end
end

# Does any subexpression of `eq` apply the operation `f`, either directly or broadcast
# over slices (`broadcast(f, ...)`, which is how the array form applies it)? In the
# broadcast form the applied function is carried as a symbolic wrapper rather than the
# function itself, so it is matched by name.
function hasoperation(eq, f)
    names_f(x) = x === f || string(x) == string(f)
    function walk(x)
        x = Symbolics.unwrap(x)
        SymbolicUtils.iscall(x) || return false
        op = SymbolicUtils.operation(x)
        names_f(op) && return true
        args = SymbolicUtils.arguments(x)
        op === broadcast && !isempty(args) && names_f(first(args)) && return true
        return any(walk, args)
    end
    return walk(eq.lhs) || walk(eq.rhs)
end

@testset "Winding form: ifelse for coefficients constant over the grid" begin
    # The pointwise path emits `ifelse(coef > 0, coef*pos, coef*neg)`. When the coefficient
    # does not vary over the grid the wind direction is a single scalar condition, so the
    # array path must emit that same `ifelse` rather than the `max`/`min` surrogate (the
    # two differ when the unselected stencil is non-finite).
    @parameters t x vel
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)]
    bcs = [u(0, x) ~ exp(-100 * (x - 0.3)^2) + 1.0, u(t, 0) ~ 1.0, u(t, 1) ~ 1.0]

    for (name, advection, ps) in [
            ("literal", -2.0 * Dx(u(t, x)), Num[]),
            ("time dependent", -(1 + t) * Dx(u(t, x)), Num[]),
        ]
        eq = Dt(u(t, x)) ~ advection + 0.05 * Dxx(u(t, x))
        @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)], ps)
        sol_arr, sys_arr = solve_discretized(pdesys, [x => 0.05], t)
        @test any(eq -> hasoperation(eq, ifelse), get_eqs(sys_arr))
        @test narrayeqs_interior(sys_arr) == 1
    end

    # A coefficient that varies over the grid still needs the per-point surrogate, since
    # `ifelse` cannot broadcast over a symbolic array condition.
    eq = Dt(u(t, x)) ~ -(1 + x) * Dx(u(t, x)) + 0.05 * Dxx(u(t, x))
    @named pdesys_x = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
    sol_arr, sys_arr = solve_discretized(pdesys_x, [x => 0.05], t)
    @test !any(eq -> hasoperation(eq, ifelse), get_eqs(sys_arr))
end

@testset "Interior representation is independent of grid resolution" begin
    # One interior equation whose expression size does not grow with the grid.
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq = Dt(u(t, x)) ~ Dxx(u(t, x))
    bcs = [u(0, x) ~ sinpi(x), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0]
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    # count nodes in an expression tree
    function treesize(x)
        x = Symbolics.unwrap(x)
        SymbolicUtils.iscall(x) || return 1
        return 1 + sum(treesize, SymbolicUtils.arguments(x); init = 0)
    end
    interior_size(sys) = sum(
        treesize(eq.lhs) + treesize(eq.rhs)
            for eq in get_eqs(sys) if occursin("Differential", string(eq));
        init = 0
    )

    sizes = map([21, 81]) do n
        disc = MOLFiniteDifference([x => 1 / (n - 1)], t)
        sys, _ = symbolic_discretize(pdesys, disc)
        (; n, size = interior_size(sys), narr = narrayeqs_interior(sys))
    end

    coarse, fine = sizes
    # one array equation regardless of resolution
    @test coarse.narr == 1
    @test fine.narr == 1
    # array interior expression does not grow with the grid
    @test fine.size == coarse.size
end

@testset "unsupported patterns fall back to pointwise equations" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)]
    cases = [
        (
            Dt(u(t, x)) ~ Dxx(u(t, x)) + u(t, 0.5),
            [u(0, x) ~ sinpi(x), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0],
        ),
        (
            Dt(u(t, x)) ~ Dxx(u(t, x)) + Dx(u(t, 1)),
            [u(0, x) ~ sinpi(x), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0],
        ),
        (
            Dt(u(t, x)) ~ Dxx(u(t, x)) + u(0, x),
            [u(0, x) ~ sinpi(x), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0],
        ),
    ]
    for (eq, bcs) in cases
        @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
        sys, _ = symbolic_discretize(pdesys, MOLFiniteDifference([x => 0.1], t))
        @test narrayeqs_interior(sys) == 0
    end
end

@testset "Boundary conditions collapse to one equation per face" begin
    # Boundaries on a face are sliceable for the same reason the interior is: the index
    # along the boundary's own direction is fixed across the face, so every point there
    # selects the same stencil. Without this the boundary stays pointwise and dominates
    # the equation count in 2D/3D, where it scales with the surface.
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

    counts = map([8, 16]) do n
        disc = MOLFiniteDifference(
            [x => 1 / (n - 1), y => 1 / (n - 1)], t
        )
        sys, _ = symbolic_discretize(pdesys, disc)
        length(get_eqs(sys))
    end
    # total equation count is independent of resolution in 2D
    @test counts[1] == counts[2]
    # one interior equation plus one per face
    @test counts[1] <= 12

    disc = MOLFiniteDifference([x => 0.1, y => 0.1], t)
    sol = solve(
        ode_discretize(pdesys, disc), Rodas4();
        reltol = 1.0e-10, abstol = 1.0e-10, saveat = 0.025
    )
    @test successful_retcode(sol)
end

@testset "Boundary value in interior equation (1D both edges)" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq = Dt(u(t, x)) ~ Dxx(u(t, x)) + u(t, 0) - u(t, 1)
    bcs = [u(0, x) ~ sinpi(x), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0]
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    sol_arr, sys_arr = solve_discretized(pdesys, [x => 0.05], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
    # dx = 0.05 → 21 points: u(t, 0) → [1], u(t, 1) → [21]
    int_eq = only(filter(eq -> isinterioreq(eq) && isarrayeq(eq), get_eqs(sys_arr)))
    int_str = string(int_eq)
    @test occursin("[1]", int_str)
    @test occursin("[21]", int_str)

    treesize(x) = let u = Symbolics.unwrap(x)
        SymbolicUtils.iscall(u) ? 1 + sum(treesize, SymbolicUtils.arguments(u); init = 0) : 1
    end
    counts = map([11, 41]) do n
        disc = MOLFiniteDifference(
            [x => 1 / (n - 1)], t
        )
        sys, _ = symbolic_discretize(pdesys, disc)
        int = only(filter(isinterioreq, get_eqs(sys)))
        (narrayeqs_interior(sys), treesize(int.lhs) + treesize(int.rhs))
    end
    @test counts[1][1] == counts[2][1] == 1
    @test counts[1][2] == counts[2][2]
end

@testset "Boundary value with nonzero Dirichlet actually drives the interior" begin
    # Nonzero BC: bitwise RHS proves substitution. Trajectory checks use rtol like the rest
    # of this file; array vs scalar codegen is not bit-identical across Rosenbrock stages.
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq = Dt(u(t, x)) ~ Dxx(u(t, x)) + u(t, 1)
    bcs = [u(0, x) ~ 0.0, u(t, 0) ~ 0.0, u(t, 1) ~ exp(-t)]
    domains = [t ∈ Interval(0.0, 0.2), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    dxs = [x => 0.05]
    disc = MOLFiniteDifference(dxs, t)
    sys_arr, _ = symbolic_discretize(pdesys, disc)
    prob_arr = ode_discretize(pdesys, disc)

    @test narrayeqs_interior(sys_arr) == 1
    int_eq = only(filter(eq -> isinterioreq(eq) && isarrayeq(eq), get_eqs(sys_arr)))
    @test occursin("[21]", string(int_eq))

    du_arr = similar(prob_arr.u0)
    prob_arr.f(du_arr, prob_arr.u0, prob_arr.p, 0.0)
    @test maximum(abs.(du_arr)) > 1

    sol_arr, _ = solve_discretized(pdesys, dxs, t)
    @test SciMLBase.successful_retcode(sol_arr)
    @test maximum(abs.(sol_arr[u(t, x)])) > 0.05

    @parameters y
    Dyy = Differential(y)^2
    eq2 = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y)) + u(t, 0, y)
    bcs2 = [
        u(0, x, y) ~ 0.0,
        u(t, 0, y) ~ sinpi(y), u(t, 1, y) ~ 0.0,
        u(t, x, 0) ~ 0.0, u(t, x, 1) ~ 0.0,
    ]
    domains2 = [
        t ∈ Interval(0.0, 0.05), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0),
    ]
    @named pdesys2 = PDESystem(eq2, bcs2, domains2, [t, x, y], [u(t, x, y)])
    dxs2 = [x => 0.1, y => 0.1]
    disc2 = MOLFiniteDifference(dxs2, t)
    sys2, _ = symbolic_discretize(pdesys2, disc2)
    prob2_arr = ode_discretize(pdesys2, disc2)

    @test narrayeqs_interior(sys2) == 1
    face_eq = only(filter(eq -> isinterioreq(eq) && isarrayeq(eq), get_eqs(sys2)))
    @test occursin("1:1", string(face_eq))

    du2_arr = similar(prob2_arr.u0)
    prob2_arr.f(du2_arr, prob2_arr.u0, prob2_arr.p, 0.0)
    @test maximum(abs.(du2_arr)) > 1

    sol2_arr, _ = solve_discretized(pdesys2, dxs2, t)
    @test SciMLBase.successful_retcode(sol2_arr)
    @test maximum(abs.(sol2_arr[u(t, x, y)])) > 0.05
end

@testset "Boundary value in interior equation (2D face slice and corner)" begin
    @parameters t x y
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    domains = [
        t ∈ Interval(0.0, 0.05), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0),
    ]
    bcs = [
        u(0, x, y) ~ sinpi(x) * sinpi(y),
        u(t, 0, y) ~ 0.0, u(t, 1, y) ~ 0.0,
        u(t, x, 0) ~ 0.0, u(t, x, 1) ~ 0.0,
    ]

    @named face_sys = PDESystem(
        Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y)) + u(t, 0, y),
        bcs, domains, [t, x, y], [u(t, x, y)]
    )
    sol_arr, sys_arr = solve_discretized(face_sys, [x => 0.1, y => 0.1], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
    face_eq = only(filter(eq -> isinterioreq(eq) && isarrayeq(eq), get_eqs(sys_arr)))
    @test occursin("1:1", string(face_eq))

    # Free-standing corner: scalar boundaryvalfuncs leave u(t,0,0) symbolic; array does not.
    # No scalar parity — prove the array path is runnable and drives the interior.
    corner_bcs = [
        u(0, x, y) ~ 0.0,
        u(t, 0, y) ~ 1.0, u(t, 1, y) ~ 0.0,
        u(t, x, 0) ~ 1.0, u(t, x, 1) ~ 0.0,
    ]
    @named corner_sys = PDESystem(
        Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y)) + u(t, 0, 0),
        corner_bcs, domains, [t, x, y], [u(t, x, y)]
    )
    dxs_c = [x => 0.1, y => 0.1]
    disc = MOLFiniteDifference(
        dxs_c, t
    )
    sys_corner, _ = symbolic_discretize(corner_sys, disc)
    @test narrayeqs_interior(sys_corner) == 1
    corner_eq = only(filter(eq -> isinterioreq(eq) && isarrayeq(eq), get_eqs(sys_corner)))
    corner_str = string(corner_eq)
    @test occursin("[1, 1]", corner_str) || occursin("[1,1]", corner_str)
    @test !occursin("u(t, 0, 0)", corner_str) && !occursin("u(t,0,0)", corner_str)
    for eq in get_eqs(sys_corner)
        s = string(eq)
        @test !occursin("u(t, 0, 0)", s) && !occursin("u(t,0,0)", s)
    end

    prob_c = ode_discretize(corner_sys, disc)
    du_c = similar(prob_c.u0)
    prob_c.f(du_c, prob_c.u0, prob_c.p, 0.0)
    @test maximum(abs.(du_c)) > 1
    sol_c = solve(prob_c, Rodas4())
    @test SciMLBase.successful_retcode(sol_c)
    @test maximum(abs.(sol_c[u(t, x, y)])) > 0.05
end

@testset "Boundary value of a coupled variable in the interior" begin
    @parameters t x
    @variables u(..) v(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eqs = [
        Dt(u(t, x)) ~ Dxx(u(t, x)) + v(t, 1),
        Dt(v(t, x)) ~ Dxx(v(t, x)) - u(t, x),
    ]
    bcs = [
        u(0, x) ~ sinpi(x), v(0, x) ~ 0.0,
        u(t, 0) ~ 0.0, u(t, 1) ~ 0.0,
        v(t, 0) ~ 0.0, v(t, 1) ~ 0.0,
    ]
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eqs, bcs, domains, [t, x], [u(t, x), v(t, x)])

    sol_arr, sys_arr = solve_discretized(pdesys, [x => 0.05], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 2
end

@testset "Boundary value in interior on a nonuniform grid" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq = Dt(u(t, x)) ~ Dxx(u(t, x)) + u(t, 1)
    bcs = [u(0, x) ~ sinpi(x), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0]
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    gridvec = [0.5 * (1 - cospi(i / 20)) for i in 0:20]
    sol_arr, sys_arr = solve_discretized(pdesys, [x => gridvec], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
end

@testset "Boundary value as upwind coefficient stays in array form" begin
    # Pointwise path also leaves u(t,1) symbolic inside winding coefficients, so no parity.
    # Array must substitute via winding coefctx and remain runnable.
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    eq = Dt(u(t, x)) ~ -u(t, 1) * Dx(u(t, x))
    bcs = [u(0, x) ~ sinpi(x), u(t, 0) ~ 0.0, u(t, 1) ~ exp(-t)]
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    dxs = [x => 0.05]
    disc = MOLFiniteDifference(
        dxs, t
    )
    sys, _ = symbolic_discretize(pdesys, disc)
    @test narrayeqs_interior(sys) == 1
    wind_eq = only(filter(eq -> isinterioreq(eq) && isarrayeq(eq), get_eqs(sys)))
    wind_str = string(wind_eq)
    @test occursin("[21]", wind_str)
    for e in get_eqs(sys)
        s = string(e)
        @test !occursin("u(t, 1)", s) && !occursin("u(t,1)", s)
    end

    prob = ode_discretize(pdesys, disc)
    du = similar(prob.u0)
    prob.f(du, prob.u0, prob.p, 0.0)
    @test all(isfinite, du)
    @test maximum(abs.(du)) > 0
    sol = solve(prob, Rodas4())
    @test SciMLBase.successful_retcode(sol)
end

@testset "Boundary value in a periodic direction stays in array form" begin
    # Scalar boundaryvalfuncs skip interface faces, so no scalar parity here.
    # Prove every wrap/core equation substitutes u(t,0) and the ODE is runnable.
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq = Dt(u(t, x)) ~ Dxx(u(t, x)) + u(t, 0)
    bcs = [u(0, x) ~ sinpi(2x), u(t, 0) ~ u(t, 1)]
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    dxs = [x => 0.05]
    disc = MOLFiniteDifference(
        dxs, t
    )
    sys, _ = symbolic_discretize(pdesys, disc)
    @test narrayeqs_interior(sys) == 1
    per_eq = only(filter(eq -> isinterioreq(eq) && isarrayeq(eq), get_eqs(sys)))
    @test occursin("[1]", string(per_eq))
    for e in get_eqs(sys)
        s = string(e)
        @test !occursin("u(t, 0)", s) && !occursin("u(t,0)", s)
    end

    prob = ode_discretize(pdesys, disc)
    du = similar(prob.u0)
    prob.f(du, prob.u0, prob.p, 0.0)
    @test all(isfinite, du)
    @test maximum(abs.(du)) > 0
    sol = solve(prob, Rodas4())
    @test SciMLBase.successful_retcode(sol)
end

@testset "Face boundary value in a doubly periodic domain" begin
    # u(t,0,y) keeps a free argument, so on the four all-singleton wrap boxes the
    # pointwise equation carries it as u(t, 0, <grid value>) after valmaps; the element
    # rules must key on that form. Prove nothing stays symbolic and the ODE is runnable.
    @parameters t x y
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y)) + u(t, 0, y)
    bcs = [
        u(0, x, y) ~ sinpi(2x) * sinpi(2y),
        u(t, 0, y) ~ u(t, 1, y),
        u(t, x, 0) ~ u(t, x, 1),
    ]
    domains = [t ∈ Interval(0.0, 0.05), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])
    dxs = [x => 0.2, y => 0.2]

    disc = MOLFiniteDifference(dxs, t)
    sys, _ = symbolic_discretize(pdesys, disc)
    # 3 bands per periodic direction: 5 multi-point boxes in array form, 4
    # all-singleton wrap boxes as scalar equations.
    @test narrayeqs_interior(sys) == 5
    for e in get_eqs(sys)
        s = string(e)
        @test !occursin("u(t, 0,", s) && !occursin("u(t,0,", s)
    end
    # Each wrap-point equation references the substituted seam element u[1, j].
    wrap_eqs = filter(e -> isinterioreq(e) && !isarrayeq(e), get_eqs(sys))
    @test length(wrap_eqs) == 4
    @test all(e -> occursin("[1, ", string(e)) || occursin("[1,", string(e)), wrap_eqs)

    prob = ode_discretize(pdesys, disc)
    du = similar(prob.u0)
    prob.f(du, prob.u0, prob.p, 0.0)
    @test all(isfinite, du)
    sol = solve(prob, Rodas4())
    @test SciMLBase.successful_retcode(sol)
end

# Staggered grids: each variable's alignment fixes its interior stencil taps, so the
# interior collapses to one array equation per PDE. Staggered problems build
# SplitODEProblems without symbolic indexing, so solutions are compared positionally.
function staggered_wave(dx; periodic = false)
    @parameters t x
    @variables ρ(..) ϕ(..)
    Dt = Differential(t)
    Dx = Differential(x)
    a = 5.0
    L = 2.0
    eq = [
        Dt(ρ(t, x)) + Dx(ϕ(t, x)) ~ 0,
        Dt(ϕ(t, x)) + a^2 * Dx(ρ(t, x)) ~ 0,
    ]
    bcs = if periodic
        [
            ρ(0, x) ~ exp(-(x - L / 2)^2), ϕ(0.0, x) ~ 0.0,
            ρ(t, L) ~ ρ(t, -L), ϕ(t, -L) ~ ϕ(t, L),
        ]
    else
        [
            ρ(0, x) ~ exp(-x^2), ϕ(0.0, x) ~ 0.0,
            Dx(ρ(t, L)) ~ 0.0, ϕ(t, -L) ~ 0.0,
        ]
    end
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(-L, L)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [ρ(t, x), ϕ(t, x)])
    disc = MOLFiniteDifference(
        [x => dx], t; grid_align = MethodOfLines.StaggeredGrid(),
        edge_aligned_var = ϕ(t, x)
    )
    return pdesys, disc
end

@testset "Staggered 1D wave equation, mixed BCs" begin
    pdesys, disc = staggered_wave(0.125)
    sys, _ = symbolic_discretize(pdesys, disc)
    # one array equation per PDE
    @test narrayeqs_interior(sys) == 2

    # equation count is independent of resolution
    pdesys2, disc2 = staggered_wave(0.0625)
    sys2, _ = symbolic_discretize(pdesys2, disc2)
    @test length(get_eqs(sys2)) == length(get_eqs(sys))

    prob_arr = discretize(pdesys, disc)
    dt = (0.125 / 5.0)^2
    sol_arr = solve(prob_arr, SplitEuler(), dt = dt)
    @test successful_retcode(sol_arr)
end

@testset "Staggered 1D wave equation, periodic BCs" begin
    pdesys, disc = staggered_wave(0.125; periodic = true)
    sys, _ = symbolic_discretize(pdesys, disc)
    @test narrayeqs_interior(sys) == 2

    pdesys2, disc2 = staggered_wave(0.0625; periodic = true)
    sys2, _ = symbolic_discretize(pdesys2, disc2)
    @test length(get_eqs(sys2)) == length(get_eqs(sys))

    prob_arr = discretize(pdesys, disc)
    dt = (0.125 / 5.0)^2
    sol_arr = solve(prob_arr, SplitEuler(), dt = dt)
    @test successful_retcode(sol_arr)
end

@testset "2D mixed derivative" begin
    @parameters t x y
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    Dxy = Differential(x) * Differential(y)

    eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y)) + 0.5 * Dxy(u(t, x, y))
    bcs = [
        u(0, x, y) ~ sinpi(x) * sinpi(y),
        u(t, 0, y) ~ 0.0, u(t, 1, y) ~ 0.0,
        u(t, x, 0) ~ 0.0, u(t, x, 1) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.05), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0),
    ]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])

    sol_arr, sys_arr = solve_discretized(pdesys, [x => 0.1, y => 0.1], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1

    # Fourth order widens both stencils of the tensor product at once
    sol_arr, sys_arr = solve_discretized(
        pdesys, [x => 0.1, y => 0.1], t; disc_kwargs = (; approx_order = 4)
    )
    @test narrayeqs_interior(sys_arr) == 1
end

@testset "2D mixed derivative with a variable coefficient" begin
    @parameters t x y
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    Dxy = Differential(x) * Differential(y)

    eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y)) +
        (0.3 + 0.2 * x * y) * Dxy(u(t, x, y))
    bcs = [
        u(0, x, y) ~ sinpi(x) * sinpi(y),
        u(t, 0, y) ~ 0.0, u(t, 1, y) ~ 0.0,
        u(t, x, 0) ~ 0.0, u(t, x, 1) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.05), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0),
    ]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])

    sol_arr, sys_arr = solve_discretized(pdesys, [x => 0.1, y => 0.1], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
end

@testset "Mixed derivative on a nonuniform grid" begin
    # The interior weights of both factors vary from point to point, so the tensor product
    # weight is a numeric array rather than a scalar.
    @parameters t x y
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    Dxy = Differential(x) * Differential(y)

    eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y)) + 0.5 * Dxy(u(t, x, y))
    bcs = [
        u(0, x, y) ~ sinpi(x) * sinpi(y),
        u(t, 0, y) ~ 0.0, u(t, 1, y) ~ 0.0,
        u(t, x, 0) ~ 0.0, u(t, x, 1) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.05), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0),
    ]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])

    xgrid = collect(range(0.0, 1.0, length = 11)) .^ 1.2
    ygrid = collect(range(0.0, 1.0, length = 11)) .^ 0.9
    sol_arr, sys_arr = solve_discretized(pdesys, [x => xgrid, y => ygrid], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
end

@testset "Mixed derivative with a periodic direction" begin
    # Both axes of the tensor product wrap independently, as `mixed_central_difference`
    # does with the two `bwrap`ped tap sets it takes the product of.
    @parameters t x y
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    Dxy = Differential(x) * Differential(y)

    eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y)) + 0.5 * Dxy(u(t, x, y))
    bcs = [
        u(0, x, y) ~ sinpi(2x) * sinpi(y),
        u(t, 0, y) ~ u(t, 1, y), Dx(u(t, 0, y)) ~ Dx(u(t, 1, y)),
        u(t, x, 0) ~ 0.0, u(t, x, 1) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.05), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0),
    ]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])

    sol_arr, sys_arr = solve_discretized(pdesys, [x => 0.1, y => 0.1], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    # one box per band of the periodic decomposition, a count independent of the grid
    @test narrayeqs_interior(sys_arr) == 2
end

@testset "3D mixed derivative" begin
    @parameters t x y z
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    Dzz = Differential(z)^2
    Dxz = Differential(x) * Differential(z)

    eq = Dt(u(t, x, y, z)) ~ Dxx(u(t, x, y, z)) + Dyy(u(t, x, y, z)) +
        Dzz(u(t, x, y, z)) + 0.4 * Dxz(u(t, x, y, z))
    bcs = [
        u(0, x, y, z) ~ sinpi(x) * sinpi(y) * sinpi(z),
        u(t, 0, y, z) ~ 0.0, u(t, 1, y, z) ~ 0.0,
        u(t, x, 0, z) ~ 0.0, u(t, x, 1, z) ~ 0.0,
        u(t, x, y, 0) ~ 0.0, u(t, x, y, 1) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.02), x ∈ Interval(0.0, 1.0),
        y ∈ Interval(0.0, 1.0), z ∈ Interval(0.0, 1.0),
    ]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x, y, z], [u(t, x, y, z)])

    sol_arr, sys_arr = solve_discretized(pdesys, [x => 0.2, y => 0.2, z => 0.2], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
end

@testset "Mixed derivative alone sets the band width" begin
    # `Dx(Dy(u))` reaches along `x` with the centered first order stencil, which at higher
    # approximation orders is wider than the winding stencil the order 1 entry of
    # `pdeorders` would otherwise select. With nothing else in the equation to widen the
    # band, that is what has to set it, or the array form reaches past the grid and the
    # whole equation degrades to pointwise.
    @parameters t x y
    @variables u(..)
    Dt = Differential(t)
    Dxy = Differential(x) * Differential(y)

    eq = Dt(u(t, x, y)) ~ Dxy(u(t, x, y))
    bcs = [
        u(0, x, y) ~ sinpi(x) * sinpi(y),
        u(t, 0, y) ~ 0.0, u(t, 1, y) ~ 0.0,
        u(t, x, 0) ~ 0.0, u(t, x, 1) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.01), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0),
    ]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])

    for order in (2, 4, 6)
        sol_arr, sys_arr = solve_discretized(
            pdesys, [x => 0.05, y => 0.05], t; disc_kwargs = (; approx_order = order)
        )
        @test narrayeqs_interior(sys_arr) == 1
    end
end

@testset "2D higher mixed derivatives" begin
    # `(Differential(x)^m * Differential(y)^n)(u)` is the same tensor product as
    # `Dx(Dy(u))`, with the centered operators of orders `m` and `n`. Bare high-order
    # mixed terms are not dissipative (`u_xxyy` grows like π⁴u), so the solve cases
    # sit next to `Dxx + Dyy` the way the first-order mixed tests do.
    @parameters t x y
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    ops = (
        (Differential(x)^2) * Differential(y),
        Differential(x) * (Differential(y)^2),
        (Differential(x)^2) * (Differential(y)^2),
        (Differential(x)^3) * Differential(y),
    )
    bcs = [
        u(0, x, y) ~ sinpi(x) * sinpi(y),
        u(t, 0, y) ~ 0.0, u(t, 1, y) ~ 0.0,
        u(t, x, 0) ~ 0.0, u(t, x, 1) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.01), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0),
    ]
    for Dmix in ops
        eq = Dt(u(t, x, y)) ~ Dmix(u(t, x, y))
        @named pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])
        disc = MOLFiniteDifference([x => 0.1, y => 0.1], t)
        sys, _ = symbolic_discretize(pdesys, disc)
        @test narrayeqs_interior(sys) == 1
    end

    Dxxy = (Differential(x)^2) * Differential(y)
    eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y)) + 0.5 * Dxxy(u(t, x, y))
    @named pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])
    sol_arr, sys_arr = solve_discretized(pdesys, [x => 0.1, y => 0.1], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1

    # Fourth and sixth order widen both factors of `Dxxy` at once
    for order in (4, 6)
        sol_arr, sys_arr = solve_discretized(
            pdesys, [x => 0.05, y => 0.05], t; disc_kwargs = (; approx_order = order)
        )
        @test sol_arr.retcode == SciMLBase.ReturnCode.Success
        @test narrayeqs_interior(sys_arr) == 1
    end
end

@testset "2D higher mixed derivative with a variable coefficient" begin
    @parameters t x y
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    Dxxy = (Differential(x)^2) * Differential(y)

    eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y)) +
        (0.3 + 0.2 * x * y) * Dxxy(u(t, x, y))
    bcs = [
        u(0, x, y) ~ sinpi(x) * sinpi(y),
        u(t, 0, y) ~ 0.0, u(t, 1, y) ~ 0.0,
        u(t, x, 0) ~ 0.0, u(t, x, 1) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.05), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0),
    ]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])

    sol_arr, sys_arr = solve_discretized(pdesys, [x => 0.1, y => 0.1], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
end

@testset "Higher mixed derivative on a nonuniform grid" begin
    @parameters t x y
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    Dxxy = (Differential(x)^2) * Differential(y)

    eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y)) + 0.5 * Dxxy(u(t, x, y))
    bcs = [
        u(0, x, y) ~ sinpi(x) * sinpi(y),
        u(t, 0, y) ~ 0.0, u(t, 1, y) ~ 0.0,
        u(t, x, 0) ~ 0.0, u(t, x, 1) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.05), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0),
    ]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])

    xgrid = collect(range(0.0, 1.0, length = 11)) .^ 1.2
    ygrid = collect(range(0.0, 1.0, length = 11)) .^ 0.9
    sol_arr, sys_arr = solve_discretized(pdesys, [x => xgrid, y => ygrid], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
end

@testset "Higher mixed derivative with a periodic direction" begin
    @parameters t x y
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    Dxxy = (Differential(x)^2) * Differential(y)

    eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y)) + 0.5 * Dxxy(u(t, x, y))
    bcs = [
        u(0, x, y) ~ sinpi(2x) * sinpi(y),
        u(t, 0, y) ~ u(t, 1, y), Dx(u(t, 0, y)) ~ Dx(u(t, 1, y)),
        u(t, x, 0) ~ 0.0, u(t, x, 1) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.05), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0),
    ]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])

    sol_arr, sys_arr = solve_discretized(pdesys, [x => 0.1, y => 0.1], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) >= 1

    # Wrap-box count follows the stencil width, not the grid resolution. Total
    # `get_eqs` still grows here: the Dirichlet y-frame is pointwise along x.
    disc2 = MOLFiniteDifference([x => 0.05, y => 0.05], t)
    sys_arr2, _ = symbolic_discretize(pdesys, disc2)
    @test narrayeqs_interior(sys_arr2) == narrayeqs_interior(sys_arr)

    # Doubly periodic: no Dirichlet frame, so the total equation count is O(stencil).
    bcs_pp = [
        u(0, x, y) ~ sinpi(2x) * sinpi(2y),
        u(t, 0, y) ~ u(t, 1, y),
        u(t, x, 0) ~ u(t, x, 1),
    ]
    @named pdesys_pp = PDESystem(eq, bcs_pp, domains, [t, x, y], [u(t, x, y)])
    counts = map([8, 16]) do n
        disc = MOLFiniteDifference([x => 1 / (n - 1), y => 1 / (n - 1)], t)
        sys, _ = symbolic_discretize(pdesys_pp, disc)
        (; n = length(get_eqs(sys)), narr = narrayeqs_interior(sys))
    end
    @test counts[1].n == counts[2].n
    @test counts[1].narr >= 1
    @test counts[1].narr == counts[2].narr
end

@testset "3D higher mixed derivative" begin
    @parameters t x y z
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    Dzz = Differential(z)^2
    Dxxz = (Differential(x)^2) * Differential(z)

    eq = Dt(u(t, x, y, z)) ~ Dxx(u(t, x, y, z)) + Dyy(u(t, x, y, z)) +
        Dzz(u(t, x, y, z)) + 0.4 * Dxxz(u(t, x, y, z))
    bcs = [
        u(0, x, y, z) ~ sinpi(x) * sinpi(y) * sinpi(z),
        u(t, 0, y, z) ~ 0.0, u(t, 1, y, z) ~ 0.0,
        u(t, x, 0, z) ~ 0.0, u(t, x, 1, z) ~ 0.0,
        u(t, x, y, 0) ~ 0.0, u(t, x, y, 1) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.02), x ∈ Interval(0.0, 1.0),
        y ∈ Interval(0.0, 1.0), z ∈ Interval(0.0, 1.0),
    ]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x, y, z], [u(t, x, y, z)])

    sol_arr, sys_arr = solve_discretized(pdesys, [x => 0.2, y => 0.2, z => 0.2], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
end

@testset "Higher mixed derivative alone sets the band width" begin
    # `Dx^3(Dy(u))` reaches along `x` with the centered third-order stencil, which is
    # wider than the winding stencil the order-3 entry of `pdeorders` would select.
    # With nothing else in the equation to widen the band, that is what has to set it.
    @parameters t x y
    @variables u(..)
    Dt = Differential(t)
    Dxxxy = (Differential(x)^3) * Differential(y)

    eq = Dt(u(t, x, y)) ~ Dxxxy(u(t, x, y))
    bcs = [
        u(0, x, y) ~ sinpi(x) * sinpi(y),
        u(t, 0, y) ~ 0.0, u(t, 1, y) ~ 0.0,
        u(t, x, 0) ~ 0.0, u(t, x, 1) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.01), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0),
    ]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])

    for order in (2, 4, 6)
        disc = MOLFiniteDifference(
            [x => 0.05, y => 0.05], t; approx_order = order
        )
        sys_arr, _ = symbolic_discretize(pdesys, disc)
        @test narrayeqs_interior(sys_arr) == 1
    end
end

@testset "Higher mixed derivative manufactured solution" begin
    # u = sin(πx) sin(πy) is steady for Dt(u) ~ u_xxy + π³ sin(πx) cos(πy)
    @parameters t x y
    @variables u(..)
    Dt = Differential(t)
    Dxxy = (Differential(x)^2) * Differential(y)

    eq = Dt(u(t, x, y)) ~
        Dxxy(u(t, x, y)) + (pi^3) * sinpi(x) * cospi(y)
    bcs = [
        u(0, x, y) ~ sinpi(x) * sinpi(y),
        u(t, 0, y) ~ 0.0, u(t, 1, y) ~ 0.0,
        u(t, x, 0) ~ 0.0, u(t, x, 1) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.05), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0),
    ]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])

    sol_arr, sys_arr = solve_discretized(pdesys, [x => 0.05, y => 0.05], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
    xdisc = sol_arr[x]
    ydisc = sol_arr[y]
    exact = [sinpi(xi) * sinpi(yi) for xi in xdisc, yi in ydisc]
    @test maximum(abs.(sol_arr[u(t, x, y)][end, :, :] .- exact)) < 5.0e-2
end

@testset "Fallback: three-direction mixed derivative without transform" begin
    # `Dx(Dy(Dz(u)))` is not a two-direction mixed term. With transformation off it
    # reaches `arrayify` unhandled and must fall back rather than error.
    @parameters t x y z
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dy = Differential(y)
    Dz = Differential(z)

    eq = Dt(u(t, x, y, z)) ~ Dx(Dy(Dz(u(t, x, y, z))))
    bcs = [
        u(0, x, y, z) ~ sinpi(x) * sinpi(y) * sinpi(z),
        u(t, 0, y, z) ~ 0.0, u(t, 1, y, z) ~ 0.0,
        u(t, x, 0, z) ~ 0.0, u(t, x, 1, z) ~ 0.0,
        u(t, x, y, 0) ~ 0.0, u(t, x, y, 1) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.01), x ∈ Interval(0.0, 1.0),
        y ∈ Interval(0.0, 1.0), z ∈ Interval(0.0, 1.0),
    ]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x, y, z], [u(t, x, y, z)])

    disc = MOLFiniteDifference(
        [x => 0.2, y => 0.2, z => 0.2], t; should_transform = false
    )
    sys, _ = symbolic_discretize(pdesys, disc)
    @test narrayeqs_interior(sys) == 0
end

# Numeric arrays / `array_literal`s are data, not a growing symbolic scheme.
function scheme_treesize(x)
    x = Symbolics.unwrap(x)
    x isa AbstractArray && return 1
    SymbolicUtils.iscall(x) || return 1
    op = SymbolicUtils.operation(x)
    if op === SymbolicUtils.array_literal ||
            (op isa Function && nameof(op) === :array_literal)
        return 1
    end
    return 1 + sum(scheme_treesize, SymbolicUtils.arguments(x); init = 0)
end
eq_scheme_size(eq) = scheme_treesize(eq.lhs) + scheme_treesize(eq.rhs)

function trap_scan(u, dx)
    n = length(u)
    I = zeros(eltype(u), n)
    for k in 2:n
        Δ = dx isa Number ? dx : dx[k - 1]
        I[k] = I[k - 1] + (Δ / 2) * (u[k - 1] + u[k])
    end
    return I
end

@testset "1D cumulative integral is a slice reduction" begin
    @parameters t x
    @variables integrand(..) cumuSum(..)
    xmin = 0.0
    xmax = 2.0 * pi
    Ix = Integral(x in DomainSets.ClosedInterval(xmin, x))
    eqs = [cumuSum(t, x) ~ Ix(integrand(t, x)), integrand(t, x) ~ t * cos(x)]
    bcs = [cumuSum(0, x) ~ 0.0, integrand(0, x) ~ 0.0]
    domains = [t ∈ Interval(0.0, 1.0), x ∈ Interval(xmin, xmax)]
    @named pdesys = PDESystem(eqs, bcs, domains, [t, x], [integrand(t, x), cumuSum(t, x)])

    counts = map([11, 41]) do n
        sys, _ = symbolic_discretize(pdesys, MOLFiniteDifference([x => n], t))
        scan = only(filter(eq -> occursin("axis_cumsum", string(eq)), get_eqs(sys)))
        (narrayeqs(sys), eq_scheme_size(scan))
    end
    @test counts[1][1] == counts[2][1] == 2
    @test counts[1][2] == counts[2][2]

    disc = MOLFiniteDifference([x => 21], t)
    prob = discretize(pdesys, disc)
    @test prob isa SciMLBase.DAEProblem
    sol = solve(prob)
    @test SciMLBase.successful_retcode(sol)
    xdisc = sol[x]
    exact = [ti * sin(xi) for ti in sol[t], xi in xdisc]
    @test sol[cumuSum(t, x)] ≈ exact atol = 0.36
    dx = xdisc[2] - xdisc[1]
    @test sol[cumuSum(t, x)][end, :] ≈ trap_scan(sol[integrand(t, x)][end, :], dx) atol = 1.0e-10
end

@testset "Nonuniform cumulative integral stays a slice" begin
    @parameters t x
    @variables integrand(..) cumuSum(..)
    xmin = 0.0
    xmax = 2.0 * pi
    Ix = Integral(x in DomainSets.ClosedInterval(xmin, x))
    eqs = [cumuSum(t, x) ~ Ix(integrand(t, x)), integrand(t, x) ~ t * cos(x)]
    bcs = [cumuSum(0, x) ~ 0.0, integrand(0, x) ~ 0.0]
    domains = [t ∈ Interval(0.0, 1.0), x ∈ Interval(xmin, xmax)]
    @named pdesys = PDESystem(eqs, bcs, domains, [t, x], [integrand(t, x), cumuSum(t, x)])

    counts = map([11, 31]) do n
        xs = collect(range(xmin, xmax; length = n))
        sys, _ = symbolic_discretize(pdesys, MOLFiniteDifference([x => xs], t))
        scan = only(filter(eq -> occursin("axis_cumsum", string(eq)), get_eqs(sys)))
        (narrayeqs(sys), eq_scheme_size(scan))
    end
    @test counts[1][1] == counts[2][1] == 2
    @test counts[1][2] == counts[2][2]

    xs = collect(range(xmin, xmax; length = 21))
    sol = solve(discretize(pdesys, MOLFiniteDifference([x => xs], t)))
    @test SciMLBase.successful_retcode(sol)
    dxs = diff(collect(sol[x]))
    @test sol[cumuSum(t, x)][end, :] ≈ trap_scan(sol[integrand(t, x)][end, :], dxs) atol = 1.0e-10
end

@testset "2D cumulative integral along one axis" begin
    @parameters t x y
    @variables u(..) cumuSum(..)
    Ixc = Integral(x in DomainSets.ClosedInterval(0.0, x))
    eqs = [cumuSum(t, x, y) ~ Ixc(u(t, x, y)), u(t, x, y) ~ t * cos(x) * sinpi(y)]
    bcs = [cumuSum(0, x, y) ~ 0.0, u(0, x, y) ~ 0.0]
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eqs, bcs, domains, [t, x, y], [u(t, x, y), cumuSum(t, x, y)])

    counts = map([(8, 6), (16, 10)]) do (nx, ny)
        sys, _ = symbolic_discretize(pdesys, MOLFiniteDifference([x => nx, y => ny], t))
        scan = only(filter(eq -> occursin("axis_cumsum", string(eq)), get_eqs(sys)))
        (narrayeqs(sys), eq_scheme_size(scan))
    end
    @test counts[1][1] == counts[2][1] == 2
    @test counts[1][2] == counts[2][2]
end

@testset "PIDE: wrapped cumulative plus first-order derivative" begin
    @parameters t x
    @variables u(..) cumuSum(..)
    Dt = Differential(t)
    Dx = Differential(x)
    xmin = 0.0
    xmax = 2.0 * pi
    Ix = Integral(x in DomainSets.ClosedInterval(xmin, x))
    eqs = [
        cumuSum(t, x) ~ Ix(u(t, x)),
        Dt(u(t, x)) + 2 * u(t, x) + 5 * Dx(cumuSum(t, x)) ~ 1,
    ]
    bcs = [u(0.0, x) ~ cos(x), Dx(u(t, xmin)) ~ 0.0, Dx(u(t, xmax)) ~ 0]
    domains = [t ∈ Interval(0.0, 0.2), x ∈ Interval(xmin, xmax)]
    @named pdesys = PDESystem(eqs, bcs, domains, [t, x], [u(t, x), cumuSum(t, x)])

    counts = map([12, 24]) do n
        sys, _ = symbolic_discretize(pdesys, MOLFiniteDifference([x => n], t))
        (
            narrayeqs(sys), narrayeqs_interior(sys),
            eq_scheme_size(only(filter(eq -> occursin("axis_cumsum", string(eq)), get_eqs(sys)))),
        )
    end
    @test counts[1][1] == counts[2][1]
    @test counts[1][2] == counts[2][2] == 1
    @test counts[1][3] == counts[2][3]

    sol = solve(discretize(pdesys, MOLFiniteDifference([x => 16], t)))
    @test SciMLBase.successful_retcode(sol)
end

@testset "Integral and central diffusion in one equation" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    Ix = Integral(x in DomainSets.ClosedInterval(0.0, x))
    eq = Dt(u(t, x)) ~ Dxx(u(t, x)) + Ix(u(t, x))
    bcs = [u(0, x) ~ cos(x), u(t, 0) ~ 1.0, u(t, 2π) ~ 1.0]
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 2π)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    sys, _ = symbolic_discretize(pdesys, MOLFiniteDifference([x => 16], t))
    @test narrayeqs_interior(sys) == 1
    int = only(filter(eq -> isinterioreq(eq) && isarrayeq(eq), get_eqs(sys)))
    @test occursin("axis_cumsum", string(int))
end

@testset "0D whole-domain integral is a compact sum" begin
    @parameters t x
    @variables integrand(..) cumuSum(..)
    xmin = 0.0
    xmax = 2.0 * pi
    Ix = Integral(x in DomainSets.ClosedInterval(xmin, xmax))
    eqs = [cumuSum(t) ~ Ix(integrand(t, x)), integrand(t, x) ~ t * cos(x)]
    bcs = [cumuSum(0) ~ 0.0, integrand(0, x) ~ 0.0]
    domains = [t ∈ Interval(0.0, 1.0), x ∈ Interval(xmin, xmax)]
    @named pdesys = PDESystem(eqs, bcs, domains, [t, x], [integrand(t, x), cumuSum(t)])

    counts = map([11, 41]) do n
        sys, _ = symbolic_discretize(pdesys, MOLFiniteDifference([x => n], t))
        zeq = only(filter(eq -> occursin("cumuSum", string(eq)), get_eqs(sys)))
        (narrayeqs(sys), !isarrayeq(zeq), eq_scheme_size(zeq))
    end
    @test counts[1][1] == counts[2][1] == 1
    @test counts[1][2] && counts[2][2]
    @test counts[1][3] == counts[2][3]

    sol = solve(discretize(pdesys, MOLFiniteDifference([x => 21], t)))
    @test SciMLBase.successful_retcode(sol)
    @test sol[cumuSum(t)] ≈ zeros(length(sol[t])) atol = 0.3
end

@testset "Rank-dropping whole-domain integral (sum along one axis)" begin
    @parameters t x y
    @variables u(..) v(..)
    Ix = Integral(x in DomainSets.ClosedInterval(0.0, 1.0))
    eqs = [v(t, y) ~ Ix(u(t, x, y)), u(t, x, y) ~ t * cos(x) * sinpi(y)]
    bcs = [v(0, y) ~ 0.0, u(0, x, y) ~ 0.0]
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eqs, bcs, domains, [t, x, y], [u(t, x, y), v(t, y)])

    counts = map([(8, 6), (16, 10)]) do (nx, ny)
        sys, _ = symbolic_discretize(pdesys, MOLFiniteDifference([x => nx, y => ny], t))
        red = only(filter(eq -> occursin("axis_sum", string(eq)), get_eqs(sys)))
        (narrayeqs(sys), eq_scheme_size(red))
    end
    @test counts[1][1] == counts[2][1] == 2
    @test counts[1][2] == counts[2][2]
end

@testset "SIR-age: 0D lift keeps the spatial equation in slice form" begin
    β = 0.0005
    γ = 0.25
    @parameters t a
    @variables S(..) I(..) R(..)
    Dt = Differential(t)
    Da = Differential(a)
    Ia = Integral(a in DomainSets.ClosedInterval(0.0, 40.0))
    eqs = [
        Dt(S(t)) ~ -β * S(t) * Ia(I(a, t)),
        Dt(I(a, t)) + Da(I(a, t)) ~ -γ * I(a, t),
        Dt(R(t)) ~ γ * Ia(I(a, t)),
    ]
    bcs = [
        S(0) ~ 990.0,
        I(0, t) ~ β * S(t) * Ia(I(a, t)),
        I(a, 0) ~ 10.0 / 40.0,
        R(0) ~ 0.0,
    ]
    @named pdesys = PDESystem(
        eqs, bcs, [t ∈ (0.0, 1.0), a ∈ (0.0, 40.0)], [a, t], [S(t), I(a, t), R(t)]
    )

    counts = map([10, 20]) do n
        sys, _ = symbolic_discretize(pdesys, MOLFiniteDifference([a => n], t))
        eqs_out = get_eqs(sys)
        Seq = only(
            filter(
                eq -> occursin("Differential(t", string(eq)) && occursin("S(t)", string(eq)) &&
                    !isarrayeq(eq), eqs_out
            )
        )
        Req = only(
            filter(
                eq -> occursin("Differential(t", string(eq)) && occursin("R(t)", string(eq)),
                eqs_out
            )
        )
        (
            narrayeqs_interior(sys), length(eqs_out),
            eq_scheme_size(Seq), eq_scheme_size(Req),
            occursin("array_weight_scale", string(Seq)),
            occursin("array_weight_scale", string(Req)),
        )
    end
    @test counts[1][1] == counts[2][1] == 1
    @test counts[1][2] == counts[2][2]
    @test counts[1][3] == counts[2][3]
    @test counts[1][4] == counts[2][4]
    @test counts[1][5] && counts[2][5]
    @test counts[1][6] && counts[2][6]

    sol = solve(discretize(pdesys, MOLFiniteDifference([a => 16], t)))
    @test SciMLBase.successful_retcode(sol)
end

@testset "Fallback: stationary system with an integral" begin
    @parameters x
    @variables u(..) I(..)
    Ix = Integral(x in DomainSets.ClosedInterval(0.0, x))
    @named pdesys = PDESystem(
        [I(x) ~ Ix(u(x)), u(x) ~ cos(x)], [I(0) ~ 0.0],
        [x ∈ Interval(0.0, 1.0)], [x], [u(x), I(x)]
    )
    sys, _ = symbolic_discretize(pdesys, MOLFiniteDifference([x => 8]))
    @test narrayeqs(sys) == 0
end

@testset "Higher mixed derivative and integral stay in slice form" begin
    @parameters t x y
    @variables u(..)
    Dt = Differential(t)
    Dxxy = (Differential(x)^2) * Differential(y)
    Ix = Integral(x in DomainSets.ClosedInterval(0.0, x))
    eq = Dt(u(t, x, y)) ~ Dxxy(u(t, x, y)) + Ix(u(t, x, y))
    bcs = [
        u(0, x, y) ~ sinpi(x) * sinpi(y),
        u(t, 0, y) ~ 0.0, u(t, 1, y) ~ 0.0,
        u(t, x, 0) ~ 0.0, u(t, x, 1) ~ 0.0,
    ]
    domains = [t ∈ Interval(0.0, 0.01), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])
    sys, _ = symbolic_discretize(pdesys, MOLFiniteDifference([x => 0.2, y => 0.2], t))
    @test narrayeqs_interior(sys) == 1
    int = only(filter(eq -> isinterioreq(eq) && isarrayeq(eq), get_eqs(sys)))
    @test occursin("axis_cumsum", string(int))
end

@testset "Fallback: three-direction mixed still falls back with an integral" begin
    @parameters t x y z
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dy = Differential(y)
    Dz = Differential(z)
    Ix = Integral(x in DomainSets.ClosedInterval(0.0, x))
    eq = Dt(u(t, x, y, z)) ~ Dx(Dy(Dz(u(t, x, y, z)))) + Ix(u(t, x, y, z))
    bcs = [
        u(0, x, y, z) ~ sinpi(x) * sinpi(y) * sinpi(z),
        u(t, 0, y, z) ~ 0.0, u(t, 1, y, z) ~ 0.0,
        u(t, x, 0, z) ~ 0.0, u(t, x, 1, z) ~ 0.0,
        u(t, x, y, 0) ~ 0.0, u(t, x, y, 1) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.01), x ∈ Interval(0.0, 1.0),
        y ∈ Interval(0.0, 1.0), z ∈ Interval(0.0, 1.0),
    ]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x, y, z], [u(t, x, y, z)])
    disc = MOLFiniteDifference(
        [x => 0.2, y => 0.2, z => 0.2], t; should_transform = false
    )
    sys, _ = symbolic_discretize(pdesys, disc)
    @test narrayeqs_interior(sys) == 0
end

@testset "1D two-domain nonlinear laplacian" begin
    @parameters t x1 x2
    @variables c1(..) c2(..)
    Dt = Differential(t)
    Dx1 = Differential(x1)
    Dx2 = Differential(x2)
    D1(c) = 1 + c / 10
    D2(c) = 1 / 10 + c / 10

    eqs = [
        Dt(c1(t, x1)) ~ Dx1(D1(c1(t, x1)) * Dx1(c1(t, x1))),
        Dt(c2(t, x2)) ~ Dx2(D2(c2(t, x2)) * Dx2(c2(t, x2))),
    ]
    bcs = [
        c1(0, x1) ~ 1 + cospi(2 * x1),
        c2(0, x2) ~ 1 + cospi(2 * x2),
        Dx1(c1(t, 0)) ~ 0,
        c1(t, 0.5) ~ c2(t, 0.5),
        -D1(c1(t, 0.5)) * Dx1(c1(t, 0.5)) ~ -D2(c2(t, 0.5)) * Dx2(c2(t, 0.5)),
        Dx2(c2(t, 1)) ~ 0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.05), x1 ∈ Interval(0.0, 0.5), x2 ∈ Interval(0.5, 1.0),
    ]
    @named pdesys = PDESystem(eqs, bcs, domains, [t, x1, x2], [c1(t, x1), c2(t, x2)])

    sol_arr, sys_arr = solve_discretized(pdesys, [x1 => 0.05, x2 => 0.05], t)
    @test SciMLBase.successful_retcode(sol_arr)
    @test narrayeqs_interior(sys_arr) == 2
end

@testset "1D two-domain, unequal grid lengths" begin
    # Lower wrap uses the destination length; n1 ≠ n2 is the case that same-n
    # periodic arithmetic would get wrong.
    @parameters t x1 x2
    @variables c1(..) c2(..)
    Dt = Differential(t)
    Dxx1 = Differential(x1)^2
    Dxx2 = Differential(x2)^2

    eqs = [Dt(c1(t, x1)) ~ Dxx1(c1(t, x1)), Dt(c2(t, x2)) ~ Dxx2(c2(t, x2))]
    bcs = [
        c1(0, x1) ~ sinpi(2x1),
        c2(0, x2) ~ sinpi(x2),
        c1(t, 0) ~ 0.0,
        c1(t, 0.5) ~ c2(t, 0.5),
        c2(t, 1.5) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.05), x1 ∈ Interval(0.0, 0.5), x2 ∈ Interval(0.5, 1.5),
    ]
    @named pdesys = PDESystem(eqs, bcs, domains, [t, x1, x2], [c1(t, x1), c2(t, x2)])

    sol_arr, sys_arr = solve_discretized(pdesys, [x1 => 0.05, x2 => 0.05], t)
    @test SciMLBase.successful_retcode(sol_arr)
    @test narrayeqs_interior(sys_arr) == 2
end

@testset "Three-domain interface chain" begin
    @parameters t x1 x2 x3
    @variables c1(..) c2(..) c3(..)
    Dt = Differential(t)
    Dxx1 = Differential(x1)^2
    Dxx2 = Differential(x2)^2
    Dxx3 = Differential(x3)^2

    eqs = [
        Dt(c1(t, x1)) ~ Dxx1(c1(t, x1)),
        Dt(c2(t, x2)) ~ Dxx2(c2(t, x2)),
        Dt(c3(t, x3)) ~ Dxx3(c3(t, x3)),
    ]
    bcs = [
        c1(0, x1) ~ sinpi(3x1),
        c2(0, x2) ~ sinpi(3x2),
        c3(0, x3) ~ sinpi(3x3),
        c1(t, 0) ~ 0.0,
        c1(t, 1 / 3) ~ c2(t, 1 / 3),
        c2(t, 2 / 3) ~ c3(t, 2 / 3),
        c3(t, 1) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.05),
        x1 ∈ Interval(0.0, 1 / 3),
        x2 ∈ Interval(1 / 3, 2 / 3),
        x3 ∈ Interval(2 / 3, 1.0),
    ]
    @named pdesys = PDESystem(
        eqs, bcs, domains, [t, x1, x2, x3], [c1(t, x1), c2(t, x2), c3(t, x3)]
    )

    sol_arr, sys_arr = solve_discretized(
        pdesys, [x1 => 1 / 12, x2 => 1 / 12, x3 => 1 / 12], t
    )
    @test SciMLBase.successful_retcode(sol_arr)
    @test narrayeqs_interior(sys_arr) == 3
end

@testset "2D two-domain interface" begin
    @parameters t x1 x2 y
    @variables c1(..) c2(..)
    Dt = Differential(t)
    Dxx1 = Differential(x1)^2
    Dxx2 = Differential(x2)^2
    Dyy = Differential(y)^2

    eqs = [
        Dt(c1(t, x1, y)) ~ Dxx1(c1(t, x1, y)) + Dyy(c1(t, x1, y)),
        Dt(c2(t, x2, y)) ~ Dxx2(c2(t, x2, y)) + Dyy(c2(t, x2, y)),
    ]
    bcs = [
        c1(0, x1, y) ~ sinpi(2x1) * sinpi(y),
        c2(0, x2, y) ~ sinpi(2x2) * sinpi(y),
        c1(t, 0, y) ~ 0.0,
        c1(t, 0.5, y) ~ c2(t, 0.5, y),
        -Differential(x1)(c1(t, 0.5, y)) ~ -Differential(x2)(c2(t, 0.5, y)),
        c2(t, 1, y) ~ 0.0,
        c1(t, x1, 0) ~ 0.0, c1(t, x1, 1) ~ 0.0,
        c2(t, x2, 0) ~ 0.0, c2(t, x2, 1) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.05),
        x1 ∈ Interval(0.0, 0.5), x2 ∈ Interval(0.5, 1.0), y ∈ Interval(0.0, 1.0),
    ]
    @named pdesys = PDESystem(
        eqs, bcs, domains, [t, x1, x2, y], [c1(t, x1, y), c2(t, x2, y)]
    )

    sol_arr, sys_arr = solve_discretized(
        pdesys, [x1 => 0.1, x2 => 0.1, y => 0.1], t
    )
    @test SciMLBase.successful_retcode(sol_arr)
    @test narrayeqs_interior(sys_arr) >= 2
    # identification face + flux face are slices, so the equation count is
    # independent of the transverse resolution
    n8 = length(
        get_eqs(
            first(
                symbolic_discretize(
                    pdesys, MOLFiniteDifference([x1 => 0.1, x2 => 0.1, y => 1 / 8], t)
                )
            )
        )
    )
    n16 = length(
        get_eqs(
            first(
                symbolic_discretize(
                    pdesys, MOLFiniteDifference([x1 => 0.1, x2 => 0.1, y => 1 / 16], t)
                )
            )
        )
    )
    @test n8 == n16
end

@testset "2D two-domain in x, periodic in y" begin
    @parameters t x1 x2 y
    @variables c1(..) c2(..)
    Dt = Differential(t)
    Dxx1 = Differential(x1)^2
    Dxx2 = Differential(x2)^2
    Dyy = Differential(y)^2

    eqs = [
        Dt(c1(t, x1, y)) ~ Dxx1(c1(t, x1, y)) + Dyy(c1(t, x1, y)),
        Dt(c2(t, x2, y)) ~ Dxx2(c2(t, x2, y)) + Dyy(c2(t, x2, y)),
    ]
    bcs = [
        c1(0, x1, y) ~ sinpi(2x1) * sinpi(2y),
        c2(0, x2, y) ~ sinpi(2x2) * sinpi(2y),
        c1(t, 0, y) ~ 0.0,
        c1(t, 0.5, y) ~ c2(t, 0.5, y),
        c2(t, 1, y) ~ 0.0,
        c1(t, x1, 0) ~ c1(t, x1, 1),
        c2(t, x2, 0) ~ c2(t, x2, 1),
    ]
    domains = [
        t ∈ Interval(0.0, 0.05),
        x1 ∈ Interval(0.0, 0.5), x2 ∈ Interval(0.5, 1.0), y ∈ Interval(0.0, 1.0),
    ]
    @named pdesys = PDESystem(
        eqs, bcs, domains, [t, x1, x2, y], [c1(t, x1, y), c2(t, x2, y)]
    )

    sol_arr, sys_arr = solve_discretized(
        pdesys, [x1 => 0.1, x2 => 0.1, y => 0.1], t
    )
    @test SciMLBase.successful_retcode(sol_arr)
    @test narrayeqs_interior(sys_arr) >= 2
end

@testset "WENO two-domain interface on nonuniform grids" begin
    @parameters t x1 x2
    @variables c1(..) c2(..)
    Dt = Differential(t)
    Dx1 = Differential(x1)
    Dx2 = Differential(x2)

    eqs = [Dt(c1(t, x1)) ~ -Dx1(c1(t, x1)), Dt(c2(t, x2)) ~ -Dx2(c2(t, x2))]
    bcs = [
        c1(0, x1) ~ exp(-100 * (x1 - 0.25)^2),
        c2(0, x2) ~ exp(-100 * (x2 - 0.25)^2),
        c1(t, 0) ~ 0.0,
        c1(t, 0.5) ~ c2(t, 0.5),
        Dx2(c2(t, 1)) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.05), x1 ∈ Interval(0.0, 0.5), x2 ∈ Interval(0.5, 1.0),
    ]
    @named pdesys = PDESystem(eqs, bcs, domains, [t, x1, x2], [c1(t, x1), c2(t, x2)])

    g1 = [0.25 * (1 - cospi(i / 20)) for i in 0:20]
    g2 = [0.5 + 0.25 * (1 - cospi(i / 24)) for i in 0:24]
    sol_arr, sys_arr = solve_discretized(
        pdesys, [x1 => g1, x2 => g2], t;
        disc_kwargs = (; advection_scheme = WENOScheme()), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-3)
    )
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 2
end

@testset "Fallback: linear operators on a nonuniform two-domain interface" begin
    # First-order upwind on NU two-domain is coordinate-aware on the pointwise path,
    # but the array weights are not; the whole equation stays pointwise.
    @parameters t x1 x2
    @variables c1(..) c2(..)
    Dt = Differential(t)
    Dx1 = Differential(x1)
    Dx2 = Differential(x2)

    eqs = [Dt(c1(t, x1)) ~ -Dx1(c1(t, x1)), Dt(c2(t, x2)) ~ -Dx2(c2(t, x2))]
    bcs = [
        c1(0, x1) ~ exp(-80 * (x1 - 0.2)^2),
        c2(0, x2) ~ exp(-80 * (x2 - 0.2)^2),
        c1(t, 0) ~ 0.0,
        c1(t, 0.5) ~ c2(t, 0.5),
        Dx2(c2(t, 1)) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.05), x1 ∈ Interval(0.0, 0.5), x2 ∈ Interval(0.5, 1.0),
    ]
    @named pdesys = PDESystem(eqs, bcs, domains, [t, x1, x2], [c1(t, x1), c2(t, x2)])

    g1 = [0.25 * (1 - cospi(i / 16)) for i in 0:16]
    g2 = [0.5 + 0.25 * (1 - cospi(i / 16)) for i in 0:16]
    disc = MOLFiniteDifference([x1 => g1, x2 => g2], t)
    sys, _ = symbolic_discretize(pdesys, disc)
    @test narrayeqs_interior(sys) == 0
end

@testset "Fallback: nonlinear laplacian coefficient depends on the interface IV" begin
    @parameters t x1 x2
    @variables c1(..) c2(..)
    Dt = Differential(t)
    Dx1 = Differential(x1)
    Dx2 = Differential(x2)

    eqs = [
        Dt(c1(t, x1)) ~ Dx1((1 + x1) * c1(t, x1) * Dx1(c1(t, x1))),
        Dt(c2(t, x2)) ~ Dx2((1 + x2) * c2(t, x2) * Dx2(c2(t, x2))),
    ]
    bcs = [
        c1(0, x1) ~ 1 + 0.1 * sinpi(2x1),
        c2(0, x2) ~ 1 + 0.1 * sinpi(2x2),
        c1(t, 0) ~ 1.0,
        c1(t, 0.5) ~ c2(t, 0.5),
        c2(t, 1) ~ 1.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.02), x1 ∈ Interval(0.0, 0.5), x2 ∈ Interval(0.5, 1.0),
    ]
    @named pdesys = PDESystem(eqs, bcs, domains, [t, x1, x2], [c1(t, x1), c2(t, x2)])

    disc = MOLFiniteDifference([x1 => 0.05, x2 => 0.05], t)
    sys, _ = symbolic_discretize(pdesys, disc)
    @test narrayeqs_interior(sys) == 0
end

@testset "2D mixed derivative with a two-domain interface" begin
    # #654's tensor-product mixed stencil uses the same wrap slices; a two-domain
    # join must not send the interior back to pointwise on a uniform grid.
    @parameters t x1 x2 y
    @variables c1(..) c2(..)
    Dt = Differential(t)
    Dxx1 = Differential(x1)^2
    Dxx2 = Differential(x2)^2
    Dyy = Differential(y)^2
    Dxy1 = Differential(x1) * Differential(y)
    Dxy2 = Differential(x2) * Differential(y)

    eqs = [
        Dt(c1(t, x1, y)) ~ Dxx1(c1(t, x1, y)) + Dyy(c1(t, x1, y)) + 0.5 * Dxy1(c1(t, x1, y)),
        Dt(c2(t, x2, y)) ~ Dxx2(c2(t, x2, y)) + Dyy(c2(t, x2, y)) + 0.5 * Dxy2(c2(t, x2, y)),
    ]
    bcs = [
        c1(0, x1, y) ~ 0.0, c2(0, x2, y) ~ 0.0,
        c1(t, 0, y) ~ 0.0,
        c1(t, 0.5, y) ~ c2(t, 0.5, y),
        c2(t, 1, y) ~ 0.0,
        c1(t, x1, 0) ~ 0.0, c1(t, x1, 1) ~ 0.0,
        c2(t, x2, 0) ~ 0.0, c2(t, x2, 1) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.05),
        x1 ∈ Interval(0.0, 0.5), x2 ∈ Interval(0.5, 1.0), y ∈ Interval(0.0, 1.0),
    ]
    @named pdesys = PDESystem(
        eqs, bcs, domains, [t, x1, x2, y], [c1(t, x1, y), c2(t, x2, y)]
    )

    sol_arr, sys_arr = solve_discretized(
        pdesys, [x1 => 0.1, x2 => 0.1, y => 0.1], t
    )
    @test SciMLBase.successful_retcode(sol_arr)
    @test narrayeqs_interior(sys_arr) >= 2
end

@testset "2D two-domain interface along y" begin
    # Wrap and face slices must be valid when the interface axis is not the first
    # spatial argument.
    @parameters t x y1 y2
    @variables c1(..) c2(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    Dyy1 = Differential(y1)^2
    Dyy2 = Differential(y2)^2

    eqs = [
        Dt(c1(t, x, y1)) ~ Dxx(c1(t, x, y1)) + Dyy1(c1(t, x, y1)),
        Dt(c2(t, x, y2)) ~ Dxx(c2(t, x, y2)) + Dyy2(c2(t, x, y2)),
    ]
    bcs = [
        c1(0, x, y1) ~ sinpi(x) * sinpi(2y1),
        c2(0, x, y2) ~ sinpi(x) * sinpi(2y2),
        c1(t, 0, y1) ~ 0.0, c1(t, 1, y1) ~ 0.0,
        c2(t, 0, y2) ~ 0.0, c2(t, 1, y2) ~ 0.0,
        c1(t, x, 0) ~ 0.0,
        c1(t, x, 0.5) ~ c2(t, x, 0.5),
        c2(t, x, 1) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.05),
        x ∈ Interval(0.0, 1.0),
        y1 ∈ Interval(0.0, 0.5),
        y2 ∈ Interval(0.5, 1.0),
    ]
    @named pdesys = PDESystem(
        eqs, bcs, domains, [t, x, y1, y2], [c1(t, x, y1), c2(t, x, y2)]
    )

    sol_arr, sys_arr = solve_discretized(
        pdesys, [x => 0.2, y1 => 0.1, y2 => 0.1], t
    )
    @test SciMLBase.successful_retcode(sol_arr)
    @test narrayeqs_interior(sys_arr) >= 2
    n8 = length(
        get_eqs(
            first(
                symbolic_discretize(
                    pdesys, MOLFiniteDifference([x => 1 / 8, y1 => 0.1, y2 => 0.1], t)
                )
            )
        )
    )
    n16 = length(
        get_eqs(
            first(
                symbolic_discretize(
                    pdesys, MOLFiniteDifference([x => 1 / 16, y1 => 0.1, y2 => 0.1], t)
                )
            )
        )
    )
    @test n8 == n16
end

@testset "2D two-domain interface along y, membrane HOIB" begin
    @parameters t x y1 y2
    @variables c1(..) c2(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    Dy1 = Differential(y1)
    Dy2 = Differential(y2)
    Dyy1 = Dy1^2
    Dyy2 = Dy2^2

    eqs = [
        Dt(c1(t, x, y1)) ~ Dxx(c1(t, x, y1)) + Dyy1(c1(t, x, y1)),
        Dt(c2(t, x, y2)) ~ Dxx(c2(t, x, y2)) + Dyy2(c2(t, x, y2)),
    ]
    bcs = [
        c1(0, x, y1) ~ 0.5, c2(0, x, y2) ~ 0.5,
        c1(t, 0, y1) ~ 0.0, c2(t, 1, y2) ~ 1.0,
        Differential(x)(c1(t, 1, y1)) ~ 0.0,
        Differential(x)(c2(t, 0, y2)) ~ 0.0,
        Dy1(c1(t, x, 0)) ~ 0.0, Dy2(c2(t, x, 1)) ~ 0.0,
        Dy1(c1(t, x, 0.5)) + c1(t, x, 0.5) ~ c2(t, x, 0.5),
        Dy2(c2(t, x, 0.5)) + c2(t, x, 0.5) ~ c1(t, x, 0.5),
    ]
    domains = [
        t ∈ Interval(0.0, 0.05),
        x ∈ Interval(0.0, 1.0),
        y1 ∈ Interval(0.0, 0.5),
        y2 ∈ Interval(0.5, 1.0),
    ]
    @named pdesys = PDESystem(
        eqs, bcs, domains, [t, x, y1, y2], [c1(t, x, y1), c2(t, x, y2)]
    )

    sol_arr, sys_arr = solve_discretized(
        pdesys, [x => 0.2, y1 => 0.1, y2 => 0.1], t
    )
    @test SciMLBase.successful_retcode(sol_arr)
    @test narrayeqs_interior(sys_arr) >= 2
    @test narrayeqs(sys_arr) >= 4
end

@testset "Error: two-domain interface with incompatible argument slots" begin
    # Joining c1(t, x1, y) to c2(t, y, x2) puts the interface axis in different
    # CartesianIndex slots. A slice remap would be invented geometry.
    @parameters t x1 x2 y
    @variables c1(..) c2(..)
    Dt = Differential(t)
    Dxx1 = Differential(x1)^2
    Dxx2 = Differential(x2)^2
    Dyy = Differential(y)^2
    eq1 = Dt(c1(t, x1, y)) ~ Dxx1(c1(t, x1, y)) + Dyy(c1(t, x1, y))
    eq2 = Dt(c2(t, y, x2)) ~ Dyy(c2(t, y, x2)) + Dxx2(c2(t, y, x2))
    bcs = [
        c1(0, x1, y) ~ 0.0,
        c2(0, y, x2) ~ 0.0,
        c1(t, 0, y) ~ 0.0,
        c1(t, 0.5, y) ~ c2(t, y, 0.5),
        c2(t, y, 1) ~ 0.0,
        c1(t, x1, 0) ~ 0.0, c1(t, x1, 1) ~ 0.0,
        c2(t, 0, x2) ~ 0.0, c2(t, 1, x2) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.05),
        x1 ∈ Interval(0.0, 0.5), x2 ∈ Interval(0.5, 1.0), y ∈ Interval(0.0, 1.0),
    ]
    dxs = [x1 => 0.1, x2 => 0.1, y => 0.1]
    for (eqs, deps) in (
            ([eq1, eq2], [c1(t, x1, y), c2(t, y, x2)]),
            ([eq2, eq1], [c2(t, y, x2), c1(t, x1, y)]),
        )
        @named pdesys = PDESystem(eqs, bcs, domains, [t, x1, x2, y], deps)
        @test throws_incompatible_layout(pdesys, dxs, t)
    end
end

@testset "Error: two-domain interface with mismatched transverse grids" begin
    # Same argument slots, but the shared-index write is still invalid when the
    # non-interface axis has a different discrete length.
    @parameters t x1 x2 y1 y2
    @variables c1(..) c2(..)
    Dt = Differential(t)
    Dxx1 = Differential(x1)^2
    Dxx2 = Differential(x2)^2
    Dyy1 = Differential(y1)^2
    Dyy2 = Differential(y2)^2
    eq1 = Dt(c1(t, x1, y1)) ~ Dxx1(c1(t, x1, y1)) + Dyy1(c1(t, x1, y1))
    eq2 = Dt(c2(t, x2, y2)) ~ Dxx2(c2(t, x2, y2)) + Dyy2(c2(t, x2, y2))
    bcs = [
        c1(0, x1, y1) ~ 0.0,
        c2(0, x2, y2) ~ 0.0,
        c1(t, 0, y1) ~ 0.0,
        c1(t, 0.5, y1) ~ c2(t, 0.5, y2),
        c2(t, 1, y2) ~ 0.0,
        c1(t, x1, 0) ~ 0.0, c1(t, x1, 1) ~ 0.0,
        c2(t, x2, 0) ~ 0.0, c2(t, x2, 1) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.05),
        x1 ∈ Interval(0.0, 0.5), x2 ∈ Interval(0.5, 1.0),
        y1 ∈ Interval(0.0, 1.0), y2 ∈ Interval(0.0, 1.0),
    ]
    dxs = [x1 => 0.1, x2 => 0.1, y1 => 0.1, y2 => 0.2]
    for (eqs, deps) in (
            ([eq1, eq2], [c1(t, x1, y1), c2(t, x2, y2)]),
            ([eq2, eq1], [c2(t, x2, y2), c1(t, x1, y1)]),
        )
        @named pdesys = PDESystem(eqs, bcs, domains, [t, x1, x2, y1, y2], deps)
        @test throws_incompatible_layout(pdesys, dxs, t)
    end
end

@testset "Error: HOIB two-domain interface with incompatible argument slots" begin
    # Flux faces go through `boundary_value_maps`, not `generate_bc_eqs!(::InterfaceBoundary)`.
    @parameters t x1 x2 y
    @variables c1(..) c2(..)
    Dt = Differential(t)
    Dx1 = Differential(x1)
    Dx2 = Differential(x2)
    Dxx1 = Dx1^2
    Dxx2 = Dx2^2
    Dyy = Differential(y)^2
    eq1 = Dt(c1(t, x1, y)) ~ Dxx1(c1(t, x1, y)) + Dyy(c1(t, x1, y))
    eq2 = Dt(c2(t, y, x2)) ~ Dyy(c2(t, y, x2)) + Dxx2(c2(t, y, x2))
    bcs = [
        c1(0, x1, y) ~ 0.0,
        c2(0, y, x2) ~ 0.0,
        c1(t, 0, y) ~ 0.0,
        Dx1(c1(t, 0.5, y)) ~ Dx2(c2(t, y, 0.5)),
        c2(t, y, 1) ~ 0.0,
        c1(t, x1, 0) ~ 0.0, c1(t, x1, 1) ~ 0.0,
        c2(t, 0, x2) ~ 0.0, c2(t, 1, x2) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.05),
        x1 ∈ Interval(0.0, 0.5), x2 ∈ Interval(0.5, 1.0), y ∈ Interval(0.0, 1.0),
    ]
    dxs = [x1 => 0.1, x2 => 0.1, y => 0.1]
    for (eqs, deps) in (
            ([eq1, eq2], [c1(t, x1, y), c2(t, y, x2)]),
            ([eq2, eq1], [c2(t, y, x2), c1(t, x1, y)]),
        )
        @named pdesys = PDESystem(eqs, bcs, domains, [t, x1, x2, y], deps)
        @test throws_incompatible_layout(pdesys, dxs, t)
    end
end
