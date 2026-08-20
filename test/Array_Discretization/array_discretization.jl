# Tests for the ArrayDiscretization strategy (issue #428): the interior of each PDE is
# represented as a single symbolic array equation over slices of the array variables.
# Where both strategies can express the same substitutions, numerics match (often
# bitwise on the RHS). The array path also covers periodic-face and free-standing-corner
# boundary values that scalar `boundaryvalfuncs` leave symbolic; those cases assert the
# array path is runnable rather than scalar parity. Unsupported patterns fall back.

using MethodOfLines, ModelingToolkit, OrdinaryDiffEq, DomainSets, Symbolics
using SciMLBase
using SciMLBase: successful_retcode
using OrdinaryDiffEqRosenbrock: Rodas4
using OrdinaryDiffEqBDF: DFBDF
using OrdinaryDiffEqSSPRK: SSPRK22
using OrdinaryDiffEqLowOrderRK: SplitEuler
using NonlinearSolve: NewtonRaphson
using ModelingToolkit: get_eqs
using SymbolicUtils: symtype
using Test

# Solve pdesys with the array form and with the pointwise form it falls back to, and
# return (array_sol, pointwise_sol, array_sys). The array form discretizes to a
# `DAEProblem`, the pointwise form to an `ODEProblem`, so each arm gets its own solver.
function solve_both(pdesys, dxs, t; disc_kwargs = (;), solver = Rodas4(), kwsolve = (;))
    disc_arr = MOLFiniteDifference(
        dxs, t; discretization_strategy = ArrayDiscretization(), disc_kwargs...
    )
    disc_pt = MOLFiniteDifference(
        dxs, t; discretization_strategy = MethodOfLines.PointwiseDiscretization(),
        disc_kwargs...
    )
    sys_arr, _ = symbolic_discretize(pdesys, disc_arr)
    prob_arr = discretize(pdesys, disc_arr)
    prob_pt = discretize(pdesys, disc_pt)
    arr_solver = prob_arr isa SciMLBase.DAEProblem ? DFBDF() : solver
    sol_arr = solve(prob_arr, arr_solver; reltol = 1.0e-10, abstol = 1.0e-10, kwsolve...)
    sol_scal = solve(prob_pt, solver; reltol = 1.0e-10, abstol = 1.0e-10, kwsolve...)
    return sol_arr, sol_scal, sys_arr
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
    @test narrayeqs_interior(sys_arr) == 1
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
    @test narrayeqs_interior(sys_arr) == 1
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
    @test narrayeqs_interior(sys_arr) == 1
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
    @test narrayeqs_interior(sys_arr) == 1
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
    @test narrayeqs_interior(sys_arr) == 1
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
    @test narrayeqs_interior(sys_arr) == 1
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
    @test narrayeqs_interior(sys_arr) == 1
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
    @test narrayeqs_interior(sys_arr) == 1
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
    @test narrayeqs_interior(sys_arr) == 2
    @test sol_arr[u(t, x)] ≈ sol_scal[u(t, x)] rtol = 1.0e-6
    @test sol_arr[v(t, x)] ≈ sol_scal[v(t, x)] rtol = 1.0e-6
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

    sol_arr, sol_scal, sys_arr = solve_both(pdesys, [x => 0.05], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    # The stencil is translation invariant across the whole periodic direction, so the
    # interior is one array equation plus the points whose stencils wrap over the seam,
    # which in 1D are single points and stay scalar.
    @test narrayeqs_interior(sys_arr) == 1
    @test sol_arr[u(t, x)] ≈ sol_scal[u(t, x)] rtol = 1.0e-6

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

    sol_arr, sol_scal, sys_arr = solve_both(pdesys, [x => 0.02], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
    @test sol_arr[u(t, x)] ≈ sol_scal[u(t, x)] rtol = 1.0e-6
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

    sol_arr, sol_scal, sys_arr = solve_both(pdesys, [x => 0.1, y => 0.1], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    # One equation for the points whose stencils do not wrap, one for each of the four
    # slabs along the seams, and a scalar one for each of the four points where two seams
    # meet: the slabs are the array equations that make this scale.
    @test narrayeqs_interior(sys_arr) == 5
    @test sol_arr[u(t, x, y)] ≈ sol_scal[u(t, x, y)] rtol = 1.0e-6

    # and the equation count does not grow with the grid
    counts = map([8, 16]) do n
        disc = MOLFiniteDifference(
            [x => 1 / (n - 1), y => 1 / (n - 1)], t;
            discretization_strategy = ArrayDiscretization()
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

    sol_arr, sol_scal, sys_arr = solve_both(pdesys, [x => 0.1, y => 0.1], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    # the core plus one slab per seam in x; y contributes no wrapping
    @test narrayeqs_interior(sys_arr) == 3
    @test sol_arr[u(t, x, y)] ≈ sol_scal[u(t, x, y)] rtol = 1.0e-6
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
            [x => 1 / n, y => 1 / n], t; discretization_strategy = ArrayDiscretization()
        )
        sys, _ = symbolic_discretize(pdesys, disc)
        (; narr = narrayeqs_interior(sys), n = length(get_eqs(sys)))
    end
    @test counts[1].narr == 10
    @test counts[1].n == counts[2].n

    sols = map([ArrayDiscretization(), MethodOfLines.PointwiseDiscretization()]) do st
        disc = MOLFiniteDifference(
            [x => 1 / 8, y => 1 / 8], t; discretization_strategy = st
        )
        sol = solve(
            discretize(pdesys, disc), Rodas4();
            reltol = 1.0e-10, abstol = 1.0e-10, saveat = 0.5
        )
        (sol[u(x, y, t)], sol[v(x, y, t)])
    end
    @test sols[1][1] ≈ sols[2][1] rtol = 1.0e-8
    @test sols[1][2] ≈ sols[2][2] rtol = 1.0e-8
end

@testset "Fallback: interface joining two variables on two domains" begin
    # An interface between two domains shifts the stencil taps onto another variable's
    # array; that has no slice form here, so it must fall back rather than error.
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

    sol_arr, sol_scal, sys_arr = solve_both(pdesys, [x1 => 0.05, x2 => 0.05], t)
    @test SciMLBase.successful_retcode(sol_arr)
    @test narrayeqs_interior(sys_arr) == 0
    @test sol_arr[c1(t, x1)] ≈ sol_scal[c1(t, x1)] rtol = 1.0e-6
    @test sol_arr[c2(t, x2)] ≈ sol_scal[c2(t, x2)] rtol = 1.0e-6
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

    sol_arr, sol_scal, sys_arr = solve_both(pdesys, [x => 0.05], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    # The half-offset coefficient expression is translation invariant over the core, so
    # the interior collapses to one array equation (issue #623).
    @test narrayeqs_interior(sys_arr) == 1
    @test sol_arr[u(t, x)] ≈ sol_scal[u(t, x)] rtol = 1.0e-6

    # and the equation count does not grow with the grid
    counts = map([21, 41]) do n
        disc = MOLFiniteDifference(
            [x => 1 / (n - 1)], t; discretization_strategy = ArrayDiscretization()
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

    sol_arr, sol_scal, sys_arr = solve_both(pdesys, [x => 0.04], t)
    @test SciMLBase.successful_retcode(sol_arr)
    @test narrayeqs_interior(sys_arr) == 1
    @test sol_arr[u(t, x)] ≈ sol_scal[u(t, x)] rtol = 1.0e-6

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

    sol_arr, sol_scal, sys_arr = solve_both(pdesys, [x => 0.05], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
    @test sol_arr[u(t, x)] ≈ sol_scal[u(t, x)] rtol = 1.0e-6
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
    sol_arr, sol_scal, sys_arr = solve_both(pdesys_div, [x => 0.05], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
    @test sol_arr[u(t, x)] ≈ sol_scal[u(t, x)] rtol = 1.0e-6

    # A grid-constant prefactor and a parameter divisor on the whole laplacian
    eq = Dt(u(t, x)) ~ 3.0 * Dx(u(t, x) * Dx(u(t, x))) / p
    @named pdesys_pre = PDESystem(eq, bcs, domains, [t, x], [u(t, x)], [p => 2.0])
    sol_arr, sol_scal, sys_arr = solve_both(pdesys_pre, [x => 0.05], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
    @test sol_arr[u(t, x)] ≈ sol_scal[u(t, x)] rtol = 1.0e-6
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

    sol_arr, sol_scal, sys_arr = solve_both(
        pdesys, [x => 0.05], t; disc_kwargs = (; approx_order = 4)
    )
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
    @test sol_arr[u(t, x)] ≈ sol_scal[u(t, x)] rtol = 1.0e-6
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

    sol_arr, sol_scal, sys_arr = solve_both(pdesys, [x => 0.05], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
    @test sol_arr[u(t, x)] ≈ sol_scal[u(t, x)] rtol = 1.0e-6
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
    sol_arr, sol_scal, sys_arr = solve_both(pdesys, [x => gridvec], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
    @test sol_arr[u(t, x)] ≈ sol_scal[u(t, x)] rtol = 1.0e-6
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

    sol_arr, sol_scal, sys_arr = solve_both(pdesys, [x => 0.1, y => 0.1], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
    @test sol_arr[u(t, x, y)] ≈ sol_scal[u(t, x, y)] rtol = 1.0e-6
end

@testset "1D spherical laplacian" begin
    # Cardinalization rewrites Dt(u) ~ Dr(r^2*Dr(u))/r^2 with a Mul numerator, the shape
    # the scalar path discretizes through the nonlinear laplacian rules: r^2 enters at
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

    sol_arr, sol_scal, sys_arr = solve_both(pdesys, [r => 0.1], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
    @test sol_arr[u(t, r)] ≈ sol_scal[u(t, r)] rtol = 1.0e-6

    rdisc = sol_arr[r][2:(end - 1)]
    for (i, ti) in enumerate(sol_arr[t])
        @test sol_arr[u(t, r)][i, 2:(end - 1)] ≈ u_exact.(rdisc, ti) atol = 0.05
    end

    # and the equation count does not grow with the grid
    counts = map([21, 41]) do n
        disc = MOLFiniteDifference(
            [r => 1 / (n - 1)], t; discretization_strategy = ArrayDiscretization()
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

    sol_arr, sol_scal, sys_arr = solve_both(
        pdesys, [r => 0.1], t; disc_kwargs = (; approx_order = 4)
    )
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
    @test sol_arr[u(t, r)] ≈ sol_scal[u(t, r)] rtol = 1.0e-6
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

    sol_arr, sol_scal, sys_arr = solve_both(pdesys, [r => 0.1], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
    @test sol_arr[u(t, r)] ≈ sol_scal[u(t, r)] rtol = 1.0e-6

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

    sol_arr, sol_scal, sys_arr = solve_both(pdesys, [r => 0.1], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
    @test sol_arr[u(t, r)] ≈ sol_scal[u(t, r)] rtol = 1.0e-6
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
    sol_arr, sol_scal, sys_arr = solve_both(pdesys, [r => gridvec], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
    @test sol_arr[u(t, r)] ≈ sol_scal[u(t, r)] rtol = 1.0e-6
end

@testset "Fallback: grid-varying factor multiplying a nonlinear laplacian" begin
    # The scalar path leaves such factors undiscretized (a pre-existing scalar-path
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
        [x => 0.05], t; discretization_strategy = ArrayDiscretization()
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

    sol_arr, sol_scal, sys_arr = solve_both(
        pdesys, [x => 0.02], t;
        disc_kwargs = (; advection_scheme = WENOScheme()), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-3)
    )
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
    @test sol_arr[u(t, x)] == sol_scal[u(t, x)]

    # the scheme is traced once, so the equation count is resolution independent
    counts = map([51, 101]) do n
        disc = MOLFiniteDifference(
            [x => 1 / (n - 1)], t; advection_scheme = WENOScheme(),
            discretization_strategy = ArrayDiscretization()
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

    sol_arr, sol_scal, sys_arr = solve_both(
        pdesys, [x => 0.02], t;
        disc_kwargs = (; advection_scheme = WENOScheme()), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-4)
    )
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
    @test sol_arr[u(t, x)] == sol_scal[u(t, x)]
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

    sol_arr, sol_scal, sys_arr = solve_both(
        pdesys, [x => 0.02], t;
        disc_kwargs = (; advection_scheme = WENOScheme()), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-3)
    )
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    # one array equation for the points whose taps do not wrap, the rest pointwise
    @test narrayeqs_interior(sys_arr) == 1
    @test sol_arr[u(t, x)] == sol_scal[u(t, x)]

    # the number of wrap points is fixed by the stencil, so the equation count is
    # resolution independent
    counts = map([51, 101]) do n
        disc = MOLFiniteDifference(
            [x => 1 / (n - 1)], t; advection_scheme = WENOScheme(),
            discretization_strategy = ArrayDiscretization()
        )
        sys, _ = symbolic_discretize(pdesys, disc)
        length(get_eqs(sys))
    end
    @test counts[1] == counts[2]

    # a periodic nonuniform direction goes through the coefficient split: seam windows
    # take the periodically shifted coordinates `bcoord` produces, so parity is bitwise
    gridvec = [0.5 * (1 - cospi(i / 50)) for i in 0:50]
    solp_arr, solp_scal, sysp_arr = solve_both(
        pdesys, [x => gridvec], t;
        disc_kwargs = (; advection_scheme = WENOScheme()), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-3)
    )
    @test solp_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sysp_arr) == 1
    @test solp_arr[u(t, x)] == solp_scal[u(t, x)]
    strictp = MOLFiniteDifference(
        [x => gridvec], t; discretization_strategy = StrictArrayDiscretization(),
        advection_scheme = WENOScheme()
    )
    sysp_strict, _ = symbolic_discretize(pdesys, strictp)
    @test narrayeqs_interior(sysp_strict) == 1

    # resolution independence holds on the periodic nonuniform path too
    pcounts = map([40, 80]) do n
        disc = MOLFiniteDifference(
            [x => [0.5 * (1 - cospi(i / n)) for i in 0:n]], t;
            advection_scheme = WENOScheme(),
            discretization_strategy = ArrayDiscretization()
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
    sol_arr, sol_scal, sys_arr = solve_both(
        pdesys, [x => gridvec], t;
        disc_kwargs = (; advection_scheme = WENOScheme()), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-4)
    )
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
    @test sol_arr[u(t, x)] == sol_scal[u(t, x)]

    # the kernel is traced once, so the equation count is resolution independent
    counts = map([40, 80]) do n
        disc = MOLFiniteDifference(
            [x => [0.5 * (1 - cospi(i / n)) for i in 0:n]], t;
            advection_scheme = WENOScheme(),
            discretization_strategy = ArrayDiscretization()
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
    sol_arr, sol_scal, sys_arr = solve_both(
        advdiff, [x => cosgrid(40)], t;
        disc_kwargs = (; advection_scheme = WENOScheme()), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-4)
    )
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
    @test sol_arr[u(t, x)] == sol_scal[u(t, x)]

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
    solc_arr, solc_scal, sysc_arr = solve_both(
        coupled, [x => cosgrid(40)], t;
        disc_kwargs = (; advection_scheme = WENOScheme()), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-4)
    )
    @test solc_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sysc_arr) == 2
    @test maximum(abs.(solc_arr[u(t, x)] .- solc_scal[u(t, x)])) < 1.0e-12
    @test maximum(abs.(solc_arr[v(t, x)] .- solc_scal[v(t, x)])) < 1.0e-12

    # smallest representable grid (n = 7: three-point core, frame at 2 and 6)
    @named adv = PDESystem(
        Dt(u(t, x)) ~ -Dx(u(t, x)),
        [u(0, x) ~ exp(-100 * (x - 0.3)^2), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0],
        domains, [t, x], [u(t, x)]
    )
    sol7_arr, sol7_scal, sys7_arr = solve_both(
        adv, [x => cosgrid(6)], t;
        disc_kwargs = (; advection_scheme = WENOScheme()), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-4)
    )
    @test sol7_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys7_arr) == 1
    @test sol7_arr[u(t, x)] == sol7_scal[u(t, x)]
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
    # scalar path has no reference to match. Accuracy is checked against the exact
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
    disc2(strat) = MOLFiniteDifference(
        [x => cosgrid(20), y => 0.05], t; advection_scheme = WENOScheme(),
        discretization_strategy = strat
    )
    @test_throws AssertionError symbolic_discretize(
        pdesys2, disc2(ScalarizedDiscretization())
    )

    strict2 = disc2(StrictArrayDiscretization())
    sys2, _ = symbolic_discretize(pdesys2, strict2)
    interior2 = filter(isinterioreq, get_eqs(sys2))
    @test !isempty(interior2) && all(isarrayeq, interior2)
    sol2 = solve(discretize(pdesys2, strict2), SSPRK22(); dt = 1.0e-4)
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
    @test_throws MethodOfLines.ArrayDiscretizationError symbolic_discretize(
        pdesys2m, strict2
    )

    # smallest grid the boundary extrapolator admits (7 points); every window wraps
    @named adv = PDESystem(
        Dt(u(t, x)) ~ -Dx(u(t, x)),
        [u(0, x) ~ sinpi(2x), u(t, 0) ~ u(t, 1)],
        dom1, [t, x], [u(t, x)]
    )
    sol7_arr, sol7_scal, _ = solve_both(
        adv, [x => cosgrid(6)], t;
        disc_kwargs = (; advection_scheme = WENOScheme()), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-4)
    )
    @test sol7_arr.retcode == SciMLBase.ReturnCode.Success
    @test sol7_arr[u(t, x)] == sol7_scal[u(t, x)]

    # linear stencils still have no seam form on a nonuniform grid
    @named advdiffp = PDESystem(
        Dt(u(t, x)) ~ -Dx(u(t, x)) + 0.05 * Dxx(u(t, x)),
        [u(0, x) ~ sinpi(2x), u(t, 0) ~ u(t, 1)],
        dom1, [t, x], [u(t, x)]
    )
    strictmix = MOLFiniteDifference(
        [x => cosgrid(20)], t; discretization_strategy = StrictArrayDiscretization(),
        advection_scheme = WENOScheme()
    )
    @test_throws MethodOfLines.ArrayDiscretizationError symbolic_discretize(
        advdiffp, strictmix
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
    sol_arr, sol_scal, sys_arr = solve_both(
        pdesys, [x => 0.05, y => 0.05], t;
        disc_kwargs = (; advection_scheme = WENOScheme()), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-3)
    )
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
    @test sol_arr[w(t, x, y)] == sol_scal[w(t, x, y)]

    # mixed grid: x traces the uniform kernel, y goes through the coefficient split
    gridvec = [0.5 * (1 - cospi(i / 20)) for i in 0:20]
    solm_arr, solm_scal, sysm_arr = solve_both(
        pdesys, [x => 0.05, y => gridvec], t;
        disc_kwargs = (; advection_scheme = WENOScheme()), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-3)
    )
    @test solm_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sysm_arr) == 1
    @test solm_arr[w(t, x, y)] == solm_scal[w(t, x, y)]

    # a trace and central differences in one 2D equation
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    @named advdiff2d = PDESystem(
        Dt(w(t, x, y)) ~ -Dx(w(t, x, y)) - Dy(w(t, x, y)) +
            0.05 * (Dxx(w(t, x, y)) + Dyy(w(t, x, y))),
        bcs2, dom2, [t, x, y], [w(t, x, y)]
    )
    sold_arr, sold_scal, sysd_arr = solve_both(
        advdiff2d, [x => 0.05, y => 0.05], t;
        disc_kwargs = (; advection_scheme = WENOScheme()), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-3)
    )
    @test sold_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sysd_arr) == 1
    @test sold_arr[w(t, x, y)] == sold_scal[w(t, x, y)]

    # and both forms pass strict mode
    strict2 = MOLFiniteDifference(
        [x => 0.05, y => gridvec], t;
        discretization_strategy = StrictArrayDiscretization(),
        advection_scheme = WENOScheme()
    )
    sys_strict2, _ = symbolic_discretize(pdesys, strict2)
    @test narrayeqs_interior(sys_strict2) == 1
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
    sol_arr, sol_scal, sys_arr = solve_both(
        pdesys, [x => 0.02], t;
        disc_kwargs = (; advection_scheme = scheme), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-3)
    )
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
    @test sol_arr[u(t, x)] == sol_scal[u(t, x)]

    # A scheme that reads the grid coordinate falls back: a trace cannot reproduce the
    # scalar path's Float64 coordinate folds digit for digit.
    coord3(u, p, t, x, dx) = (u[3] - u[1]) / (x[3] - x[1])
    xscheme = FunctionalScheme{3, 1}(
        coord3, [nothing], [nothing], false, []; name = "coord3"
    )
    sol_arr2, sol_scal2, sys_arr2 = solve_both(
        pdesys, [x => 0.02], t;
        disc_kwargs = (; advection_scheme = xscheme), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-3)
    )
    @test sol_arr2.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr2) == 0
    @test sol_arr2[u(t, x)] == sol_scal2[u(t, x)]

    # A nonuniform scheme without an `array_scheme_split` falls back; the pointwise
    # result is untouched.
    nu3(u, p, t, x, dx) = (u[3] - u[1]) / (x[3] - x[1])
    nuscheme = FunctionalScheme{3, 1}(
        nu3, [nothing], [nothing], true, []; name = "nu3"
    )
    gridvec = [0.5 * (1 - cospi(i / 40)) for i in 0:40]
    sol_arr3, sol_scal3, sys_arr3 = solve_both(
        pdesys, [x => gridvec], t;
        disc_kwargs = (; advection_scheme = nuscheme), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-4)
    )
    @test sol_arr3.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr3) == 0
    @test sol_arr3[u(t, x)] == sol_scal3[u(t, x)]

    # A trace failure (hard branch on coordinates, which are symbols while tracing)
    # falls back, not errors.
    xbranch(u, p, t, x, dx) = x[3] > 0.5 ? (u[3] - u[2]) / dx : (u[2] - u[1]) / dx
    bscheme = FunctionalScheme{3, 1}(
        xbranch, [nothing], [nothing], false, []; name = "xbranch"
    )
    sol_arr4, sol_scal4, sys_arr4 = solve_both(
        pdesys, [x => 0.02], t;
        disc_kwargs = (; advection_scheme = bscheme), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-3)
    )
    @test sol_arr4.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr4) == 0
    @test sol_arr4[u(t, x)] == sol_scal4[u(t, x)]

    # A time-dependent flux traces fine: `t` stays symbolic in the array equation.
    tflux(u, p, t, x, dx) = (1 + 0.1 * t) * (u[3] - u[1]) / (2 * dx)
    tscheme = FunctionalScheme{3, 1}(
        tflux, [nothing], [nothing], false, []; name = "tflux"
    )
    sol_arr5, sol_scal5, sys_arr5 = solve_both(
        pdesys, [x => 0.02], t;
        disc_kwargs = (; advection_scheme = tscheme), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-3)
    )
    @test sol_arr5.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr5) == 1
    @test sol_arr5[u(t, x)] == sol_scal5[u(t, x)]
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

    sol_arr, sol_scal, sys_arr = solve_both(
        pdesys, [x => 0.02], t;
        disc_kwargs = (; advection_scheme = WENOScheme()), solver = SSPRK22(),
        kwsolve = (; dt = 1.0e-4)
    )
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
    @test maximum(abs.(sol_arr[u(t, x)] .- sol_scal[u(t, x)])) < 1.0e-14
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
        [x => 0.05]; discretization_strategy = MethodOfLines.PointwiseDiscretization()
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

    disc_arr = MOLFiniteDifference(
        [x => 0.1], t; discretization_strategy = ArrayDiscretization()
    )
    disc_scal = MOLFiniteDifference(
        [x => 0.1], t; discretization_strategy = MethodOfLines.PointwiseDiscretization()
    )
    prob_arr = discretize(pdesys, disc_arr)
    prob_scal = discretize(pdesys, disc_scal)
    sol_arr = solve(prob_arr, Rodas4(); reltol = 1.0e-10, abstol = 1.0e-10, saveat = 0.2)
    sol_scal = solve(prob_scal, Rodas4(); reltol = 1.0e-10, abstol = 1.0e-10, saveat = 0.2)
    @test SciMLBase.successful_retcode(sol_arr)
    for i in 1:n_comp
        @test sol_arr[u(t, x)[i]] ≈ sol_scal[u(t, x)[i]] rtol = 1.0e-6
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
    # The scalar path emits `ifelse(coef > 0, coef*pos, coef*neg)`. When the coefficient
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
        sol_arr, sol_scal, sys_arr = solve_both(pdesys, [x => 0.05], t)
        @test any(eq -> hasoperation(eq, ifelse), get_eqs(sys_arr))
        @test narrayeqs_interior(sys_arr) == 1
        @test sol_arr[u(t, x)] ≈ sol_scal[u(t, x)] rtol = 1.0e-6
    end

    # A coefficient that varies over the grid still needs the per-point surrogate, since
    # `ifelse` cannot broadcast over a symbolic array condition.
    eq = Dt(u(t, x)) ~ -(1 + x) * Dx(u(t, x)) + 0.05 * Dxx(u(t, x))
    @named pdesys_x = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
    sol_arr, sol_scal, sys_arr = solve_both(pdesys_x, [x => 0.05], t)
    @test !any(eq -> hasoperation(eq, ifelse), get_eqs(sys_arr))
    @test sol_arr[u(t, x)] ≈ sol_scal[u(t, x)] rtol = 1.0e-6
end

@testset "ArrayDiscretization is the default" begin
    # v1 removed `ScalarizedDiscretization`: the array form is the only strategy, and the
    # scalar form is its fallback for patterns with no slice representation.
    @parameters t x
    disc_default = MOLFiniteDifference([x => 0.1], t)
    @test disc_default.disc_strategy isa ArrayDiscretization

    disc_steady = MOLFiniteDifference([x => 0.1])
    @test disc_steady.disc_strategy isa ArrayDiscretization

    disc_opt = MOLFiniteDifference(
        [x => 0.1], t; discretization_strategy = ArrayDiscretization()
    )
    @test disc_opt.disc_strategy isa ArrayDiscretization
end

@testset "Interior representation is independent of grid resolution" begin
    # The point of the array form: one interior equation whose expression size does not
    # grow with the grid, where the scalarized form emits one equation per point. This is
    # asserted structurally rather than by timing, but it is what makes generated code
    # compile in constant rather than linear time (see the PR benchmark).
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
        disc_a = MOLFiniteDifference(
            [x => 1 / (n - 1)], t; discretization_strategy = ArrayDiscretization()
        )
        disc_s = MOLFiniteDifference(
            [x => 1 / (n - 1)], t;
            discretization_strategy = MethodOfLines.PointwiseDiscretization()
        )
        sys_a, _ = symbolic_discretize(pdesys, disc_a)
        sys_s, _ = symbolic_discretize(pdesys, disc_s)
        (;
            n, arr = interior_size(sys_a), scal = interior_size(sys_s),
            narr = narrayeqs_interior(sys_a),
        )
    end

    coarse, fine = sizes
    # one array equation regardless of resolution
    @test coarse.narr == 1
    @test fine.narr == 1
    # array interior expression does not grow with the grid
    @test fine.arr == coarse.arr
    # the scalarized interior does grow, roughly in proportion to the point count
    @test fine.scal > 3 * coarse.scal
end

@testset "StrictArrayDiscretization errors instead of falling back" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)]
    strict = MOLFiniteDifference(
        [x => 0.1], t; discretization_strategy = StrictArrayDiscretization()
    )

    # Patterns with no slice representation must raise rather than silently discretize
    # pointwise.
    unsupported = [
        (
            "off-edge boundary sampling",
            Dt(u(t, x)) ~ Dxx(u(t, x)) + u(t, 0.5),
            [u(0, x) ~ sinpi(x), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0],
        ),
        (
            "derivative of boundary value",
            Dt(u(t, x)) ~ Dxx(u(t, x)) + Dx(u(t, 1)),
            [u(0, x) ~ sinpi(x), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0],
        ),
        (
            "time-literal in interior",
            Dt(u(t, x)) ~ Dxx(u(t, x)) + u(0, x),
            [u(0, x) ~ sinpi(x), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0],
        ),
    ]
    for (name, eq, bcs) in unsupported
        @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
        err = try
            symbolic_discretize(pdesys, strict)
            nothing
        catch e
            e
        end
        @test err isa MethodOfLines.ArrayDiscretizationError
        @test !occursin("BoundsError", err.msg)
        if name == "time-literal in interior"
            @test occursin("time-literal", err.msg)
        end
        # the permissive strategy still handles it, pointwise
        lenient = MOLFiniteDifference(
            [x => 0.1], t; discretization_strategy = ArrayDiscretization()
        )
        sys, _ = symbolic_discretize(pdesys, lenient)
        @test narrayeqs_interior(sys) == 0
    end

    @named bval_sys = PDESystem(
        Dt(u(t, x)) ~ Dxx(u(t, x)) + u(t, 1),
        [u(0, x) ~ sinpi(x), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0],
        domains, [t, x], [u(t, x)]
    )
    sys_bval, _ = symbolic_discretize(bval_sys, strict)
    @test narrayeqs_interior(sys_bval) == 1
    sol_bval_arr, sol_bval_scal, _ = solve_both(bval_sys, [x => 0.1], t)
    @test maximum(abs.(sol_bval_arr[u(t, x)] .- sol_bval_scal[u(t, x)])) == 0.0

    # A supported equation must go through strict mode unchanged.
    eq = Dt(u(t, x)) ~ Dxx(u(t, x))
    bcs = [u(0, x) ~ sinpi(x), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0]
    @named ok_sys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
    sys_strict, _ = symbolic_discretize(ok_sys, strict)
    @test narrayeqs_interior(sys_strict) == 1

    # Nonlinear laplacians are supported and must go through strict mode too.
    @named nllap_sys = PDESystem(
        Dt(u(t, x)) ~ Dx(u(t, x) * Dx(u(t, x))),
        [u(0, x) ~ 1.0 + sinpi(x) / 2, u(t, 0) ~ 1.0, u(t, 1) ~ 1.0],
        domains, [t, x], [u(t, x)]
    )
    sys_nllap, _ = symbolic_discretize(nllap_sys, strict)
    @test narrayeqs_interior(sys_nllap) == 1

    # Spherical laplacians too, in both cardinalized shapes.
    @named sph_sys = PDESystem(
        Dt(u(t, x)) ~ Dx(x^2 * Dx(u(t, x))) / x^2,
        [u(0, x) ~ 1.0 + sinpi(x) / 2, u(t, 0) ~ 1.0, u(t, 1) ~ 1.0],
        domains, [t, x], [u(t, x)]
    )
    sys_sph, _ = symbolic_discretize(sph_sys, strict)
    @test narrayeqs_interior(sys_sph) == 1
    @named sph_bare_sys = PDESystem(
        Dx(x^2 * Dx(u(t, x))) / x^2 ~ Dt(u(t, x)),
        [u(0, x) ~ 1.0 + sinpi(x) / 2, u(t, 0) ~ 1.0, u(t, 1) ~ 1.0],
        domains, [t, x], [u(t, x)]
    )
    sys_sph_bare, _ = symbolic_discretize(sph_bare_sys, strict)
    @test narrayeqs_interior(sys_sph_bare) == 1

    # Periodic boundaries are supported: the points whose stencils wrap over the seam are
    # pointwise for the same structural reason the frame is, so strict mode accepts them.
    @named periodic_sys = PDESystem(
        eq, [u(0, x) ~ sinpi(2x), u(t, 0) ~ u(t, 1)], domains, [t, x], [u(t, x)]
    )
    sys_periodic, _ = symbolic_discretize(periodic_sys, strict)
    @test narrayeqs_interior(sys_periodic) == 1

    # Frame points near a boundary are pointwise under either strategy because their
    # stencils genuinely differ; that is structural, not an unsupported pattern, so
    # strict mode must accept it.
    strict4 = MOLFiniteDifference(
        [x => 0.05], t; discretization_strategy = StrictArrayDiscretization(),
        approx_order = 4
    )
    sys4, _ = symbolic_discretize(ok_sys, strict4)
    @test narrayeqs_interior(sys4) == 1
    @test length(get_eqs(sys4)) > 3   # array interior + BCs + scalar frame equations

    # WENO is supported on uniform and (through its coefficient split) nonuniform
    # grids, including periodic nonuniform directions; a nonuniform scheme without a
    # split has no traced form and errors.
    @named adv = PDESystem(
        Dt(u(t, x)) ~ -Dx(u(t, x)),
        [u(0, x) ~ sinpi(x), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0],
        domains, [t, x], [u(t, x)]
    )
    strict_nu = MOLFiniteDifference(
        [x => [0.5 * (1 - cospi(i / 20)) for i in 0:20]], t;
        discretization_strategy = StrictArrayDiscretization(),
        advection_scheme = WENOScheme()
    )
    sys_nu, _ = symbolic_discretize(adv, strict_nu)
    @test narrayeqs_interior(sys_nu) == 1
    @named advp = PDESystem(
        Dt(u(t, x)) ~ -Dx(u(t, x)),
        [u(0, x) ~ sinpi(2x), u(t, 0) ~ u(t, 1)],
        domains, [t, x], [u(t, x)]
    )
    sys_nup, _ = symbolic_discretize(advp, strict_nu)
    @test narrayeqs_interior(sys_nup) == 1
    strict_weno = MOLFiniteDifference(
        [x => 0.05], t; discretization_strategy = StrictArrayDiscretization(),
        advection_scheme = WENOScheme()
    )
    sys_weno, _ = symbolic_discretize(adv, strict_weno)
    @test narrayeqs_interior(sys_weno) == 1
    nu3strict(u_, p_, t_, x_, dx_) = (u_[3] - u_[1]) / (x_[3] - x_[1])
    strict_nosplit = MOLFiniteDifference(
        [x => [0.5 * (1 - cospi(i / 20)) for i in 0:20]], t;
        discretization_strategy = StrictArrayDiscretization(),
        advection_scheme = FunctionalScheme{3, 1}(
            nu3strict, [nothing], [nothing], true, []; name = "nu3"
        )
    )
    @test_throws MethodOfLines.ArrayDiscretizationError symbolic_discretize(
        adv, strict_nosplit
    )

    # The error names the offending equation and the reason.
    @named bad = PDESystem(
        Dt(u(t, x)) ~ Dxx(u(t, x)) + Dx(u(t, 1)),
        [u(0, x) ~ sinpi(x), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0],
        domains, [t, x], [u(t, x)]
    )
    msg = try
        symbolic_discretize(bad, strict)
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("StrictArrayDiscretization", msg)
    @test occursin("Reason:", msg)
    @test occursin("ArrayDiscretization()", msg)
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
            [x => 1 / (n - 1), y => 1 / (n - 1)], t;
            discretization_strategy = ArrayDiscretization()
        )
        sys, _ = symbolic_discretize(pdesys, disc)
        length(get_eqs(sys))
    end
    # total equation count is independent of resolution in 2D
    @test counts[1] == counts[2]
    # one interior equation plus one per face
    @test counts[1] <= 12

    # and the solution still matches the scalar path exactly
    sols = map(
        [ArrayDiscretization(), MethodOfLines.PointwiseDiscretization()]
    ) do st
        disc = MOLFiniteDifference(
            [x => 0.1, y => 0.1], t; discretization_strategy = st
        )
        solve(
            discretize(pdesys, disc), Rodas4();
            reltol = 1.0e-10, abstol = 1.0e-10, saveat = 0.025
        )[u(t, x, y)]
    end
    @test sols[1] ≈ sols[2] rtol = 1.0e-8
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

    sol_arr, sol_scal, sys_arr = solve_both(pdesys, [x => 0.05], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
    @test maximum(abs.(sol_arr[u(t, x)] .- sol_scal[u(t, x)])) == 0.0
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
            [x => 1 / (n - 1)], t; discretization_strategy = ArrayDiscretization()
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
    disc_arr = MOLFiniteDifference(
        dxs, t; discretization_strategy = ArrayDiscretization()
    )
    disc_scal = MOLFiniteDifference(
        dxs, t; discretization_strategy = MethodOfLines.PointwiseDiscretization()
    )
    sys_arr, _ = symbolic_discretize(pdesys, disc_arr)
    prob_arr = discretize(pdesys, disc_arr)
    prob_scal = discretize(pdesys, disc_scal)

    @test narrayeqs_interior(sys_arr) == 1
    int_eq = only(filter(eq -> isinterioreq(eq) && isarrayeq(eq), get_eqs(sys_arr)))
    @test occursin("[21]", string(int_eq))

    du_arr = similar(prob_arr.u0)
    du_scal = similar(prob_scal.u0)
    prob_arr.f(du_arr, prob_arr.u0, prob_arr.p, 0.0)
    prob_scal.f(du_scal, prob_scal.u0, prob_scal.p, 0.0)
    @test du_arr == du_scal
    @test maximum(abs.(du_arr)) > 1

    sol_arr, sol_scal, _ = solve_both(pdesys, dxs, t)
    @test SciMLBase.successful_retcode(sol_arr)
    @test sol_arr[u(t, x)] ≈ sol_scal[u(t, x)] rtol = 1.0e-6
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
    disc2_arr = MOLFiniteDifference(
        dxs2, t; discretization_strategy = ArrayDiscretization()
    )
    disc2_scal = MOLFiniteDifference(
        dxs2, t; discretization_strategy = MethodOfLines.PointwiseDiscretization()
    )
    sys2, _ = symbolic_discretize(pdesys2, disc2_arr)
    prob2_arr = discretize(pdesys2, disc2_arr)
    prob2_scal = discretize(pdesys2, disc2_scal)

    @test narrayeqs_interior(sys2) == 1
    face_eq = only(filter(eq -> isinterioreq(eq) && isarrayeq(eq), get_eqs(sys2)))
    @test occursin("1:1", string(face_eq))

    du2_arr = similar(prob2_arr.u0)
    du2_scal = similar(prob2_scal.u0)
    prob2_arr.f(du2_arr, prob2_arr.u0, prob2_arr.p, 0.0)
    prob2_scal.f(du2_scal, prob2_scal.u0, prob2_scal.p, 0.0)
    @test du2_arr == du2_scal
    @test maximum(abs.(du2_arr)) > 1

    sol2_arr, sol2_scal, _ = solve_both(pdesys2, dxs2, t)
    @test SciMLBase.successful_retcode(sol2_arr)
    @test sol2_arr[u(t, x, y)] ≈ sol2_scal[u(t, x, y)] rtol = 1.0e-6
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
    sol_arr, sol_scal, sys_arr = solve_both(face_sys, [x => 0.1, y => 0.1], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
    @test maximum(abs.(sol_arr[u(t, x, y)] .- sol_scal[u(t, x, y)])) == 0.0
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
        dxs_c, t; discretization_strategy = ArrayDiscretization()
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

    prob_c = discretize(corner_sys, disc)
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

    sol_arr, sol_scal, sys_arr = solve_both(pdesys, [x => 0.05], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 2
    @test maximum(abs.(sol_arr[u(t, x)] .- sol_scal[u(t, x)])) == 0.0
    @test maximum(abs.(sol_arr[v(t, x)] .- sol_scal[v(t, x)])) == 0.0
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
    sol_arr, sol_scal, sys_arr = solve_both(pdesys, [x => gridvec], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
    @test maximum(abs.(sol_arr[u(t, x)] .- sol_scal[u(t, x)])) == 0.0
end

@testset "Boundary value as upwind coefficient stays in array form" begin
    # Scalar path also leaves u(t,1) symbolic inside winding coefficients, so no parity.
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
        dxs, t; discretization_strategy = ArrayDiscretization()
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

    strict = MOLFiniteDifference(
        dxs, t; discretization_strategy = StrictArrayDiscretization()
    )
    sys_strict, _ = symbolic_discretize(pdesys, strict)
    @test narrayeqs_interior(sys_strict) == 1

    prob = discretize(pdesys, disc)
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
        dxs, t; discretization_strategy = ArrayDiscretization()
    )
    sys, _ = symbolic_discretize(pdesys, disc)
    @test narrayeqs_interior(sys) == 1
    per_eq = only(filter(eq -> isinterioreq(eq) && isarrayeq(eq), get_eqs(sys)))
    @test occursin("[1]", string(per_eq))
    for e in get_eqs(sys)
        s = string(e)
        @test !occursin("u(t, 0)", s) && !occursin("u(t,0)", s)
    end

    strict = MOLFiniteDifference(
        dxs, t; discretization_strategy = StrictArrayDiscretization()
    )
    sys_strict, _ = symbolic_discretize(pdesys, strict)
    @test narrayeqs_interior(sys_strict) == 1

    prob = discretize(pdesys, disc)
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
    # rules must key on that form. No scalar parity (boundaryvalfuncs skip interface
    # faces): prove nothing stays symbolic and the ODE is runnable, in both modes.
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

    for strategy in (ArrayDiscretization(), StrictArrayDiscretization())
        disc = MOLFiniteDifference(dxs, t; discretization_strategy = strategy)
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
    end

    disc = MOLFiniteDifference(dxs, t; discretization_strategy = ArrayDiscretization())
    prob = discretize(pdesys, disc)
    du = similar(prob.u0)
    prob.f(du, prob.u0, prob.p, 0.0)
    @test all(isfinite, du)
    sol = solve(prob, Rodas4())
    @test SciMLBase.successful_retcode(sol)
end

# Staggered grids: each variable's alignment fixes its interior stencil taps, so the
# interior collapses to one array equation per PDE. Staggered problems build
# SplitODEProblems without symbolic indexing, so solutions are compared positionally.
function staggered_wave(dx; periodic = false, strategy = ArrayDiscretization())
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
        edge_aligned_var = ϕ(t, x), discretization_strategy = strategy
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

    pdesys_s, disc_s = staggered_wave(0.125; strategy = MethodOfLines.PointwiseDiscretization())
    prob_arr = discretize(pdesys, disc)
    prob_scal = discretize(pdesys_s, disc_s)
    @test prob_arr.u0 == prob_scal.u0
    dt = (0.125 / 5.0)^2
    sol_arr = solve(prob_arr, SplitEuler(), dt = dt)
    sol_scal = solve(prob_scal, SplitEuler(), dt = dt)
    @test successful_retcode(sol_arr)
    @test successful_retcode(sol_scal)
    @test Array(sol_arr) ≈ Array(sol_scal) atol = 1.0e-12
end

@testset "Staggered 1D wave equation, periodic BCs" begin
    pdesys, disc = staggered_wave(0.125; periodic = true)
    sys, _ = symbolic_discretize(pdesys, disc)
    @test narrayeqs_interior(sys) == 2

    pdesys2, disc2 = staggered_wave(0.0625; periodic = true)
    sys2, _ = symbolic_discretize(pdesys2, disc2)
    @test length(get_eqs(sys2)) == length(get_eqs(sys))

    pdesys_s, disc_s = staggered_wave(
        0.125; periodic = true, strategy = MethodOfLines.PointwiseDiscretization()
    )
    prob_arr = discretize(pdesys, disc)
    prob_scal = discretize(pdesys_s, disc_s)
    @test prob_arr.u0 == prob_scal.u0
    dt = (0.125 / 5.0)^2
    sol_arr = solve(prob_arr, SplitEuler(), dt = dt)
    sol_scal = solve(prob_scal, SplitEuler(), dt = dt)
    @test successful_retcode(sol_arr)
    @test successful_retcode(sol_scal)
    @test Array(sol_arr) ≈ Array(sol_scal) atol = 1.0e-12
end

@testset "Staggered grid under StrictArrayDiscretization" begin
    # 1D staggered boundaries are single points — benign fallbacks — so strict mode
    # accepts the whole system.
    pdesys, disc = staggered_wave(0.125; strategy = StrictArrayDiscretization())
    sys, _ = symbolic_discretize(pdesys, disc)
    @test narrayeqs_interior(sys) == 2
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

    sol_arr, sol_scal, sys_arr = solve_both(pdesys, [x => 0.1, y => 0.1], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
    @test sol_arr[u(t, x, y)] ≈ sol_scal[u(t, x, y)] rtol = 1.0e-6

    # Fourth order widens both stencils of the tensor product at once
    sol_arr, sol_scal, sys_arr = solve_both(
        pdesys, [x => 0.1, y => 0.1], t; disc_kwargs = (; approx_order = 4)
    )
    @test narrayeqs_interior(sys_arr) == 1
    @test sol_arr[u(t, x, y)] ≈ sol_scal[u(t, x, y)] rtol = 1.0e-6
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

    sol_arr, sol_scal, sys_arr = solve_both(pdesys, [x => 0.1, y => 0.1], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
    @test sol_arr[u(t, x, y)] ≈ sol_scal[u(t, x, y)] rtol = 1.0e-6
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
    sol_arr, sol_scal, sys_arr = solve_both(pdesys, [x => xgrid, y => ygrid], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
    @test sol_arr[u(t, x, y)] ≈ sol_scal[u(t, x, y)] rtol = 1.0e-6
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

    sol_arr, sol_scal, sys_arr = solve_both(pdesys, [x => 0.1, y => 0.1], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    # one box per band of the periodic decomposition, a count independent of the grid
    @test narrayeqs_interior(sys_arr) == 2
    @test sol_arr[u(t, x, y)] ≈ sol_scal[u(t, x, y)] rtol = 1.0e-6
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

    sol_arr, sol_scal, sys_arr = solve_both(pdesys, [x => 0.2, y => 0.2, z => 0.2], t)
    @test sol_arr.retcode == SciMLBase.ReturnCode.Success
    @test narrayeqs_interior(sys_arr) == 1
    @test sol_arr[u(t, x, y, z)] ≈ sol_scal[u(t, x, y, z)] rtol = 1.0e-6
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
        sol_arr, sol_scal, sys_arr = solve_both(
            pdesys, [x => 0.05, y => 0.05], t; disc_kwargs = (; approx_order = order)
        )
        @test narrayeqs_interior(sys_arr) == 1
        @test sol_arr[u(t, x, y)] ≈ sol_scal[u(t, x, y)] rtol = 1.0e-6
    end
end

@testset "Fallback: mixed derivative of higher order in one direction" begin
    # `generate_mixed_rules` only has a scheme for `Dx(Dy(u))`; anything else reaches
    # `arrayify` with a spatial differential still in place, and must fall back rather
    # than error.
    @parameters t x y
    @variables u(..)
    Dt = Differential(t)
    Dxxy = (Differential(x)^2) * Differential(y)

    eq = Dt(u(t, x, y)) ~ Dxxy(u(t, x, y))
    bcs = [
        u(0, x, y) ~ sinpi(x) * sinpi(y),
        u(t, 0, y) ~ 0.0, u(t, 1, y) ~ 0.0,
        u(t, x, 0) ~ 0.0, u(t, x, 1) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.01), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0),
    ]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])

    lenient = MOLFiniteDifference(
        [x => 0.1, y => 0.1], t; discretization_strategy = ArrayDiscretization()
    )
    sys, _ = symbolic_discretize(pdesys, lenient)
    @test narrayeqs_interior(sys) == 0

    strict = MOLFiniteDifference(
        [x => 0.1, y => 0.1], t; discretization_strategy = StrictArrayDiscretization()
    )
    @test_throws MethodOfLines.ArrayDiscretizationError symbolic_discretize(pdesys, strict)
end
