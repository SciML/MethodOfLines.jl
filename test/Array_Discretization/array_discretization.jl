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
using OrdinaryDiffEqSSPRK: SSPRK22
using OrdinaryDiffEqLowOrderRK: SplitEuler
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

    sols = map([ArrayDiscretization(), ScalarizedDiscretization()]) do st
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
    @test narrayeqs_interior(sys_arr) == 0
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
        [x => 0.1], t; discretization_strategy = ScalarizedDiscretization()
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

@testset "Default strategy is unchanged (opt-in)" begin
    # This strategy is opt-in: existing code must keep getting the scalarized
    # discretization, so that adding it cannot change any current user's results.
    @parameters t x
    disc_default = MOLFiniteDifference([x => 0.1], t)
    @test disc_default.disc_strategy isa ScalarizedDiscretization

    disc_steady = MOLFiniteDifference([x => 0.1])
    @test disc_steady.disc_strategy isa ScalarizedDiscretization

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
            discretization_strategy = ScalarizedDiscretization()
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
            "spherical laplacian",
            Dt(u(t, x)) ~ Dx(x^2 * Dx(u(t, x))) / x^2,
            [u(0, x) ~ 1.0 + sinpi(x) / 2, u(t, 0) ~ 1.0, u(t, 1) ~ 1.0],
        ),
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

    # The error names the offending equation and the reason.
    @named bad = PDESystem(
        Dt(u(t, x)) ~ Dx(x^2 * Dx(u(t, x))) / x^2,
        [u(0, x) ~ 1.0 + sinpi(x) / 2, u(t, 0) ~ 1.0, u(t, 1) ~ 1.0],
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
        [ArrayDiscretization(), ScalarizedDiscretization()]
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
        dxs, t; discretization_strategy = ScalarizedDiscretization()
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
        dxs2, t; discretization_strategy = ScalarizedDiscretization()
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

    pdesys_s, disc_s = staggered_wave(0.125; strategy = ScalarizedDiscretization())
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
        0.125; periodic = true, strategy = ScalarizedDiscretization()
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
