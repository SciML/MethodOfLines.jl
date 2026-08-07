# Tests for the ArrayDiscretization strategy (issue #428): the interior of each PDE is
# represented as a single symbolic array equation over slices of the array variables.
# Every case is checked for agreement against ScalarizedDiscretization, which the array
# strategy must reproduce exactly (it uses the same stencils, and falls back to the
# scalar path for unsupported patterns).

using MethodOfLines, ModelingToolkit, OrdinaryDiffEq, DomainSets, Symbolics
using SciMLBase
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
    @test narrayeqs_interior(sys_arr) == 0
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

@testset "Default strategy is the array form, with scalar available opt-in" begin
    # ArrayDiscretization is the default. ScalarizedDiscretization remains available and
    # is what the array path falls back to for patterns with no slice representation, so
    # it must stay selectable explicitly.
    @parameters t x
    disc_default = MOLFiniteDifference([x => 0.1], t)
    @test disc_default.disc_strategy isa ArrayDiscretization

    disc_steady = MOLFiniteDifference([x => 0.1])
    @test disc_steady.disc_strategy isa ArrayDiscretization

    disc_scalar = MOLFiniteDifference(
        [x => 0.1], t; discretization_strategy = ScalarizedDiscretization()
    )
    @test disc_scalar.disc_strategy isa ScalarizedDiscretization

    disc_strict = MOLFiniteDifference(
        [x => 0.1], t; discretization_strategy = StrictArrayDiscretization()
    )
    @test disc_strict.disc_strategy isa StrictArrayDiscretization
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
            "nonlinear laplacian",
            Dt(u(t, x)) ~ Dx(u(t, x) * Dx(u(t, x))),
            [u(0, x) ~ 1.0 + sinpi(x) / 2, u(t, 0) ~ 1.0, u(t, 1) ~ 1.0],
        ),
        (
            "boundary value in interior",
            Dt(u(t, x)) ~ Dxx(u(t, x)) + u(t, 1),
            [u(0, x) ~ sinpi(x), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0],
        ),
    ]
    for (name, eq, bcs) in unsupported
        @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
        @test_throws MethodOfLines.ArrayDiscretizationError symbolic_discretize(
            pdesys, strict
        )
        # the permissive strategy still handles it, pointwise
        lenient = MOLFiniteDifference(
            [x => 0.1], t; discretization_strategy = ArrayDiscretization()
        )
        sys, _ = symbolic_discretize(pdesys, lenient)
        @test narrayeqs_interior(sys) == 0
    end

    # A supported equation must go through strict mode unchanged.
    eq = Dt(u(t, x)) ~ Dxx(u(t, x))
    bcs = [u(0, x) ~ sinpi(x), u(t, 0) ~ 0.0, u(t, 1) ~ 0.0]
    @named ok_sys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
    sys_strict, _ = symbolic_discretize(ok_sys, strict)
    @test narrayeqs_interior(sys_strict) == 1

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
        Dt(u(t, x)) ~ Dx(u(t, x) * Dx(u(t, x))),
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
