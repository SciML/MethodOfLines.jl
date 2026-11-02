# FAQ

## Why is my result reducing over time, it shouldn't be!

Looks like you've been using the upwind scheme! This is the default scheme for odd ordered derivatives, and while it is very fast and uses low memory, it suffers from a property called numerical dispersion.
This causes sharp peaks and discontinuities to smooth out over time, and is the reason that we have the [WENO Scheme](@ref adschemes), which while resource intensive and sometimes problematic with exotic BCs, does not have this problem.

To see numerical dispersion in action, take a look at this example:

```julia
using ModelingToolkit, MethodOfLines, LinearAlgebra, OrdinaryDiffEq, DomainSets
using ModelingToolkit: Differential
using Plots
plotlyjs()

@parameters t x y
@variables u(..)
Dx = Differential(x)
Dy = Differential(y)
Dt = Differential(t)

t_min = 0.0
t_max = 7.0
x_min = -5.0
x_max = 5.0
y_min = -5.0
y_max = 5.0
dx = 0.25
dy = 0.25
order = 3

A = 1.0
x0 = 0.0
y0 = 0.0

sigma_x = 1.0
sigma_y = 1.0
theta = 0.0

# create function for bivariate gaussian

function bivariate_gaussian(
        x, y; A = 1.0, x0 = 0.0, y0 = 0.0, sigma_x = 1.0, sigma_y = 1.0, theta = 0.0)
    a = cos(theta)^2 / (2 * sigma_x^2) + sin(theta)^2 / (2 * sigma_y^2)
    b = -1 * sin(2 * theta) / (4 * sigma_x^2) + sin(2 * theta) / (4 * sigma_y^2)
    c = sin(theta)^2 / (2 * sigma_x^2) + cos(theta)^2 / (2 * sigma_y^2)
    return A * exp(-(a * (x - x0)^2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0)^2))
end;

# equation system

eq = Dt(u(t, x, y)) - Dx(u(t, x, y)) - Dy(u(t, x, y)) ~ 0

# boundary conditions

bcs = [
    u(t_min, x, y) ~ bivariate_gaussian(
        x, y; A = A, x0 = x0, y0 = y0, sigma_x = sigma_x, sigma_y = sigma_y, theta = theta),
    u(t, x_min, y) ~ u(t, x_max, y),    # <--- SHOULD BE A PERIODIC BOUNDARY CONDITION
    u(t, x, y_min) ~ u(t, x, y_max)    # <--- SHOULD BE A PERIODIC BOUNDARY CONDITION
]

domains = [
    t ∈ Interval(t_min, t_max),
    x ∈ Interval(x_min, x_max),
    y ∈ Interval(y_min, y_max)
]

@named pdesys = PDESystem([eq], bcs, domains, [t, x, y], [u(t, x, y)])

discretization = MOLFiniteDifference(
    [x => dx, y => dy], t; advection_scheme = UpwindScheme())

sys, tspan = symbolic_discretize(pdesys, discretization)
prob = ODEProblem(mtkcompile(sys), nothing, tspan)

sol = solve(prob, SSPRK54(), dt = 0.01, saveat = 0.1)

discrete_t = sol[t]
solu = sol[u(t, x, y)]

anim = @animate for i in 1:length(discrete_t)
    surface(solu[i, :, :], camera = (55.0, 30.0), size = (500, 500), zlabel = ("z"),
        zlims = (0.0, 1.2), xlabel = ("x"), ylabel = ("y"), clims = (0.0, 1.0),
        title = "t = $i")
end

gif(anim, "mol_convection_2d_test.gif", fps = 5)
```

![convection test](https://github.com/SciML/MethodOfLines.jl/assets/9698054/45f4ace0-6291-478d-abb8-93d68ae3c9aa)

## Why is my large discretized system taking so long to compile?

At the moment, MOL effectively generates an assignment statement to a calculation in its generated code for all points in space, for all variables. Due to the configuration of LLVM for Julia, this leads the compiler to check all operations against all other operations to see what to [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data), leading to bad scaling properties with increasing point count.

There are changes in the works to roll this back in to `for` loops over each point, by maintaining some structural information, which will remedy this problem. Watch this space!

## Why are the corners of my domain held at 0?

The corner points do not have a valid discretization, and as such are held at 0.

## Why does a steady solve return `Stalled` although the answer is exact?

MethodOfLines does not set a tolerance for stationary problems, so `solve` uses NonlinearSolve's default `abstol` (and `reltol`) of about `3e-13`. The residual of a second-order stencil scales like `1/h^2`, so in floating point it can be limited by roundoff to a typical scale of `4 * eps / h^2` (`eps` being machine epsilon, `h` the grid spacing). This is an estimate, not a guaranteed floor: it depends on the size of the solution and on cancellation (for `u(x) = x` on a uniform grid the central second difference can be exactly zero). For solutions of order one, such as the steady 2D Laplace case `u = x*y`, the residual floor was measured at `6e-13` to `1e-12` on 31 points per side and at `3e-12` to `4.5e-12` on 61 points per side, so the default tolerance can be unreachable from about 31 points per side on.

Newton's method then reaches the exact discrete answer in one step but can never satisfy the tolerance. NonlinearSolve's step-stagnation rule (more than 32 steps smaller than `abstol`) ends the solve with `ReturnCode.Stalled` after about 33 steps, each of which costs a full Jacobian factorisation (about 118 ms per step on 61 points per side, making the solve roughly 30 times slower than needed). Pass your own `abstol` instead:

```julia
using ModelingToolkit, MethodOfLines, NonlinearSolve, DomainSets

@parameters x
@variables u(..)
Dxx = Differential(x)^2

pdesys = PDESystem(
    [Dxx(u(x)) ~ -pi^2 * sin(pi * x)], [u(0.0) ~ 0.0, u(1.0) ~ 0.0],
    [x ∈ Interval(0.0, 1.0)], [x], [u(x)]; name = :poisson)
prob = discretize(pdesys, MOLFiniteDifference([x => 0.01], nothing, approx_order = 2))

solve(prob, NewtonRaphson())                 # Stalled after 33 steps, 1.35 s (after compilation)
solve(prob, NewtonRaphson(); abstol = 1e-9)  # Success after 10 steps, 0.03 s, same solution
```

The timings above were measured on 101 grid points with ModelingToolkit 11.46.1, NonlinearSolve 4.32.0 and Julia 1.12; both solves return the same solution (maximum error against `sin(pi * x)` of `8.2251e-5`, the discretization error). Note that on ModelingToolkit versions without the fix for the stationary collapse (ModelingToolkit PR #5352), `discretize` also reduces the stencil system to a handful of unknowns (here 1), so step counts differ from what you get with the full set of unknowns, but the `Stalled` default and the remedy are the same.

Choose `abstol` well above the floor: for the 2D Laplace problem `1e-9` and `1e-10` succeeded in one step on both grids, `1e-11` took one or two, while `1e-12` and `3e-12` still stalled on 61 points per side. A rule of thumb is `abstol ≈ 1e3 * eps * h^-2 * norm(u, Inf)`, which gives `1e-9` to `1e-8` for these sizes. The size of a residual depends on how the equation is scaled and on conditioning (multiplying the equation by `1e-12` would let a badly wrong iterate pass `abstol = 1e-9`), so choose `abstol` relative to the size of the residual terms of your problem, for example by comparing with the norm of the terms of the equation at the initial guess, and check the solution and residual at the returned `u` rather than relying on the retcode alone. Loosening `reltol` alone has no effect, and switching the termination mode (`Rel`, `Abs` or `Norm`) at the default tolerances runs to `MaxIters` instead.
