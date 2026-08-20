# [Discretization](@id molfd)
```julia
struct MOLFiniteDifference{G, D} <: AbstractEquationSystemDiscretization
    dxs::Any
    time::Any
    approx_order::Int
    advection_scheme::Any
    grid_align::G
    should_transform::Bool
    use_ODAE::Bool
    disc_strategy::D
    useIR::Bool
    callbacks::Any
    kwargs::Any
end
```

```julia
eq = [your system of equations, see examples for possibilities]
bcs = [your boundary conditions, see examples for possibilities]

domain = [your domain, a vector of Intervals i.e. x ∈ Interval(x_min, x_max)]

@named pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])

discretization = MOLFiniteDifference(dxs, 
                                      <your choice of continuous variable, usually time>; 
                                      advection_scheme = <UpwindScheme() or WENOScheme()>, 
                                      approx_order = <Order of derivative approximation, starting from 2> 
                                      grid_align = <your grid type choice>,
                                      should_transform = <Whether to automatically transform the PDESystem (see below)>
                                      use_ODAE = <Whether to retain the compiled ODEProblem path>)
prob = discretize(pdesys, discretization)
```
Where `dxs` is a vector of pairs of parameters to the grid step in this dimension, i.e. `[x=>0.2, y=>0.1]`. If the value given for a dimension is a subtype of `Integer`, the domain for that variable will be discretized in to that integer number of equally spaced points.

For a non-uniform rectilinear grid, replace any or all of the step sizes with the grid you'd like to use with that variable, must be an `AbstractVector` but not a `StepRangeLen`. See [Non-Uniform Rectilinear Grids](nonuniform.md) and the [WENO tutorial](@ref weno_tutorial) for worked examples.

Note that the second argument to `MOLFiniteDifference` is optional, all parameters can be discretized if all required boundary conditions are specified.

Currently, implemented options for `advection_scheme` are `UpwindScheme()` and `WENOScheme()`, defaults to upwind. See [advection schemes](@ref adschemes) for more information, and the [WENO tutorial](@ref weno_tutorial) for a worked comparison of the two.

Currently supported options are `grid_align`: `center_align` and `edge_align`. Edge align will give better accuracy with Neumann boundary conditions. Defaults to `center_align`.

`center_align`: naive grid, starting from lower boundary, ending on upper boundary with step of `dx`

`edge_align`: offset grid, set halfway between the points that would be generated with center_align, with extra points at either end that are above and below the supremum and infimum by `dx/2`. This improves accuracy for Neumann BCs.

`should_transform`: Whether to automatically transform the system to make it compatible with MethodOfLines where possible, defaults to true. If your system has no mixed derivatives, all derivatives are purely of a dependent variable i.e. `Dx(u_aux(t,x))` not `Dx(v(t,x)*u(t,x))`, excepting nonlinear and spherical Laplacians for which this holds for the innermost derivative argument, and no expandable derivatives, this can be set to false for better discretization performance at the cost of generality, if you perform these transformations yourself.

`use_ODAE`: Retains the pre-v1 compiled `ODEProblem` path when set to `true`. Defaults to `false`, which returns a `DAEProblem` for supported time-dependent systems.

`discretization_strategy`: How the discretized equations are represented symbolically. `ArrayDiscretization()`, the default and only public strategy in v1, generates the interior of each PDE as a single symbolic array equation over slices of the discretized variables, e.g. `D(u[2:n-1]) - (u[1:n-2] .- 2u[2:n-1] .+ u[3:n]) ./ dx^2 ~ 0`. This keeps the number of symbolic equations independent of the grid resolution and scales much better during symbolic processing. Nonlinear Laplacians `Dx(a(u) * Dx(u))`, spherical Laplacians `r^-2 Dr(r^2 Dr(u))`, mixed first-order derivatives `Dx(Dy(u))`, WENO / functional advection (on uniform grids, and on nonuniform grids — periodic or not — for schemes that provide a coefficient split; WENO does; user schemes opt in by defining a method on `MethodOfLines.array_scheme_split`), staggered grids, self-periodic interfaces and boundary values appearing inside an interior equation (e.g. `u(t, 1)`) are included in the array form, including on wrap boxes near a periodic seam. Patterns without a slice representation (nonuniform advection schemes without a coefficient split, schemes that read the grid coordinate, integrals, higher mixed derivatives, two-domain interface BCs, derivatives of boundary values, time-literal references such as `u(0, x)`, boundary values on edge-aligned grids, stationary systems) automatically fall back to pointwise scalar equations for the affected equation. `StrictArrayDiscretization()` turns that fallback into an error. `ScalarizedDiscretization()` was removed in v1 — the scalar form is now the fallback inside the array strategy rather than a strategy of its own. Where the array form is used, numerics match the pointwise path whenever that path can express the same boundary-value substitutions; the array path also substitutes periodic-face and free-standing-corner references that pointwise `boundaryvalfuncs` currently leave symbolic.

Any unrecognized keyword arguments are passed to the generated problem constructor; see the [ModelingToolkit problem documentation](https://docs.sciml.ai/ModelingToolkit/stable/API/problems/#Dynamical-systems) for available options.

## Problem types

`discretize` returns a `DAEProblem` for a time-dependent system:

```julia
using OrdinaryDiffEqBDF: DFBDF

disc = MOLFiniteDifference([x => n], t)
prob = discretize(pdesys, disc)
sol = solve(prob, DFBDF())
```

MethodOfLines emits residuals of the form `D(u) - f ~ 0`, which are already in
implicit-DAE form. Building a `DAEProblem` therefore needs no `mtkcompile`, and the array
equations reach the generated code intact — isolating the derivative for an `ODEProblem`
is structural simplification, and it scalarizes them. Solve with an implicit-DAE
algorithm such as `DFBDF()`.

`initializealg` defaults to `BrownFullBasicInit()`, chosen only when the discretized
system's initialization equations are ones that algorithm preserves.

A few systems cannot be posed as a first-order DAE: those second order in time, and those
whose initialization equations `BrownFullBasicInit` would not honour. They fall back to
`mtkcompile` plus an `ODEProblem`, which scalarizes the array equations. Pass
`fallback = false` to `discretize` to make that an error instead.

Time-independent systems have no derivative to keep implicit, and discretize to a
`NonlinearProblem`.

The solution is a `PDETimeSeriesSolution` in every case, indexed and interpolated by the
`PDESystem`'s own variables: `sol[u(t, x)]`, `sol(t, x)`.

## Building a problem yourself: `symbolic_discretize`

To construct a problem type `discretize` does not build — an `ODEProblem`, or anything
else — start from `symbolic_discretize`, which returns the discretized system and the
time span:

```julia
sys, tspan = symbolic_discretize(pdesys, disc)

# an ODEProblem needs `D(x) = f(x)`, so compile first
prob = ODEProblem(mtkcompile(sys), nothing, tspan)
sol = solve(prob, Rodas5P())
```

Note that `mtkcompile` scalarizes the array equations, so this path gives up the scaling
benefit of the array form. Prefer `discretize` unless you specifically need an
`ODEProblem`.

## Migrating to v1

- `discretize` returns a `DAEProblem` rather than an `ODEProblem` for time-dependent
  systems. Solvers must be implicit-DAE algorithms such as `DFBDF()`; an `ODEProblem`
  solver like `Tsit5()` will no longer accept the result. Solution indexing is unchanged.
- `ScalarizedDiscretization()` was removed. It is the default fallback inside
  `ArrayDiscretization()`; the public strategy choices are `ArrayDiscretization()` and
  `StrictArrayDiscretization()`.
- To retain the pre-v1 compiled `ODEProblem` behavior, construct the discretization with
  `use_ODAE = true`. For manual problem construction, use `symbolic_discretize` plus
  `mtkcompile` as above.
