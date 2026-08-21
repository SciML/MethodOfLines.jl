# MethodOfLines.jl 1.0

## Breaking changes

MethodOfLines now preserves the array structure of a spatial discretization. Interior
equations are represented as symbolic operations over array slices instead of one scalar
equation per grid point. This substantially reduces symbolic compilation work for large
grids. Code that directly inspects or transforms the system returned by
`symbolic_discretize` must now handle symbolic array equations. Call `mtkcompile` when a
compiled, scalarized system is required.

For time-dependent systems, `discretize` now normally returns a `DAEProblem` without first
calling `mtkcompile`. Existing code that passes an ODE solver directly to the result, such
as

```julia
prob = discretize(pdesys, discretization)
sol = solve(prob, Tsit5())
```

must choose one of the following paths.

Use the array-form `DAEProblem` and let OrdinaryDiffEq select its default DAE solver:

```julia
prob = discretize(pdesys, discretization)
sol = solve(prob)
```

If an explicit Runge–Kutta method such as `Tsit5()` or `SSPRK54()` is required, explicitly
construct the compiled `ODEProblem`:

```julia
sys, tspan = symbolic_discretize(pdesys, discretization)
prob = ODEProblem(mtkcompile(sys), nothing, tspan)
sol = solve(prob, Tsit5())
```

Compiling the `ODEProblem` scalarizes the array equations, so this path does not retain the
symbolic scaling benefit of the default DAE path.

Systems that cannot be represented by the first-order DAE path automatically fall back to
a compiled `ODEProblem`. Time-independent systems continue to produce a
`NonlinearProblem`. Solutions remain wrapped as `PDETimeSeriesSolution`, and PDE-variable
indexing and interpolation are unchanged.
