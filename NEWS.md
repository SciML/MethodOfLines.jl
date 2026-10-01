# MethodOfLines.jl 1.0

## Breaking changes

With a `DAEProblem`, MethodOfLines v1.0 keeps a fixed number of symbolic array equations
independent of the grid resolution. Instead of one symbolic equation per grid point, it
generates operations over whole array slices at the `System` level. Symbolic processing
of that system can therefore be orders of magnitude faster than before, so `DAEProblem`
is now the default for time-dependent systems. Residual code generation still expands
one entry per unknown, so `discretize` time and the first-call compile of `prob.f` still
grow with resolution (see MethodOfLines.jl#691; ModelingToolkit.jl#5139).

The system returned by `symbolic_discretize` now contains these symbolic array equations.
Code that directly inspects or transforms that system must handle them.

`discretize` now normally returns a `DAEProblem` without first calling `mtkcompile`.
Existing code that passes an ODE solver directly to the result, such as

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

The `ODEProblem` path additionally scalarizes the array equations at the `System` level
via `mtkcompile` (both paths still pay per-element residual codegen until
ModelingToolkit.jl#5139). Explicit Runge–Kutta methods such as `Tsit5()` and
`SSPRK54()` require an `ODEProblem`, so construct one explicitly:

```julia
sys, tspan = symbolic_discretize(pdesys, discretization)
prob = ODEProblem(mtkcompile(sys), nothing, tspan)
sol = solve(prob, Tsit5())
```

Systems that cannot be represented by the first-order DAE path automatically fall back to
a compiled `ODEProblem`. Time-independent systems continue to produce a
`NonlinearProblem`. Solutions remain wrapped as `PDETimeSeriesSolution`, and PDE-variable
indexing and interpolation are unchanged.
