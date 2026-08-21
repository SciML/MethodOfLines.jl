# MethodOfLines.jl 1.0

## Breaking changes

With a `DAEProblem`, MethodOfLines v1.0 makes symbolic compilation O(1) with respect to
the number of grid points. Instead of generating and compiling one symbolic equation per
grid point, it generates operations over whole array slices. Compilation can therefore be
orders of magnitude faster than before, so `DAEProblem` is now the default for
time-dependent systems.

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

The O(1) compilation improvement does not apply to an `ODEProblem`: compiling one
scalarizes the array equations. Explicit Runge–Kutta methods such as `Tsit5()` and
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
