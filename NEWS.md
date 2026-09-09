# MethodOfLines.jl 1.5

`ODEProblem(pdesys, discretization)` discretizes and compiles a time-dependent system into
an `ODEProblem` for explicit time-stepping methods such as `Tsit5()`. With ModelingToolkit
v11.43 / ModelingToolkitBase v1.70 or later, the compilation keeps the array (slice-form)
equations (`mtkcompile(sys; scalarize_arrays = false)`), so the compiled `ODEProblem` path
is now O(1) in the number of grid points like the `DAEProblem` path. `ode_compile(sys)`
exposes that compilation step for systems obtained from `symbolic_discretize`; it falls
back to the scalarizing `mtkcompile` when the array-preserving compilation does not yield
an explicit ODE (for instance for systems second order in time), or with an older
ModelingToolkit.

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
