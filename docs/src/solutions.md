# [Solution Retrieval - PDESolutions](@id sol)

MethodOfLines wraps the solution of its generated `DAEProblem`, `ODEProblem`, or
`NonlinearProblem` in a PDE solution object, reshaping it and providing a convenient
interface for accessing results and interpolations. Time-dependent problems use
`PDETimeSeriesSolution`; stationary problems use `PDENoTimeSolution`.

## Solution Retrieval

For example, for a `PDESystem` such as the following:

```julia
@named pdesys = PDESystem(eqs, bcs, domains, [t, x, y], [u(t, x, y), v((t, x, y))])
```

after the solve:

```julia
sol = solve(prob)
```

You can access the solutions for `u` and `v` like this:

```julia
solu = sol[u(t, x, y)]
solv = sol[v(t, x, y)]
```

Note that the result in this case will be a 3D Array, dimensions matching the order that they appear
in the argument signature of the variable. The time variable must appear either first or last in the
arguments, or an error will be thrown.

## Grid Retrieval

To access the discretized axes for the independent variables, simply index the solution with the independent variable itself:

```julia
disc_t = sol[t]
disc_x = sol[x]
disc_y = sol[y]
```

## Interpolations

To access an interpolation of the solution, call the `sol` object:

```julia
#Interpolated solution of `u` at t = 0.4, x = 1.7, y = 2.6
u_interp = sol(0.4, 1.7, 2.6, dv = u(t, x, y))
```

To retrieve an interpolation in all dependent variables as a vector, leave off the `dv` argument.
Be sure to supply a value for every independent variable in the order that they appear in `sol.ivs`. The vector is in the order of `sol.dvs`.

```julia
uv_interp = sol(0.4, 1.7, 2.6)
```

## Original solution

The original solver solution is stored in `sol.original_sol`.

To avoid wrapping entirely, use the `wrap` keyword argument to `solve`:

```julia

sol = solve(prob; wrap = Val(false))
```

```
> typeof(sol)
DAESolution
```

This is useful where speed is important, but the shape of the solution is not.

## Differentiation

`sol[u(t, x)]`, `sol[u(t, x), inds...]` and `sol(t, x; dv)` differentiate in reverse mode through `solve`, with SciMLSensitivity loaded, and so do their stationary forms, `sol[u(x)]` and `sol(x; dv)`. The symbol is built outside the differentiated function, since symbolic expressions cannot be built inside it. With Zygote, for a system with parameters `α` and `β`, a target array `data` and a starting vector `p0`:

```julia
using SciMLSensitivity, Zygote
using SymbolicIndexingInterface: setp_oop

set_p = setp_oop(prob, [α, β])
U = u(t, x)
loss(p) = sum(abs2, solve(remake(prob; p = set_p(prob, p)); saveat = 0.1)[U] .- data)
Zygote.gradient(loss, p0)
```

`setp_oop` builds the parameter object outside `remake`; a symbolic map passed to `remake` has no rule. With Mooncake, make the symbol a constant, `const U = u(t, x)`, and set the parameters through SciMLStructures, `tunables, repack, _ = canonicalize(Tunable(), prob.p)` and `remake(prob; p = repack(tunables))`, since a parameter object from `setp_oop` does not get through ModelingToolkit's rule for it yet. `sol.u` and complex-valued variables have no rule, and a gradient through them is an error. On the `DAEProblem` path of `discretize` the initial condition is evaluated when the problem is built, so a parameter in it is not carried by `remake`.
