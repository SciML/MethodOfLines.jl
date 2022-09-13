# Solution Retrieval - PDESolutions(@id sol)

MethodOfLines automatically wraps the `ODESolution` that results from its generated `ODEProblem` in 
a `PDESolution` object, reshaping it and providing a convenient interface for accessing your results,
as well as interpolations.

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
in the arguent signature of the variable. The time variable must appear either first or last in the 
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
The original `ODESolution` is stored in `sol.original_sol`.

To avoid wrapping entirely, use the `wrap` keyword argument to `solve`:
```julia
sol = solve(prob, Tsit5(); wrap = Val{false})
```
```
> typeof(sol)
ODESolution
```
This is useful where speed is important, but the shape of the solution is not.