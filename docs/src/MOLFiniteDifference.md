# [Discretization] (@id molfd)
```julia
struct MOLFiniteDifference{G} <: DiffEqBase.AbstractDiscretization
    dxs
    time
    approx_order::Int
    advection_scheme
    grid_align::G
    should_transform::Bool
    use_ODAE::Bool
    kwargs
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
                                      use_ODAE = <Whether to use ODAEProblem>)
prob = discretize(pdesys, discretization)
```
Where `dxs` is a vector of pairs of parameters to the grid step in this dimension, i.e. `[x=>0.2, y=>0.1]`. If the value given for a dimension is a subtype of `Integer`, the domain for that variable will be discretized in to that integer number of equally spaced points.

For a non uniform rectilinear grid, replace any or all of the step sizes with the grid you'd like to use with that variable, must be an `AbstractVector` but not a `StepRangeLen`.

Note that the second argument to `MOLFiniteDifference` is optional, all parameters can be discretized if all required boundary conditions are specified.

Currently implemented options for `advection_scheme` are `UpwindScheme()` and `WENOScheme()`, defaults to upwind. See [advection schemes](@ref adschemes) for more information.

Currently supported `grid_align`: `center_align` and `edge_align`. Edge align will give better accuracy with Neumann boundary conditions. Defaults tp `center_align`.

`center_align`: naive grid, starting from lower boundary, ending on upper boundary with step of `dx`

`edge_align`: offset grid, set halfway between the points that would be generated with center_align, with extra points at either end that are above and below the supremum and infimum by `dx/2`. This improves accuracy for Neumann BCs.

`should_transform`: Whether to automatically transform the system to make it compatible with MethodOfLines where possible, defaults to true. If your system has no mixed derivatives, all derivatives are purely of a dependent variable i.e. `Dx(u_aux(t,x))` not `Dx(v(t,x)*u(t,x))`, excepting nonlinear and spherical laplacians for which this holds for the innermost derivative argument, and no expandable derivatives, this can be set to false for better discretization performance at the cost of generality, if you perform these transformations yourself.

`use_ODAE`: MethodOfLines will automatically make use of `ODAEProblem` where relevant, which improves performance for DAEs (as discretized PDEs are in general), if this is set to true. Defaults to false.

Any unrecognized keyword arguments will be passed to the `ODEProblem` constructor, see [its documentation](https://docs.sciml.ai/ModelingToolkit/stable/systems/ODESystem/#Standard-Problem-Constructors) for available options.

