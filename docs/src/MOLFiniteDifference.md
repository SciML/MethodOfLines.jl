# [Discretization] (@id molfd)
```julia
struct MOLFiniteDifference{G} <: DiffEqBase.AbstractDiscretization
    dxs
    time
    approx_order::Int
    upwind_order::Int
    grid_align::G
end

# Constructors. If no order is specified, both upwind and centered differences will be 2nd order
function MOLFiniteDifference(dxs, time=nothing; approx_order = 2, upwind_order = 1, grid_align=CenterAlignedGrid())
    
    if approx_order % 2 != 0
        @warn "Discretization approx_order must be even, rounding up to $(approx_order+1)"
    end
    @assert approx_order >= 1 "approx_order must be at least 1"
    @assert upwind_order >= 1 "upwind_order must be at least 1"
    
    return MOLFiniteDifference{typeof(grid_align)}(dxs, time, approx_order, upwind_order, grid_align)
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
Where `dxs` is a vector of pairs of parameters to the grid step in this dimension, i.e. `[x=>0.2, y=>0.1]`.
For a non uniform rectilinear grid, replace any or all of the step sizes with the grid you'd like to use with that variable, must be an `AbstractVector` but not a `StepRangeLen`.

Note that the second argument to `MOLFiniteDifference` is optional, all parameters can be discretized if all required boundary conditions are specified.

Currently implemented options for `advection_scheme` are `UpwindScheme()` and `WENOScheme()`, defaults to upwind. See [advection schemes](@ref adschemes) for more information.

Currently supported `grid_align`: `center_align` and `edge_align`. Edge align will give better accuracy with Neumann boundary conditions. Defaults tp `center_align`.

`center_align`: naive grid, starting from lower boundary, ending on upper boundary with step of `dx`

`edge_align`: offset grid, set halfway between the points that would be generated with center_align, with extra points at either end that are above and below the supremum and infimum by `dx/2`. This improves accuracy for Neumann BCs.

`should_transform`: Whether to automatically transform the system to make it compatible with MethodOfLines where possible, defaults to true. If your system has no mixed derivatives, all derivatives are purely of a dependent variable i.e. Dx(u(t,x)) not Dx(v(t,x)*u(t,x)) excepting nonlinear and spherical laplacians for which this holds for the innermost derivative argument, and no expandable derivatives, this can be set to false for better discretization performance.

`use_ODAE`: MethodOfLines automatically makes use of `ODAEProblem` where relevant, which improves performance for DAEs (as discretized PDEs are in general). To force MethodOfines to use `ODEProblem` please set this to false.

Any unrecognized keyword arguments will be passed to the `ODEProblem` or `ODAEProblem` constructor, see [its documentation](https://mtk.sciml.ai/stable/systems/ODESystem/#Standard-Problem-Constructors) for available options.