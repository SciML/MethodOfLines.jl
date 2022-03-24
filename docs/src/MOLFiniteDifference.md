# [Discretization] (@ref molfd)
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
                                      upwind_order = <Currently unstable at any value other than 1>, 
                                      approx_order = <Order of derivative approximation, starting from 2> 
                                      grid_align = <your grid type choice>)
prob = discretize(pdesys, discretization)
```
Where `dxs` is a vector of pairs of parameters to the grid step in this dimension, i.e. `[x=>0.2, y=>0.1]`

Note that the second argument to `MOLFiniteDifference` is optional, all parameters can be discretized if all required boundary conditions are specified.

Currently supported grid types: `center_align` and `edge_align`. Edge align will give better accuracy with Neumann Boundary conditions.

`center_align`: naive grid, starting from lower boundary, ending on upper boundary with step of `dx`

`edge_align`: offset grid, set halfway between the points that would be generated with center_align, with extra points at either end that are above and below the supremum and infimum by `dx/2`. This improves accuracy for neumann BCs.
