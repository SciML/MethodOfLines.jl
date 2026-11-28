# [Boundary Conditions](@id bcs)

What follows is a set of allowable boundary conditions, please note that this is not exhaustive - try your condition and see if it works, the handling is quite general. If it doesn't, please post an issue and we'll try to support it. Currently, boundary conditions have to be supplied at the edge of the domain, but there are plans to support conditions embedded in the domain.

## Definitions

```julia
using ModelingToolkit, MethodOfLines, DomainSets

@parameters x y t
@variables u(..) v(..)
Dt = Differential(t)
Dx = Differential(x)
Dy = Differential(y)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

x_min = y_min = 0.0

x_max = y_max = 1.0
```

## Dirichlet

```julia
v(t, 0, y) ~ 1.0
```

### Time dependent

```julia
u(t, 0.0, y) ~ x_min * y + 0.5t
```

### Julia function

```julia
v(t, x, y_max) ~ sin(x)
```

### User-defined function

```julia
alpha = 9

f(t, x, y) = x * y - t

function g(x, y)
    z = sin(x * y) + cos(y)
    # Note that symbolic conditionals require the use of Base.ifelse, or registration
    return ifelse(z > 0, x, 1.0)
end

u(t, x, y_min) ~ f(t, x, y_min) + alpha / g(x, y_min)
```

### Registered User Defined Function

```julia
alpha = 9

f(t, x, y) = x * y - t

function g(x, y)
    z = sin(x * y) + cos(y)
    # This function must be registered as it contains a symbolic conditional
    if z > 0
        return x
    else
        return 1.0
    end
end

@register_symbolic g(x, y)

u(t, x, y_min) ~ f(t, x, y_min) + alpha / g(x, y_min)
```

## Neumann/Robin

```julia
v(t, x_min, y) ~ 2.0 * Dx(v(t, x_min, y))
```

### Time dependent

```julia
u(t, x_min, y) ~ x_min * Dy(v(t, x_min, y)) + 0.5t
```

### Higher order

```julia
v(t, x, 1.0) ~ sin(x) + Dyy(v(t, x, y_max))
```

### Time derivative

```julia
Dt(u(t, x_min, y)) ~ 0.2
```

### User-defined function

```julia
function f(u, v)
    (u + Dyy(v) - Dy(u)) / (1 + v)
end

Dyy(u(t, x, y_min)) ~ f(u(t, x, y_min), v(t, x, y_min)) + 1
```

### 0 lhs

```julia
0 ~ u(t, x, y_max) - Dy(v(t, x, y_max))
```

## Periodic

```julia
u(t, x_min, y) ~ u(t, x_max, y)

v(t, x, y_max) ~ u(t, x_max, y)
```

Please note that if you want to use a periodic condition on a dimension with WENO schemes, please use a periodic condition on all variables in that dimension.

Periodic conditions are also supported with `WENOScheme()` on nonuniform grids; wrapped stencils use the exact periodically shifted grid coordinates.

## Interfaces

You may want to connect regions with differing dynamics together, to do this follow the following example, splitting the variable that spans these domains:

```julia
@parameters t x1 x2
@variables c1(..)
@variables c2(..)
Dt = Differential(t)

Dx1 = Differential(x1)
Dxx1 = Dx1^2

Dx2 = Differential(x2)
Dxx2 = Dx2^2

D1(c) = 1 + c / 10
D2(c) = 1 / 10 + c / 10

eqs = [Dt(c1(t, x1)) ~ Dx1(D1(c1(t, x1)) * Dx1(c1(t, x1))),
    Dt(c2(t, x2)) ~ Dx2(D2(c2(t, x2)) * Dx2(c2(t, x2)))]

bcs = [c1(0, x1) ~ 1 + cospi(2 * x1),
    c2(0, x2) ~ 1 + cospi(2 * x2),
    Dx1(c1(t, 0)) ~ 0,
    c1(t, 0.5) ~ c2(t, 0.5), # Relevant interface boundary condition
    -D1(c1(t, 0.5)) * Dx1(c1(t, 0.5)) ~ -D2(c2(t, 0.5)) * Dx2(c2(t, 0.5)), # Higher order interface condition
    Dx2(c2(t, 1)) ~ 0]

domains = [t ∈ Interval(0.0, 0.15),
    x1 ∈ Interval(0.0, 0.5),
    x2 ∈ Interval(0.5, 1.0)]

@named pdesys = PDESystem(eqs, bcs, domains,
    [t, x1, x2], [c1(t, x1), c2(t, x2)])
```

The rule for connected axes is stated in terms of the grids that are built. Equal specifications can still give grids whose gaps differ at the interface: equal integer point counts on intervals of different length give two uniform grids with different steps, and a scalar step that does not divide an interval gives a node vector with a shorter final cell next to the upper endpoint. (Unequal scalar steps are rejected, as before.) An integer point count gives that many nodes, both endpoints included: where the rounded step would end the range just below the upper endpoint, the axis is built from its endpoints and the count and stays uniform. Where the rounded step stops a whole cell short (for example 12 points on `[0, 0.1]`), the upper endpoint is appended as the last node, and the axis is a node vector whose final cell differs from the others by rounding. When the gaps next to the interface differ, or the connected variables are given different node vectors, `get_discrete` and `symbolic_discretize` accept first-order derivatives with `UpwindScheme()` (the default) in either wind direction, whose stencils use the node coordinates on both sides, and `WENOScheme()` when both resulting grids are node vectors, which then uses the coordinates of the neighbouring grid as well. They reject, with an `ArgumentError` naming the gaps and the offending derivative, derivatives of order above one (for example diffusion), mixed derivatives, `UpwindScheme` of order above one on any grid, `WENOScheme()` when either grid is uniform, and any derivative of the connected variables on a `StaggeredGrid`, whose stencils keep the weights of the grid they start on. On uniform grids only derivatives of the connected variables count, so another field on the same axis may have its own diffusion; along an axis whose grid is a node vector, derivatives of order above one are rejected in every field, whatever the gaps at the interface, because the centered stencils there do not support interfaces. That restriction is a stencil-construction check: when the connected grids and their gaps at the interface match, `get_discrete` still returns the grids, and `symbolic_discretize` raises the error. User-written `FunctionalScheme`s are not classified: they receive the tap coordinates and behave as without this check, so one that ignores them can be silently wrong across such an interface.

Note that if you want to use a higher order interface condition, this may not work if you have no simple condition of the form `c1(t, 0.5) ~ c2(t, 0.5)`.

Interfaces require the connected variables to be given the same step size or point count (the grids this produces may still differ at the interface, as described above), except when `WENOScheme()` is used with nonuniform grid vectors supplied for both connected variables, in which case the grids may differ: advection stencils that cross the interface are evaluated with the exact physical coordinates of the neighbouring grid. This exception covers first-order (advection) derivatives only — systems containing higher-order spatial derivatives (e.g. diffusion) in the connected variables are rejected with an error at discretization time.
