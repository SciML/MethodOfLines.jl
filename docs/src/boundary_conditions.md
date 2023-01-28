# [Boundary Conditions] (@ref bcs)

What follows is a set of allowable boundary conditions, please note that this is not exhaustive - try your condition and see if it works, the handling is quite general. If it doesn't please post an issue and we'll try to support it. At the moment boundary conditions have to be supplied at the edge of the domain, but there are plans to support conditions embedded in the domain.

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
### Time dependant
```julia
u(t, 0., y) ~ x_min*y+ 0.5t
```
### Julia function
```julia
v(t, x, y_max) ~ sin(x)
```
### User defined function
```julia
alpha = 9

f(t,x,y) = x*y - t

function g(x,y) 
    z = sin(x*y)+cos(y)
    # Note that symbolic conditionals require the use of IfElse.ifelse, or registration
    return IfElse.ifelse(z > 0, x, 1.0)
end

u(t,x,y_min) ~ f(t,x,y_min) + alpha/g(x,y_min)
```
### Registered User Defined Function
```julia
alpha = 9

f(t,x,y) = x*y - t

function g(x,y) 
    z = sin(x*y)+cos(y)
    # This function must be registered as it contains a symbolic conditional
    if z > 0
        return x
    else
        return 1.0
    end
end

@register g(x, y)

u(t,x,y_min) ~ f(t,x,y_min) + alpha/g(x,y_min)
```
## Neumann/Robin
```julia
v(t, x_min, y) ~ 2. * Dx(v(t, x_min, y))
```
### Time dependant
```julia
u(t, x_min, y) ~ x_min*Dy(v(t,x_min,y)) + 0.5t
```
### Higher order
```julia
v(t, x, 1.0) ~ sin(x) + Dyy(v(t, x, y_max))
```
### Time derivative
```julia
Dt(u(t, x_min, y)) ~ 0.2
```
### User defined function
```julia
function f(u, v)
    (u + Dyy(v) - Dy(u))/(1 + v)
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
Note that if you want to use a higher order interface condition, this may not work if you have no simple condition of the form `c1(t, 0.5) ~ c2(t, 0.5)`.
