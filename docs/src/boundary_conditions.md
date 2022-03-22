# [Boundary Conditions] (@ref bcs)

What follows is a set of allowable boundary conditions, please note that this is not exhaustive - Try your condition and see if it works, the handling is quite general. If it doesn't please post an issue and we'll try to support it. At the moment boundary conditions have to be supplied at the edge of the domain, but there are plans to support conditions embedded in the domain.

## Definitions
```julia
using ModelingToolkit, MethodOfLines, Domainsets

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
    # Note that symbolic conditionals require the use of IfElse.ifelse
    return IfElse.ifelse(z > 0, x, 0.0)
end

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
