# FAQ

## Why is my result reducing over time, it shouldn't be!
Looks like you've been using the upwind scheme! This is the default scheme for odd ordered derivatives, and while it is very fast and uses low memory, it suffers from a property called numerical dispersion.
This causes sharp peaks and discontinuities to smooth out over time, and is the reason that we have the [WENO Scheme](@ref adschemes), which while resource intensive and sometimes problematic with exotic BCs, does not have this problem.

To see numerical dispersion in action, take a look at this example:
```julia
using ModelingToolkit, MethodOfLines, LinearAlgebra, OrdinaryDiffEq, DomainSets
using ModelingToolkit: Differential
using Plots
plotlyjs()

@parameters t x y
@variables u(..)
Dx = Differential(x)
Dy = Differential(y)
Dt = Differential(t)


t_min = 0.0
t_max = 7.0
x_min = -5.0
x_max = 5.0
y_min = -5.0
y_max = 5.0
dx = 0.25
dy = 0.25
order = 3

A = 1.0
x0 = 0.0
y0 = 0.0

sigma_x = 1.0
sigma_y = 1.0
theta = 0.0

# create function for bivariate gaussian

function bivariate_gaussian(x, y; A=1.0, x0=0.0, y0=0.0, sigma_x=1.0, sigma_y=1.0, theta=0.0)
    
    a = cos(theta)^2/(2*sigma_x^2) + sin(theta)^2/(2*sigma_y^2)
    b = -1*sin(2*theta)/(4*sigma_x^2) + sin(2*theta)/(4*sigma_y^2)
    c = sin(theta)^2/(2*sigma_x^2) + cos(theta)^2/(2*sigma_y^2)
    return A*exp(-(a*(x - x0)^2 + 2*b*(x - x0)*(y - y0) + c*(y - y0)^2 ))
end;


# equation system

eq = Dt(u(t, x, y)) - Dx(u(t, x, y)) - Dy(u(t, x, y)) ~ 0 

# boundary conditions

bcs = [
        u(t_min, x, y) ~ bivariate_gaussian(x, y; A=A, x0=x0, y0=y0, sigma_x=sigma_x, sigma_y=sigma_y, theta=theta),
        u(t, x_min, y) ~ u(t, x_max, y),    # <--- SHOULD BE A PERIODIC BOUNDARY CONDITION
        u(t, x, y_min) ~ u(t, x, y_max),    # <--- SHOULD BE A PERIODIC BOUNDARY CONDITION
]

domains = [
            t ∈ Interval(t_min, t_max),
            x ∈ Interval(x_min, x_max),
            y ∈ Interval(y_min, y_max)
]

@named pdesys = PDESystem([eq], bcs, domains, [t, x, y], [u(t, x, y)])

discretization = MOLFiniteDifference([x => dx, y => dy], t; advection_scheme=UpwindScheme())

prob = discretize(pdesys, discretization)

sol = solve(prob, SSPRK54(), dt=0.01, saveat=0.1)

discrete_t = sol[t]
solu = sol[u(t, x, y)]


anim = @animate for i=1:length(discrete_t)
    surface(solu[i,:,:], camera=(55.0, 30.0), size=(500,500), zlabel=("z"), 
            zlims=(0.0, 1.2), xlabel = ("x"), ylabel=("y"), clims=(0.0,1.0), 
            title="t = $i" )
end

gif(anim, "mol_convection_2d_test.gif", fps=5)
```
![convection test](https://github.com/SciML/MethodOfLines.jl/assets/9698054/45f4ace0-6291-478d-abb8-93d68ae3c9aa)

## Why is my large discretized system taking so long to compile?

At the moment, MOL effectively generates an assignment statement to a calculation in its generated code for all points in space, for all variables. Due to the configuration of LLVM for Julia, this leads the compiler to check all operations against all other operations to see what to [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data), leading to bad scaling properties with increasing point count.

There are changes in the works to roll this back in to `for` loops over each point, by maintaining some structural information, which will remedy this problem. Watch this space!

## Why are the corners of my domain held at 0?

The corner points do not have a valid discretization, and as such are held at 0.
