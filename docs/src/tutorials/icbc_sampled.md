# Initial and Boundary Conditions with sampled/measured Data

Initial and boundary conditions are sometimes applied with measured data that is itself pre-discretized. In order to use such data it is recommended to leverage [`Interpolations.jl`](https://github.com/JuliaMath/Interpolations.jl), or [`DataInterpolations.jl`](https://github.com/PumasAI/DataInterpolations.jl), for better dealing with possibly noisy data (currently limited to 1D). To create a callable effectively continuous function, for example (from the `Interpolations.jl` [docs](http://juliamath.github.io/Interpolations.jl/latest/control/)):

1D:
```julia
using Interpolations

A_x = 1.:2.:40.
A = [log10(x) for x in A_x]
itp = interpolate(A, BSpline(Cubic(Line(OnGrid()))))
sitp1 = scale(itp, A_x)
sitp1(3.) # exactly log10(3.)
sitp1(3.5) # approximately log10(3.5)
```

Multidimensional:
```julia
using Interpolations 

A_x1 = 1:.1:10
A_x2 = 1:.5:20
f(x1, x2) = log10(x1+x2)
A = [f(x1,x2) for x1 in A_x1, x2 in A_x2]
itp = interpolate(A, BSpline(Cubic(Line(OnGrid()))))
sitp2 = scale(itp, A_x1, A_x2)
sitp2(5., 10.) # exactly log10(5 + 10)
sitp2(5.6, 7.1) # approximately log10(5.6 + 7.1)
```
Then, register the functions with ModelingToolkit:
```julia
sitp1_f(x) = sitp1(x)
sitp2_f(x, y) = sitp2(x, y)
@register_symbolic sitp1_f(y)
@register_symbolic sitp2_f(x, y)
```

Then as a BC or IC:
```julia
bcs = [u(0, x, y) ~ sitp2_f(x, y),
       u(t, 0, y) ~ sitp1_f(y),
       ...
       ]
```

Note that the measured data need not be measured on the same grid as will be generated for the discretization in `MethodOfLines.jl`, as long as it is defined upon the whole simulation domain it will be automatically re-sampled.

If you are using an [`edge_align` grid](@ref molfd), your interpolation will need to be defined `±dx/2 ` above and below the edges of the simulation domain where `dx` is the step size in the direction of that edge. [Extrapolation](http://juliamath.github.io/Interpolations.jl/latest/extrapolation/) may prove useful here.

