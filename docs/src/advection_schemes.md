# [Advection Schemes](@id adschemes)

Used as a keyword argument `advection_scheme` to `MOLFiniteDifference`.

## Upwind Scheme

```julia
UpwindScheme(approx_order = 1)
```

Changes the direction of the stencil based on the sign of the coefficient of the odd order derivative to be discretized. Scheme order can be increased by changing the `approx_order` argument. For more information, see [Wikipedia](https://en.wikipedia.org/wiki/Upwind_scheme).

## WENO Scheme of Jiang and Shu

```julia
WENOScheme(epsilon = 1e-6)
```

A more stable scheme, 5th order accurate, which is a weighted sum of several different schemes, weighted based on the curvature of the solution at the point in question. More stable and tolerant of discontinuities, at the cost of solve complexity.

`epsilon`is a quantity used to prevent vanishing denominators in the scheme, defaults to `1e-6`. Problems with a lower magnitude solution will benefit from a smaller value.

Problems which require this scheme may also benefit from a [Strong-Stability-Preserving (SSP) solver](https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/#Explicit-Strong-Stability-Preserving-Runge-Kutta-Methods-for-Hyperbolic-PDEs-(Conservation-Laws)).

Problems with first order derivatives which multiply one another will need to use this scheme over the upwind scheme.

Supports only first order derivatives, other odd order derivatives are unsupported with this scheme.

Nonuniform grids are supported: when a grid vector is supplied for a spatial variable, a node-centered nonuniform WENO-5 reconstruction (4th order accurate in smooth regions) is used, with one-sided reconstructions at physical boundaries.

Periodic and interface boundary conditions are supported on nonuniform grids. Stencils that wrap across a periodic boundary or an interface between two domains are evaluated with the exact physical coordinates of the connected grid, so no accuracy is lost at the seam. The grids on either side of an interface do not need to match when a nonuniform grid vector is supplied for both connected variables. This applies to first-order (advection) derivatives only: systems with spatial derivative orders above 1 in variables connected by a mismatched-grid interface are rejected at discretization time, since higher-order stencils are not coordinate-aware at the interface.

Specified on pages 8-9 of [this document](https://repository.library.brown.edu/studio/item/bdr:297524/PDF/)

## FunctionalScheme
