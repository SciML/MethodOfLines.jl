# MethodOfLines.jl

MethodOfLines.jl is a package for automated finite difference discretization
of symbolicaly-defined PDEs in N dimensions.

It uses symbolic expressions for systems of partial differential equations as defined with `ModelingToolkit.jl`, and `Interval` from `DomainSets.jl` to define the space(time) over which the simulation runs.

This project is under active development, therefore the interface is subject to change. The [docs](http://methodoflines.sciml.ai/dev/) will be updated to reflect any changes, please check back for current usage information.

Allowable terms in the system include, but are not limited to
- Advection
- Diffusion
- Reaction
- Nonlinear Diffusion
- Spherical laplacian
- Any julia function of the symbolic parameters/dependant variables and other parameters in the environment that's defined on the whole domain.

Boundary conditions include, but are not limited to:
- Dirichlet
- Neumann (can also include time derivative)
- Robin (can also include time derivative)
- Periodic
- Any equation, which can include arbitrary julia functions defined on that boundary, with the only symbolic parameters being those appering in the referenced boundary.
