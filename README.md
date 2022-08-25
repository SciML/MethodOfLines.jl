# MethodOfLines.jl

[![Github Action CI](https://github.com/SciML/MethodOfLines.jl/workflows/CI/badge.svg)](https://github.com/SciML/MethodOfLines.jl/actions)
[![Coverage Status](https://coveralls.io/repos/github/SciML/MethodOfLines.jl/badge.svg?branch=master)](https://coveralls.io/github/SciML/MethodOfLines.jl?branch=master)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](http://methodoflines.sciml.ai/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](http://methodoflines.sciml.ai/dev/)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

MethodOfLines.jl is a package for automated finite difference discretization
of symbolically-defined PDEs in N dimensions.

It uses symbolic expressions for systems of partial differential equations as defined with `ModelingToolkit.jl`, and `Interval` from `DomainSets.jl` to define the space(time) over which the simulation runs.

This project is under active development, therefore the interface is subject to change. The [docs](http://methodoflines.sciml.ai/dev/) will be updated to reflect any changes, please check back for current usage information.

Note that this package does not currently scale well to high resolution (high point count) problems, though there are changes in the works to remedy this.

Allowable terms in the system include, but are not limited to
- Advection
- Diffusion
- Reaction
- Nonlinear Diffusion
- Spherical Laplacian
- Any Julia function of the symbolic parameters/dependant variables and other parameters in the environment that's defined on the whole domain.

Boundary conditions include, but are not limited to:
- Dirichlet
- Neumann (can also include time derivative)
- Robin (can also include time derivative)
- Periodic
- Any equation, which can include arbitrary Julia functions defined on that boundary, with the only symbolic parameters being those appearing in the referenced boundary.
