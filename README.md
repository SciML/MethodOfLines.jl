# MethodOfLines.jl

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/MethodOfLines/stable/)

[![codecov](https://codecov.io/gh/SciML/MethodOfLines.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/SciML/MethodOfLines.jl)
[![Build Status](https://github.com/SciML/MethodOfLines.jl/workflows/CI/badge.svg)](https://github.com/SciML/MethodOfLines.jl/actions?query=workflow%3ACI)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)


MethodOfLines.jl is a package for automated finite difference discretization
of symbolically-defined PDEs in N dimensions.

It uses symbolic expressions for systems of partial differential equations as defined with `ModelingToolkit.jl`, and `Interval` from `DomainSets.jl` to define the space(time) over which the simulation runs.

This project is under active development, therefore the interface is subject to change. The [docs](https://docs.sciml.ai/MethodOfLines/dev/) will be updated to reflect any changes, please check back for current usage information.

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
