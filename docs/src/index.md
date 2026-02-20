# [MethodOfLines.jl: Automated Finite Difference for Physics-Informed Learning] (@id index)

[MethodOfLines.jl](https://github.com/SciML/MethodOfLines.jl)
is a Julia package for automated finite difference discretization
of symbolically-defined PDEs in N dimensions.

It uses symbolic expressions for systems of partial differential equations as defined with `ModelingToolkit.jl`, and `Interval` from `DomainSets.jl` to define the space(time) over which the simulation runs.

It is a SciML “Discretizer” package, a class of packages which export the methods:

  - `discretize(sys::PDESystem, disc::D) where {D <: AbstractDiscretization}`, which returns an `AbstractSciMLProblem` to be solved with the ecosystem's solvers.
  - `symbolic_discretize(sys::PDESystem, disc::D) where {D <: AbstractDiscretization}`, which returns an `AbstractSystem` from `ModelingToolkit.jl`.

A Discretizer also optionally provides automatic solution wrapping, for easing the retrieval of shaped portions of the solution, and multidimensional interpolations. This feature is provided by `MethodOfLines.jl`, see the [solution interface](@ref sol) page for more information.

The `AbstractDiscretization` that `MethodOfLines.jl` provides is the [`MOLFiniteDifference`](@ref molfd), see its documentation for full information about interface options.

The package's handling is quite general, it is recommended to try out your system of equations and post an issue if you run in to trouble. If you want to solve it, we want to support it.

Issues with questions on usage are also welcome as they help us improve the docs.

See [here](@ref brusselator) for a full tutorial, involving the Brusselator equation.

Allowable terms in the system include, but are not limited to

  - Advection
  - Diffusion
  - Reaction
  - Nonlinear Diffusion
  - Spherical Laplacian
  - Any Julia function of the symbolic parameters/dependent variables and other parameters in the environment that's defined on the whole domain. Note that more complicated functions may require registration with `@register`, see the [ModelingToolkit.jl docs](https://docs.sciml.ai/ModelingToolkit/stable/basics/Validation/#User-Defined-Registered-Functions-and-Types).

Boundary conditions include, but are not limited to:

  - Dirichlet
  - Neumann (can also include time derivative)
  - Robin (can also include time derivative)
  - Periodic
  - Any Julia function that returns a number

Currently, the centered difference, upwind difference, nonlinear Laplacian and spherical Laplacian schemes are implemented. If you know of a scheme with better stability or accuracy in any specific case, please post an issue with a link to a paper.

Due to an implementation detail, the maximum derivative order that can be discretized by MOL is `div(typemax(Int), 2)`, in 64 bit `4611686018427387903`. We hope that this is enough for your purposes!

## Installation

Assuming that you already have Julia correctly installed, it suffices to import
MethodOfLines.jl in the standard way:

```julia
using Pkg
Pkg.add("MethodOfLines")
```

## Contributing

  - Please refer to the
    [SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://github.com/SciML/ColPrac/blob/master/README.md)
    for guidance on PRs, issues, and other matters relating to contributing to SciML.

  - See the [SciML Style Guide](https://github.com/SciML/SciMLStyle) for common coding practices and other style decisions.
  - There are a few community forums:
    
      + The #diffeq-bridged and #sciml-bridged channels in the
        [Julia Slack](https://julialang.org/slack/)
      + The #diffeq-bridged and #sciml-bridged channels in the
        [Julia Zulip](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
      + On the [Julia Discourse forums](https://discourse.julialang.org)
      + See also [SciML Community page](https://sciml.ai/community/)

## [Known Limitations] (@id limitations)

Currently, the package can discretize almost any system, with some assumptions listed below

  - That the grid is Cartesian.
  - Boundary conditions in time are supplied as initial conditions, not at the end of the simulation interval. If your system requires a final condition, please use a change of variables to rectify this. This is unlikely to change due to upstream constraints.
  - Integral equations are partially supported. See the [PIDE tutorial](@ref integral) for details.
  - That dependent variables always have the same argument signature, except in BCs.
  - That higher order interface bcs are accompanied by a simple interface of the form `u1(t, x_int) ~ u2(t, x_int)`
  - That boundary conditions do not contain references to derivatives which are not in the direction of the boundary, except in time.
  - That odd order derivatives do not multiply or divide each other, unless the WENO Scheme is used.
  - That the WENO scheme must be used when there are mixed derivatives.
  - Note that the WENO Scheme is often unstable in more than 1 spatial dimension due to difficulties with boundary handling, this can be avoided if you supply 2 or more bcs per boundary in the dimension along which an advection term acts.

## Reproducibility

```@raw html
<details><summary>The documentation of this SciML package was built using these direct dependencies,</summary>
```

```@example
using Pkg # hide
Pkg.status() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>and using this machine and Julia version.</summary>
```

```@example
using InteractiveUtils # hide
versioninfo() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>A more complete overview of all dependencies and their versions is also provided.</summary>
```

```@example
using Pkg # hide
Pkg.status(; mode = PKGMODE_MANIFEST) # hide
```

```@raw html
</details>
```

```@eval
using TOML
using Markdown
version = TOML.parse(read("../../Project.toml", String))["version"]
name = TOML.parse(read("../../Project.toml", String))["name"]
link_manifest = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
                "/assets/Manifest.toml"
link_project = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
               "/assets/Project.toml"
Markdown.parse("""You can also download the
[manifest]($link_manifest)
file and the
[project]($link_project)
file.
""")
```
