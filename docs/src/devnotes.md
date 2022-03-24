# Notes for developers
## Getting started
First, fork the repo and clone it locally.

Then, type in the REPL
```
julia>] dev /path/to/your/repo
julia>] activate MethodOfLines
```

## Overview
MethodOfLines.jl makes heavy use of [`Symbolics.jl`](https://symbolics.juliasymbolics.org/dev/) and [`SymbolicUtils.jl`](https://symbolicutils.juliasymbolics.org), especially the replacement rules from the latter.

Take a look at [`src/discretization/MOL_discretization.jl`](https://github.com/SciML/MethodOfLines.jl/blob/master/src/discretization/MOL_discretization.jl) to get a high level overview of how the discretization works. A more consise description can be found [here](@ref hiw). Feel free to post an issue if you would like help understanding anything, or want to know developer opinions on the best way to go about implementing something.

## Adding new finite difference schemes

If you know of a finite difference scheme which is better than what is currently implemented, please first post an issue with a link to a paper.

A replacement rule is generated for each term which has a more specific higher stability/accuracy finite difference scheme than the general central difference, which represents a base case.

Take a look at [`src/discretization/generate_finite_difference_rules.jl](https://github.com/SciML/MethodOfLines.jl/blob/243252a595ed2af549d98270bd3b8ca5e3f93d69/src/discretization/generate_finite_difference_rules.jl#L252) to see how the replacement rules are generated. Note that the order that the rules are applied is important; there may be schemes that are applied first that are special cases of more general rules, for example the sphrical laplacian is a special case of the nonlinear lalacian.

First terms are split, isolating particular cases. Then, rules are generated and applied. Take a look at the docs for symbolic utils to get an idea of how these work. 

Identify a rule which will match your case, then write a function that will handle how to apply that scheme for each index in the interior, for each combination of independant and dependant variables. 

Initially, don't worry if your scheme is only implemented for specific approximation orders, it is sufficient just to warn when the requested approximation order does not match that supplied by the scheme. We can work in future pull requests to generalize the scheme to higher approximation orders, where possible.

## Inspecting generated code
To get the generated code for your system, use `code = ODEFunctionExpr(prob)`, or `MethodOfLines.generate_code(pdesys, discretization, "my_generated_code_filename.jl")`, which will create a file called `my_generated_code_filename.jl` in `pwd()`. This can be useful to find errors in the discretization, but note that it is not recommended to use this code directly, calling `solve(prob, AppropriateSolver())` will handle this for you.