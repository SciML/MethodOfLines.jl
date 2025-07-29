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

Take a look at [`src/discretization/MOL_discretization.jl`](https://github.com/SciML/MethodOfLines.jl/blob/master/src/MOL_discretization.jl) to get a high level overview of how the discretization works. A more concise description can be found [here](@ref hiw). Feel free to post an issue if you would like help understanding anything, or want to know developer opinions on the best way to go about implementing something.

## Adding new finite difference schemes

If you know of a finite difference scheme which is better than what is currently implemented, please first post an issue with a link to a paper.

A replacement rule is generated for each term which has a more specific higher stability/accuracy finite difference scheme than the general central difference, which represents a base case.

Take a look at [`src/discretization/generate_finite_difference_rules.jl`](https://github.com/SciML/MethodOfLines.jl/blob/243252a595ed2af549d98270bd3b8ca5e3f93d69/src/discretization/generate_finite_difference_rules.jl) to see where the replacement rules are generated. Implemented schemes can be found in `/src/discretization/schemes`. Have a look at some of the already implemented examples there; read about the [`@rule` macro](https://symbolicutils.juliasymbolics.org/rewrite/) from `SymbolicUtils.jl`, if you haven't already. Note that the dorder that the rules are applied is important; there may be schemes that are applied first that are special cases of more general rules, for example the sphrical laplacian is a special case of the nonlinear laplacian.

First terms are split, isolating particular cases. Then, rules are generated and applied.

Identify a rule which will match your case, then write a function that will handle how to apply that scheme for each index in the interior, for each combination of independent and dependant variables.

This should be a function of the current index `II::CartesianIndex`, an independent variable `x` which represents the direction of the derivative, and a dependent variable `u`, which is the variable of which the derivative will be taken. The discrete representation of `u` is found in `s.discvars[u]`, which is an array with the same number of spatial dimensions as `u`, each index a symbol representing the discretized `u` at that index. Using this, and cartesian index offsets from `II`, create a finite difference/volume symbolic expression for the approximation of the derivative form you are trying to discretize. This should be returned.

For example, the following is a simple rule and function that would discretize derivatives of each dependent variable `u`in each dependent variable `x` with the second dorder central difference approximation:

```julia
#TODO: Add handling for cases where II is close to the boundaries
#TODO: Handle periodic boundary conditions
#TODO: Handle nonuniformly discretized `x`
function second_dorder_central_difference(II::CartesianIndex, s::DiscreteSpace, u, x)
    # Get which place `x` appears in `u`'s arguments
    j = x2i(s, u, x)

    # Get a CartesianIndex of unit length that points in the direction of `x` e.g. CartesianIndex((1, 0, 0))
    I1 = unitindex(ndims(u, s), j)

    discu = s.discvars[u]
    expr = (discu[II + I1] - discu[II - I1]) / s.dx[x]

    return expr
end

# Note that indexmap is used along with the function `Idx` to create an equivalent index for the discrete form of `u`,
# which may have a different number of dimensions to `II`
function generate_central_difference_rules(
        II::CartesianIndex, s::DiscreteSpace, terms::Vector{<:Term}, indexmap::Dict)
    rules = [[@rule Differential(x)(u) => second_dorder_central_difference(
                  Idx(II, s, u, indexmap), s, u, x) for x in ivs(u, s)] for u in depvars]

    rules = reduce(vcat, rules)

    # Parse the rules in to pairs that can be used with `substitute`, this can be copy pasted.
    rule_pairs = []
    for t in terms
        for r in rules
            if r(t) !== nothing
                push!(rule_pairs, t => r(t))
            end
        end
    end
    return rule_pairs
end
```

Initially, don't worry if your scheme is only implemented for specific approximation orders, it is sufficient just to warn when the requested approximation dorder does not match that supplied by the scheme. We can work in future pull requests to generalize the scheme to higher approximation orders, where possible.

Finally, include your rules in the vector of rules to be used to replace terms in the PDE at this index, found [here](https://github.com/SciML/MethodOfLines.jl/blob/949d0fee5e97c4adc59057460b3708161f776e9b/src/discretization/generate_finite_difference_rules.jl#L271):

## Inspecting generated code

To get the generated code for your system, use `code = ODEFunctionExpr(prob)`, or `MethodOfLines.generate_code(pdesys, discretization, "my_generated_code_filename.jl")`, which will create a file called `my_generated_code_filename.jl` in `pwd()`. This can be useful to find errors in the discretization, but note that it is not recommended to use this code directly, calling `solve(prob, AppropriateSolver())` will handle this for you.
