# Pointwise discretization of a single interior point. `ArrayDiscretization` uses this
# for equations, boundaries and corners whose pattern has no slice representation, so
# the scalar form is a fallback within the array strategy rather than a strategy of
# its own.
function discretize_equation_at_point(
        II, s, depvars, pde, derivweights, bcmap, eqvar, indexmap, boundaryvalfuncs
    )
    boundaryrules = mapreduce(f -> f(II), vcat, boundaryvalfuncs, init = [])
    rules = vcat(
        generate_finite_difference_rules(
            II, s, depvars, pde, derivweights, bcmap, indexmap
        ),
        boundaryrules,
        valmaps(s, eqvar, depvars, II, indexmap)
    )
    try
        rdict = Dict(rules)
        return expand_derivatives(pde_substitute(pde.lhs, rdict)) ~ pde_substitute(pde.rhs, rdict)
    catch e
        println("A scheme has been incorrectly applied to the following equation: $pde.\n")
        println("The following rules were constructed at index $II:")
        display(rules)
        rethrow(e)
    end
end
