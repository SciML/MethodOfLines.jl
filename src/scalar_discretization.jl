function PDEBase.discretize_equation!(
        disc_state::PDEBase.EquationState, pde::Equation, interiormap,
        eqvar, bcmap, depvars, s::DiscreteSpace, derivweights, indexmap,
        discretization::MOLFiniteDifference{G, D}
    ) where {G, D <: ScalarizedDiscretization}
    # Handle boundary values appearing in the equation by creating functions that map each point on the interior to the correct replacement rule
    # Generate replacement rule gen closures for the boundary values like u(t, 1)
    boundaryvalfuncs = generate_boundary_val_funcs(
        s, depvars, bcmap, indexmap, derivweights
    )
    # Find boundaries for this equation
    eqvarbcs = mapreduce(x -> bcmap[operation(eqvar)][x], vcat, s.x̄)
    # Generate the boundary conditions for the correct variable
    for boundary in eqvarbcs
        generate_bc_eqs!(disc_state, s, boundaryvalfuncs, interiormap, boundary)
    end
    # Generate extrapolation eqs
    generate_extrap_eqs!(disc_state, pde, eqvar, s, derivweights, interiormap, bcmap)

    # Set invalid corner points to zero
    generate_corner_eqs!(disc_state, s, interiormap, ndims(s.discvars[eqvar]), eqvar)
    # Extract Interior
    interior = interiormap.I[pde]
    # Generate the discrete form ODEs for the interior
    eqs = if length(interior) == 0
        II = CartesianIndex()
        discretize_equation_at_point(
            II, s, depvars, pde, derivweights, bcmap, eqvar, indexmap, boundaryvalfuncs
        )
    else
        vec(
            map(interior) do II
                discretize_equation_at_point(
                    II, s, depvars, pde, derivweights, bcmap,
                    eqvar, indexmap, boundaryvalfuncs
                )
            end
        )
    end

    return vcat!(disc_state.eqs, eqs)
end

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
        return expand_derivatives(substitute(pde.lhs, rules)) ~ substitute(pde.rhs, rules)
    catch e
        println("A scheme has been incorrectly applied to the following equation: $pde.\n")
        println("The following rules were constructed at index $II:")
        display(rules)
        rethrow(e)
    end
end
