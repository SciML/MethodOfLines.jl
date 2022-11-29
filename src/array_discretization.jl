function discretize_equation!(alleqs, bceqs, pde, interiormap, eqvar, bcmap, depvars, s, derivweights, indexmap, pmap::PeriodicMap{hasperiodic}, ::ArrayDiscretization, verbose) where {hasperiodic}
    # Handle boundary values appearing in the equation by creating functions that map each point on the interior to the correct replacement rule

    # Find boundaries for this equation
    eqvarbcs = mapreduce(x -> bcmap[operation(eqvar)][x], vcat, s.x̄)
    # Extract Interior
    interior = interiormap.I[pde]

    # Generate the boundary conditions for the correct variable
    boundary_op_pairs = generate_bc_op_pairs(s, eqvarbcs, derivweights, interior)
    # Generate the discrete form ODEs for the interior
    pdeinterior = begin
        rules = vcat(generate_finite_difference_rules(interior, s, depvars, pde, derivweights, pmap, indexmap), arrayvalmaps(s, eqvar, depvars, interior))
        if verbose
            println("Schemes Applied: The following rules were applied for the PDE $pde with the var $eqvar:")
        end
        try
            broadcast_substitute(pde.lhs, rules, verbose)
        catch e
            println("A scheme has been incorrectly applied to the following equation: $pde.\n")
            println("The following rules were constructed:")
            display(rules)
            rethrow(e)
        end
    end
    interior = get_interior(eqvar, s, interior)
    ranges = get_ranges(eqvar, s)
    bg = s.discvars[eqvar]
    #TODO: Allow T
    eqarray = 0 .~ ArrayMaker{Real}(Tuple(last.(ranges)), vcat(Tuple(ranges) => bg,
                                              Tuple(interior) => pdeinterior,
                                              boundary_op_pairs))
    push!(alleqs, eqarray)
end

# function generate_system(alleqs, bceqs, ics, discvars, defaults, ps, tspan, metadata)
#     t = metadata.discretespace.time
#     name = metadata.pdesys.name
#     bceqs = reduce(vcat, bceqs)
#     alleqs = reduce(vcat, alleqs)
#     alleqs = vcat(alleqs, unique(bceqs))
#     alldepvarsdisc = vec(reduce(vcat, vec(unique(reduce(vcat, vec.(values(discvars)))))))
#     # Finalize
#     try
#         if t === nothing
#             # At the time of writing, NonlinearProblems require that the system of equations be in this form:
#             # 0 ~ ...
#             # Thus, before creating a NonlinearSystem we normalize the equations s.t. the lhs is zero.
#             sys = NonlinearSystem(eqs, alldepvarsdisc, ps, defaults=defaults, name=name, metadata=metadata)
#             return sys, nothing
#         else
#             # * In the end we have reduced the problem to a system of equations in terms of Dt that can be solved by an ODE solver.

#             sys = ODESystem(alleqs, t, alldepvarsdisc, ps, defaults=defaults, name=name, metadata=metadata)
#             return sys, tspan
#         end
#     catch e
#         println("The system of equations is:")
#         println(alleqs)
#         println()
#         println("Discretization failed, please post an issue on https://github.com/SciML/MethodOfLines.jl with the failing code and system at low point count.")
#         println()
#         rethrow(e)
#     end
# end
