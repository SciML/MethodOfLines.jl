function discretize_equation!(alleqs, bceqs, pde, interior, eqvar, bcmap, depvars, s, derivweights, indexmap, pmap::PeriodicMap{hasperiodic}) where {hasperiodic}
    # Handle boundary values appearing in the equation by creating functions that map each point on the interior to the correct replacement rule
    # Generate replacement rule gen closures for the boundary values like u(t, 1)
    boundaryvalfuncs = generate_boundary_val_funcs(s, depvars, bcmap, indexmap, derivweights)
    # Find boundaries for this equation
    eqvarbcs = mapreduce(x -> bcmap[operation(eqvar)][x], vcat, s.x̄)
    # Generate the boundary conditions for the correct variable
    for boundary in eqvarbcs
        generate_bc_eqs!(bceqs, s, boundaryvalfuncs, interiormap, boundary)
    end
    # Generate extrapolation eqs
    generate_extrap_eqs!(bceqs, pde, eqvar, s, derivweights, interiormap, pmap)

    # Set invalid corner points to zero
    generate_corner_eqs!(bceqs, s, interiormap, ndims(s.discvars[eqvar]), eqvar)

    eqs = vec(map(interior) do II
        boundaryrules = mapreduce(f -> f(II), vcat, boundaryvalfuncs)
        rules = vcat(generate_finite_difference_rules(II, s, depvars, pde, derivweights, pmap, indexmap), boundaryrules, valmaps(s, eqvar, depvars, II, indexmap))
        try
            substitute(pde.lhs, rules) ~ substitute(pde.rhs, rules)
        catch e
            println("A scheme has been incorrectly applied to the following equation: $pde.\n")
            println("The following rules were constructed at index $II:")
            display(rules)
            rethrow(e)
        end

    end)
    push!(alleqs, eqs)
end

function generate_system(alleqs, bceqs, ics, ps, defaults, tspan, metadata)
    u0 = generate_ic_defaults(ics, s)
    bceqs = reduce(vcat, bceqs)
    alleqs = reduce(vcat, alleqs)
    alleqs = vcat(alleqs, unique(bceqs))
    alldepvarsdisc = vec(reduce(vcat, vec(unique(reduce(vcat, vec.(values(s.discvars)))))))
    # Finalize
    try
        if t === nothing
            # At the time of writing, NonlinearProblems require that the system of equations be in this form:
            # 0 ~ ...
            # Thus, before creating a NonlinearSystem we normalize the equations s.t. the lhs is zero.
            eqs = map(eq -> 0 ~ eq.rhs - eq.lhs, alleqs)
            sys = NonlinearSystem(eqs, alldepvarsdisc, ps, defaults=defaults, name=pdesys.name, metadata=metadata)
            return sys, nothing
        else
            # * In the end we have reduced the problem to a system of equations in terms of Dt that can be solved by an ODE solver.

            sys = ODESystem(alleqs, t, alldepvarsdisc, ps, defaults=defaults, name=pdesys.name, metadata=metadata)
            return sys, tspan
        end
    catch e
        println("The system of equations is:")
        println(alleqs)
        println()
        println("Discretization failed, please post an issue on https://github.com/SciML/MethodOfLines.jl with the failing code and system at low point count.")
        println()
        rethrow(e)
    end
end
