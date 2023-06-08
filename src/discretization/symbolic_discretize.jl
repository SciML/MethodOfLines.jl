function cardinalize_eqs!(pdesys)
    for (i, eq) in enumerate(pdesys.eqs)
        pdesys.eqs[i] = eq.lhs - eq.rhs ~ 0
    end
end

function PDEBase.symbolic_discretize(pdesys::PDEBase.PDESystem, discretization::MOLFiniteDifference)
    symbolic_discretize(pdesys, discretization, discretization.grid_align)
end

function symbolic_discretize(pdesys::PDEBase.PDESystem, discretization::MOLFiniteDifference, ::AbstractGrid)
    @info "inside MethodOfLines extension of PDEBase.symbolic_discretize:AbstractGrid"
    t = get_time(discretization)
    cardinalize_eqs!(pdesys)
    pdesys, replaced_vars = make_pdesys_compatible(pdesys)

    ############################
    # System Parsing and Transformation
    ############################
    # Parse the variables in to the right form and store useful information about the system
    v = VariableMap(pdesys, discretization, replaced_vars = replaced_vars)
    # Check for basic interface errors
    interface_errors(pdesys, v, discretization)
    # Extract tspan
    tspan = t !== nothing ? v.intervals[t] : nothing
    # Find the derivative orders in the bcs
    bcorders = Dict(map(x -> x => d_orders(x, pdesys.bcs), all_ivs(v)))
    # Create a map of each variable to their boundary conditions including initial conditions
    boundarymap = parse_bcs(pdesys.bcs, v, bcorders)
    # Check that the boundary map is valid
    check_boundarymap(boundarymap, v, discretization)

    # Transform system so that it is compatible with the discretization
    if should_transform(pdesys, discretization)
        pdesys = transform_pde_system!(v, boundarymap, pdesys, discretization)
    end

    pdeeqs = pdesys.eqs
    bcs = pdesys.bcs

    ############################
    # Discretization of system
    ############################
    disc_state = construct_disc_state(discretization)

    # Create discretized space and variables, this is called `s` throughout
    s = construct_discrete_space(v, discretization)
    # Get the interior and variable to solve for each equation
    #TODO: do the interiormap before and independent of the discretization i.e. `s`
    vareqmap = construct_var_equation_mapping(pdeeqs, boundarymap, s, discretization)
    # Get the derivative orders appearing in each equation
    pdeorders = Dict(map(x -> x => d_orders(x, pdeeqs), indvars(v)))
    bcorders = Dict(map(x -> x => d_orders(x, bcs), indvars(v)))
    orders = Dict(map(x -> x => collect(union(pdeorders[x], bcorders[x])), indvars(v)))

    # Generate finite difference weights
    derivweights = construct_differential_discretizer(pdesys, s, discretization, orders)

    # Seperate bcs and ics
    ics = t === nothing ? [] : mapreduce(u -> boundarymap[u][t], vcat, operation.(depvars(v)))

    bcmap = Dict(map(collect(keys(boundarymap))) do u
                 u => Dict(map(indvars(v)) do x
                           x => boundarymap[u][x]
                           end)
                 end)

    ####
    # Loop over equations, Discretizing them and their dependent variables' boundary conditions
    ####
    for pde in pdeeqs
        # Read the dependent variables on both sides of the equation
        depvars_lhs = get_depvars(pde.lhs, v.depvar_ops)
        depvars_rhs = get_depvars(pde.rhs, v.depvar_ops)
        depvars = collect(depvars_lhs ∪ depvars_rhs)
        depvars = filter(u -> !any(map(x -> x isa Number, arguments(u))), depvars)

        eqvar = get_eqvar(vareqmap, pde)

        # * Assumes that all variables in the equation have same dimensionality except edgevals
        args = PDEBase.ivs(eqvar, v)
        indexmap = Dict([args[i] => i for i in 1:length(args)])
        # Generate the equations for the interior points
        discretize_equation!(disc_state, pde, vareqmap, eqvar, bcmap,
                             depvars, s, derivweights, indexmap, discretization)
    end

    u0 = generate_ic_defaults(ics, s, discretization)

    # Combine PDE equations and BC equations
    metadata = generate_metadata(s, discretization, pdesys, boundarymap)

    return generate_system(disc_state, s, u0, tspan, metadata, discretization)
end

function symbolic_discretize(pdesys::PDEBase.PDESystem, discretization::MOLFiniteDifference, ::StaggeredGrid)
    @info "inside MethodOfLines extension of PDEBase.symbolic_discretize:StaggeredGrid"
    return symbolic_discretize(pdesys::PDEBase.PDESystem, discretization, CenterAlignedGrid());
end
