
function PDEBase.transform_pde_system!(v::VariableMap, boundarymap,
                               pdesys::PDESystem, disc::MOLFiniteDifference{G}) where {G<:StaggeredGrid}
    @parameters t ξ η
    @variables ρ(..) ϕ(..)
    Dt = Differential(t);
    Dξ = Differential(ξ);
    Dη = Differential(η);

    a = 0.1;
    L = 3.0;
    dx = disc.dxs[PDEBase.remove(pdesys.ivs, t)...];

    eq = [Dt(ρ(t,ξ)) + Dη(ϕ(t,η)) ~ 0,
          Dt(ϕ(t,η)) + a^2 * Dξ(ρ(t,ξ)) ~ 0]
    bcs = [ρ(0.0,ξ) ~ exp(-(ξ-L/2)^2),
           ϕ(0.0,η) ~ 0.0,
           ρ(t,0) ~ ρ(t,L-dx),
           ϕ(t,dx) ~ ϕ(t,L)];


    domains = [t in Interval(0.0, 10.0),
               ξ in Interval(0.0, L-dx),
               η in Interval(dx, L)];

    @named pdesys = PDESystem(eq, bcs, domains, [t,ξ,η], [ρ(t,ξ), ϕ(t,η)]);

    discretization = MOLFiniteDifference([ξ=>2*dx, η=>2*dx], t, grid_align=MethodOfLines.StaggeredGrid(), approx_order = 2);

    w = VariableMap(pdesys, disc);
    # v.ū .= w.ū
    # PDEBase.update_varmap!(v, w.ū...);
    
    @info "leaving tranformation"
    return pdesys, w, discretization;
end

function PDEBase.should_transform(pdesys::PDESystem, disc::MOLFiniteDifference{G}) where {G<:StaggeredGrid}
    return true;
end

function SciMLBase.symbolic_discretize(pdesys::PDESystem, discretization::MOLFiniteDifference{G}) where {G<:StaggeredGrid}
    @info "inside new stuff"
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
        pdesys,v,discretization = transform_pde_system!(v, boundarymap, pdesys, discretization)
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
