# Method of lines discretization scheme

function interface_errors(depvars, indvars, discretization)
    for x in indvars
        @assert findfirst(dxs -> isequal(x, dxs[1].val), discretization.dxs) !== nothing "Variable $x has no step size"
    end
    if !(typeof(discretization.advection_scheme) ∈ [UpwindScheme, WENOScheme])
        throw(ArgumentError("Only `UpwindScheme()` and `WENOScheme()` are supported advection schemes."))
    end
end

function SciMLBase.symbolic_discretize(pdesys::PDESystem, discretization::MethodOfLines.MOLFiniteDifference{G}) where {G}
    t = discretization.time
    ############################
    # System Parsing and Transformation
    ############################
    # Parse the variables in to the right form and store useful information about the system
    v = VariableMap(pdesys, t)
    # Check for basic interface errors
    interface_errors(v.ū, v.x̄, discretization)
    # Extract tspan
    tspan = t !== nothing ? v.intervals[t] : nothing
    # Find the derivative orders in the bcs
    bcorders = Dict(map(x -> x => d_orders(x, pdesys.bcs), all_ivs(v)))
    # Create a map of each variable to their boundary conditions including initial conditions
    boundarymap = parse_bcs(pdesys.bcs, v, bcorders)
    # Generate a map of each variable to whether it is periodic in a given direction
    pmap = PeriodicMap(boundarymap, v)
    # Transform system so that it is compatible with the discretization
    if discretization.should_transform
        pdesys = transform_pde_system!(v, boundarymap, pmap, pdesys)
    end

    # Check if the boundaries warrant using ODAEProblem, as long as this is allowed in the interface
    use_ODAE = discretization.use_ODAE
    if use_ODAE
        bcivmap = reduce((d1, d2) -> mergewith(vcat, d1, d2), collect(values(boundarymap)))
        allbcs = mapreduce(x -> bcivmap[x], vcat, v.x̄)
        if all(bc -> bc.order > 0, allbcs)
            use_ODAE = false
        end
    end

    pdeeqs = pdesys.eqs
    bcs = pdesys.bcs

    # @show alldepvars
    # @show allindvars

    ############################
    # Discretization of system
    ############################
    alleqs = []
    bceqs = []
    # * We wamt to do this in 2 passes
    # * First parse the system and BCs, replacing with DiscreteVariables and DiscreteDerivatives
    # * periodic parameters get type info on whether they are periodic or not, and if they join up to any other parameters
    # * Then we can do the actual discretization by recursively indexing in to the DiscreteVariables

    # Create discretized space and variables, this is called `s` throughout
    s = DiscreteSpace(v, discretization)
    # Get the interior and variable to solve for each equation
    interiormap = InteriorMap(pdeeqs, boundarymap, s, discretization, pmap)
    # Get the derivative orders appearing in each equation
    pdeorders = Dict(map(x -> x => d_orders(x, pdeeqs), v.x̄))
    bcorders = Dict(map(x -> x => d_orders(x, bcs), v.x̄))
    orders = Dict(map(x -> x => collect(union(pdeorders[x], bcorders[x])), v.x̄))

    # Generate finite difference weights
    derivweights = DifferentialDiscretizer(pdesys, s, discretization, orders)

    ics = t === nothing ? [] : mapreduce(u -> boundarymap[u][t], vcat, operation.(s.ū))

    bcmap = Dict(map(collect(keys(boundarymap))) do u
        u => Dict(map(s.x̄) do x
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

        # Read the independent variables
        # ignore if the only argument is [t]
        indvars = Set(filter(xs -> !isequal(xs, [t]), map(arguments, depvars)))
        # get all parameters in the equation
        allx̄ = Set(filter(!isempty, map(u -> filter(x -> t === nothing || !isequal(x, t.val), arguments(u)), depvars)))
        # Handle the case where there are no independent variables apart from time
        if isempty(allx̄)
            rules = varmaps(s, depvars, CartesianIndex(), Dict([]))
            push!(alleqs, substitute(pde.lhs, rules) ~ substitute(pde.rhs, rules))
        else
            # make sure there is only one set of independent variables per equation
            @assert length(allx̄) == 1
            pdex̄ = first(allx̄)
            @assert length(indvars) == 1

            eqvar = interiormap.var[pde]

            eqvarbcs = mapreduce(x -> bcmap[operation(eqvar)][x], vcat, s.x̄)

            # * Assumes that all variables in the equation have same dimensionality except edgevals
            args = params(eqvar, s)
            indexmap = Dict([args[i] => i for i in 1:length(args)])

            # Handle boundary values appearing in the equation by creating functions that map each point on the interior to the correct replacement rule
            # Generate replacement rule gen closures for the boundary values like u(t, 1)
            boundaryvalfuncs = generate_boundary_val_funcs(s, depvars, bcmap, indexmap, derivweights)

            # Generate the boundary conditions for the correct variable
            for boundary in eqvarbcs
                generate_bc_eqs!(bceqs, s, boundaryvalfuncs, interiormap, boundary)
            end
            # Generate extrapolation eqs
            generate_extrap_eqs!(bceqs, pde, eqvar, s, derivweights, interiormap, pmap)

            # Set invalid corner points to zero
            generate_corner_eqs!(bceqs, s, interiormap, ndims(s.discvars[eqvar]), eqvar)

            # Generate the equations for the interior points
            pdeeqs = discretize_equation(pde, interiormap.I[pde], eqvar, depvars, s, derivweights, indexmap, boundaryvalfuncs, pmap)

            push!(alleqs, pdeeqs)
        end
    end

    u0 = generate_ic_defaults(ics, s)
    bceqs = reduce(vcat, bceqs)
    alleqs = reduce(vcat, alleqs)
    alldepvarsdisc = unique(reduce(vcat, vec.(values(s.discvars))))

    # Finalize
    defaults = pdesys.ps === nothing || pdesys.ps === SciMLBase.NullParameters() ? u0 : vcat(u0, pdesys.ps)
    ps = pdesys.ps === nothing || pdesys.ps === SciMLBase.NullParameters() ? Num[] : first.(pdesys.ps)
    # Combine PDE equations and BC equations
    metadata = MOLMetadata(s, discretization, pdesys, use_ODAE)
    try
        if t === nothing
            # At the time of writing, NonlinearProblems require that the system of equations be in this form:
            # 0 ~ ...
            # Thus, before creating a NonlinearSystem we normalize the equations s.t. the lhs is zero.
            eqs = map(eq -> 0 ~ eq.rhs - eq.lhs, vcat(alleqs, unique(bceqs)))
            sys = NonlinearSystem(eqs, vec(reduce(vcat, vec(alldepvarsdisc))), ps, defaults=Dict(defaults), name=pdesys.name, metadata=metadata)
            return sys, nothing
        else
            # * In the end we have reduced the problem to a system of equations in terms of Dt that can be solved by an ODE solver.

            sys = ODESystem(vcat(alleqs, unique(bceqs)), t, vec(reduce(vcat, vec(alldepvarsdisc))), ps, defaults=Dict(defaults), name=pdesys.name, metadata=metadata)
            return sys, tspan
        end
    catch e
        println("The system of equations is:")
        println(vcat(alleqs, unique(bceqs)))
        println()
        println("Discretization failed, please post an issue on https://github.com/SciML/MethodOfLines.jl with the failing code and system at low point count.")
        println()
        rethrow(e)
    end
end

function discretize_equation(pde, interior, eqvar, depvars, s, derivweights, indexmap, boundaryvalfuncs, pmap::PeriodicMap{hasperiodic}) where {hasperiodic}
    return vec(map(interior) do II
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
end

function SciMLBase.discretize(pdesys::PDESystem,discretization::MethodOfLines.MOLFiniteDifference)
    sys, tspan = SciMLBase.symbolic_discretize(pdesys, discretization)
    try
        simpsys = structural_simplify(sys)
        if tspan === nothing
            return prob = NonlinearProblem(simpsys, ones(length(simpsys.states)); discretization.kwargs...)
        else
            # Use ODAE if nessesary
            if getfield(sys, :metadata) isa MOLMetadata && getfield(sys, :metadata).use_ODAE
                add_metadata!(simpsys.metadata, DAEProblem(simpsys; discretization.kwargs...))
                return prob = ODAEProblem(simpsys, Pair[], tspan; discretization.kwargs...)
            else
                return prob = ODEProblem(simpsys, Pair[], tspan; discretization.kwargs...)
            end
        end
    catch e
        error_analysis(sys, e)
    end
end

function get_discrete(pdesys, discretization)
    domain = pdesys.domain

    t = discretization.time

    depvar_ops = map(x->operation(x.val),pdesys.depvars)
    # Get all dependent variables in the correct type
    alldepvars = get_all_depvars(pdesys, depvar_ops)
    alldepvars = filter(u -> !any(map(x-> x isa Number, arguments(u))), alldepvars)
    # Get all independent variables in the correct type, removing time from the list
    allindvars = remove(collect(filter(x->!(x isa Number), reduce(union, filter(xs->(!isequal(xs, [t])), map(arguments, alldepvars))))), t)
    #@show allindvars, typeof.(allindvars)

    @warn "`get_discrete` is deprecated, The solution is now automatically wrapped in a PDESolution object, which retrieves the shaped solution much faster than the previously recommended method. See the documentation for more information."

    interface_errors(alldepvars, allindvars, discretization)
    # @show alldepvars
    # @show allindvars

    # Get tspan
    tspan = nothing
    # Check that inputs make sense
    if t !== nothing
        tdomain = pdesys.domain[findfirst(d->isequal(t.val, d.variables), pdesys.domain)]
        @assert tdomain.domain isa DomainSets.Interval
        tspan = (DomainSets.infimum(tdomain.domain), DomainSets.supremum(tdomain.domain))
    end
    alleqs = []
    bceqs = []

    # Create discretized space and variables
    s = DiscreteSpace(domain, alldepvars, allindvars, discretization)

    return Dict(vcat([Num(x) => s.grid[x] for x in s.x̄], [Num(u) => s.discvars[u] for u in s.ū]) )
end

function ModelingToolkit.ODEFunctionExpr(pdesys::PDESystem,discretization::MethodOfLines.MOLFiniteDifference)
    sys, tspan = SciMLBase.symbolic_discretize(pdesys, discretization)
    try
        if tspan === nothing
            @assert true "Codegen for NonlinearSystems is not yet implemented."
        else
            simpsys = structural_simplify(sys)
            return ODEFunctionExpr(simpsys)
        end
    catch e
        println("The system of equations is:")
        println(sys.eqs)
        println()
        println("Discretization failed, please post an issue on https://github.com/SciML/MethodOfLines.jl with the failing code and system at low point count.")
        println()
        rethrow(e)
    end
end

function generate_code(pdesys::PDESystem,discretization::MethodOfLines.MOLFiniteDifference,filename="code.jl")
    code = ODEFunctionExpr(pdesys, discretization)
    rm(filename)
    open(filename, "a") do io
        println(io, code)
    end
end
