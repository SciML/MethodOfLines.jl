# Method of lines discretization scheme

function interface_errors(depvars, indvars, discretization)
    for x in indvars
        @assert haskey(discretization.dxs, Num(x)) || haskey(discretization.dxs, x) "Variable $x has no step size"
    end
    if !any(s -> discretization.advection_scheme isa s,  [UpwindScheme, FunctionalScheme])
        throw(ArgumentError("Only `UpwindScheme()` and `FunctionalScheme()` are supported advection schemes. Got $(typeof(discretization.advection_scheme))."))
    end
    if !(typeof(discretization.disc_strategy) ∈ [ScalarizedDiscretization])
        throw(ArgumentError("Only `ScalarizedDiscretization()` are supported discretization strategies."))
    end
end

function check_boundarymap(boundarymap, discretization)
    bs = filter_interfaces(flatten_vardict(boundarymap))
    for b in bs
        dx1 = discretization.dxs[Num(b.x)]
        dx2 = discretization.dxs[Num(b.x2)]
        if dx1 != dx2
            throw(ArgumentError("The step size of the connected variables $(b.x) and $(b.x2) must be the same. If you need nonuniform interface boundaries please post an issue on GitHub."))
        end
    end
end

function SciMLBase.symbolic_discretize(pdesys::PDESystem, discretization::MethodOfLines.MOLFiniteDifference{G}) where {G}
    t = discretization.time
    disc_strategy = discretization.disc_strategy
    cardinalize_eqs!(pdesys)

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

    check_boundarymap(boundarymap, discretization)

    # Transform system so that it is compatible with the discretization
    if discretization.should_transform
        if has_interfaces(boundarymap)
            @warn "The system contains interface boundaries, which are not compatible with system transformation. The system will not be transformed. Please post an issue if you need this feature."
        else
            pdesys = transform_pde_system!(v, boundarymap, pdesys)
        end
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

    ############################
    # Discretization of system
    ############################
    alleqs = []
    bceqs = []

    # Create discretized space and variables, this is called `s` throughout
    s = DiscreteSpace(v, discretization)
    # Get the interior and variable to solve for each equation
    #TODO: do the interiormap before and independent of the discretization i.e. `s`
    interiormap = InteriorMap(pdeeqs, boundarymap, s, discretization)
    # Get the derivative orders appearing in each equation
    pdeorders = Dict(map(x -> x => d_orders(x, pdeeqs), v.x̄))
    bcorders = Dict(map(x -> x => d_orders(x, bcs), v.x̄))
    orders = Dict(map(x -> x => collect(union(pdeorders[x], bcorders[x])), v.x̄))

    # Generate finite difference weights
    derivweights = DifferentialDiscretizer(pdesys, s, discretization, orders)

    # Seperate bcs and ics
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

        eqvar = interiormap.var[pde]

        # * Assumes that all variables in the equation have same dimensionality except edgevals
        args = ivs(eqvar, s)
        indexmap = Dict([args[i] => i for i in 1:length(args)])
        if disc_strategy isa ScalarizedDiscretization
            # Generate the equations for the interior points
            discretize_equation!(alleqs, bceqs, pde, interiormap, eqvar, bcmap,
                depvars, s, derivweights, indexmap)
        else
            throw(ArgumentError("Only ScalarizedDiscretization is currently supported"))
        end
    end

    u0 = generate_ic_defaults(ics, s, disc_strategy)

    defaults = Dict(pdesys.ps === nothing || pdesys.ps === SciMLBase.NullParameters() ? u0 : vcat(u0, pdesys.ps))
    ps = pdesys.ps === nothing || pdesys.ps === SciMLBase.NullParameters() ? Num[] : first.(pdesys.ps)
    # Combine PDE equations and BC equations
    metadata = MOLMetadata(s, discretization, pdesys, use_ODAE)

    return generate_system(alleqs, bceqs, ics, s.discvars, defaults, ps, tspan, metadata)
end

function SciMLBase.discretize(pdesys::PDESystem,discretization::MethodOfLines.MOLFiniteDifference; analytic = nothing, kwargs...)
    sys, tspan = SciMLBase.symbolic_discretize(pdesys, discretization)
    try
        simpsys = structural_simplify(sys)
        if tspan === nothing
            add_metadata!(get_metadata(sys), sys)
            return prob = NonlinearProblem(simpsys, ones(length(simpsys.states)); discretization.kwargs..., kwargs...)
        else
            # Use ODAE if nessesary
            if getfield(sys, :metadata) isa MOLMetadata && getfield(sys, :metadata).use_ODAE
                add_metadata!(get_metadata(simpsys), DAEProblem(simpsys; discretization.kwargs..., kwargs...))
                return prob = ODAEProblem(simpsys, Pair[], tspan; discretization.kwargs..., kwargs...)
            else
                add_metadata!(get_metadata(simpsys), sys)
                prob = ODEProblem(simpsys, Pair[], tspan; discretization.kwargs..., kwargs...)
                if analytic === nothing
                    return prob
                else
                    f = ODEFunction(pdesys, discretization, analytic=analytic, discretization.kwargs..., kwargs...)

                    return ODEProblem(f, prob.u0, prob.tspan, prob.p; discretization.kwargs..., kwargs...)
                end
            end
        end
    catch e
        error_analysis(sys, e)
    end
end

function get_discrete(pdesys, discretization)
    t = discretization.time
    disc_strategy = discretization.disc_strategy
    cardinalize_eqs!(pdesys)

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

    # Transform system so that it is compatible with the discretization
    if discretization.should_transform
        if has_interfaces(boundarymap)
            @warn "The system contains interface boundaries, which are not compatible with system transformation. The system will not be transformed. Please post an issue if you need this feature."
        else
            pdesys = transform_pde_system!(v, boundarymap, pdesys)
        end
    end

    s = DiscreteSpace(v, discretization)

    return Dict(vcat([Num(x) => s.grid[x] for x in s.x̄], [Num(u) => s.discvars[u] for u in s.ū]))
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

function SciMLBase.ODEFunction(pdesys::PDESystem, discretization::MethodOfLines.MOLFiniteDifference; analytic=nothing, kwargs...)
    sys, tspan = SciMLBase.symbolic_discretize(pdesys, discretization)
    try
        if tspan === nothing
            @assert true "Codegen for NonlinearSystems is not yet implemented."
        else
            simpsys = structural_simplify(sys)
            if analytic !== nothing
                analytic = analytic isa Dict ? analytic : Dict(analytic)
                s = getfield(sys, :metadata).discretespace
                us = get_states(simpsys)
                gridlocs = get_gridloc.(us, (s,))
                f_analytic = generate_function_from_gridlocs(analytic, gridlocs, s)
            end
            return ODEFunction(simpsys; analytic = f_analytic, discretization.kwargs..., kwargs...)
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

function generate_code(pdesys::PDESystem,discretization::MethodOfLines.MOLFiniteDifference,filename="generated_code_of_pdesys.jl")
    code = ODEFunctionExpr(pdesys, discretization)
    rm(filename; force = true)
    open(filename, "a") do io
        println(io, code)
    end
end
