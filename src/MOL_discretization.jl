# Method of lines discretization scheme

function PDEBase.interface_errors(pdesys::PDESystem, v::PDEBase.VariableMap, discretization::MOLFiniteDifference)
    depvars = v.ū
    indvars = v.x̄
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

function PDEBase.check_boundarymap(boundarymap, discretization::MOLFiniteDifference)
    bs = filter_interfaces(flatten_vardict(boundarymap))
    for b in bs
        dx1 = discretization.dxs[Num(b.x)]
        dx2 = discretization.dxs[Num(b.x2)]
        if dx1 != dx2
            throw(ArgumentError("The step size of the connected variables $(b.x) and $(b.x2) must be the same. If you need nonuniform interface boundaries please post an issue on GitHub."))
        end
    end
end

function get_discrete(pdesys, discretization)
    t = get_time(discretization)
    cardinalize_eqs!(pdesys)

    ############################
    # System Parsing and Transformation
    ############################
    # Parse the variables in to the right form and store useful information about the system
    v = VariableMap(pdesys, discretization)
    # Check for basic interface errors
    interface_errors(pdesys, v, discretization)
    # Extract tspan
    tspan = t !== nothing ? v.intervals[t] : nothing
    # Find the derivative orders in the bcs
    bcorders = Dict(map(x -> x => d_orders(x, pdesys.bcs), all_ivs(v)))
    # Create a map of each variable to their boundary conditions including initial conditions
    boundarymap = parse_bcs(pdesys.bcs, v, bcorders)
    # Check that the boundary map is valid
    check_boundarymap(boundarymap, discretization)

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
