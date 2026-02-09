# Method of lines discretization scheme

function PDEBase.interface_errors(
        pdesys::PDESystem, v::PDEBase.VariableMap, discretization::MOLFiniteDifference
    )
    depvars = v.ū
    indvars = v.x̄
    for x in indvars
        @assert haskey(discretization.dxs, Num(x))||haskey(discretization.dxs, x) "Variable $x has no step size"
    end
    if !any(s -> discretization.advection_scheme isa s, [UpwindScheme, FunctionalScheme])
        throw(ArgumentError("Only `UpwindScheme()` and `FunctionalScheme()` are supported advection schemes. Got $(typeof(discretization.advection_scheme))."))
    end
    return if !(typeof(discretization.disc_strategy) ∈ [ScalarizedDiscretization])
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
    return
end

function get_discrete(pdesys, discretization)
    t = get_time(discretization)
    PDEBase.cardinalize_eqs!(pdesys)

    ############################
    # System Parsing and Transformation
    ############################
    # Parse the variables in to the right form and store useful information about the system
    v = VariableMap(pdesys, discretization)
    # Check for basic interface errors
    PDEBase.interface_errors(pdesys, v, discretization)
    # Extract tspan
    tspan = t !== nothing ? v.intervals[t] : nothing
    # Find the derivative orders in the bcs
    bcorders = Dict(map(x -> x => d_orders(x, get_bcs(pdesys)), all_ivs(v)))
    # Create a map of each variable to their boundary conditions including initial conditions
    boundarymap = PDEBase.parse_bcs(get_bcs(pdesys), v, bcorders)
    # Check that the boundary map is valid
    PDEBase.check_boundarymap(boundarymap, discretization)

    # Transform system so that it is compatible with the discretization
    if should_transform(pdesys, discretization, boundarymap)
        pdesys = PDEBase.transform_pde_system!(v, boundarymap, pdesys, discretization)
    end

    pdeeqs = get_eqs(pdesys)
    bcs = get_bcs(pdesys)

    ############################
    # Discretization of system
    ############################
    disc_state = PDEBase.construct_disc_state(discretization)

    # Create discretized space and variables, this is called `s` throughout
    s = PDEBase.construct_discrete_space(v, discretization)

    return Dict(
        vcat(
            [Num(x) => s.grid[x] for x in s.x̄], [Num(u) => s.discvars[u] for u in s.ū]
        )
    )
end

function ODEFunctionExpr(
        pdesys::PDESystem, discretization::MethodOfLines.MOLFiniteDifference
    )
    sys, tspan = SciMLBase.symbolic_discretize(pdesys, discretization)
    return try
        if tspan === nothing
            @assert true "Codegen for NonlinearSystems is not yet implemented."
        else
            simpsys = mtkcompile(sys)
            return ODEFunction(simpsys; expression = Val{true})
        end
    catch e
        println("The system of equations is:")
        println(get_eqs(sys))
        println()
        println("Discretization failed, please post an issue on https://github.com/SciML/MethodOfLines.jl with the failing code and system at low point count.")
        println()
        rethrow(e)
    end
end

function SciMLBase.ODEFunction(
        pdesys::PDESystem, discretization::MethodOfLines.MOLFiniteDifference;
        analytic = nothing, kwargs...
    )
    sys, tspan = SciMLBase.symbolic_discretize(pdesys, discretization)
    return try
        if tspan === nothing
            @assert true "Codegen for NonlinearSystems is not yet implemented."
        else
            simpsys = mtkcompile(sys)
            if analytic !== nothing
                analytic = analytic isa Dict ? analytic : Dict(analytic)
                s = getmetadata(sys, ModelingToolkit.ProblemTypeCtx, nothing).discretespace
                us = get_unknowns(simpsys)
                gridlocs = get_gridloc.(us, (s,))
                f_analytic = generate_function_from_gridlocs(analytic, gridlocs, s)
            end
            return ODEFunction(
                simpsys; analytic = f_analytic, eval_module = @__MODULE__,
                discretization.kwargs..., kwargs...
            )
        end
    catch e
        println("The system of equations is:")
        println(get_eqs(sys))
        println()
        println("Discretization failed, please post an issue on https://github.com/SciML/MethodOfLines.jl with the failing code and system at low point count.")
        println()
        rethrow(e)
    end
end

function generate_code(
        pdesys::PDESystem, discretization::MethodOfLines.MOLFiniteDifference,
        filename = "generated_code_of_pdesys.jl"
    )
    code = ODEFunctionExpr(pdesys, discretization)
    rm(filename; force = true)
    return open(filename, "a") do io
        println(io, code)
    end
end

# Override generate_system to include unit correction parameters for spatial variables
# with units. These are symbolic constants (value 1.0 with spatial unit) needed to
# give finite difference stencil coefficients the correct dimensional units.
function PDEBase.generate_system(
        disc_state::PDEBase.EquationState, s::DiscreteSpace, u0, tspan,
        metadata::MOLMetadata, disc::MOLFiniteDifference
    )
    discvars = PDEBase.get_discvars(s)
    t = PDEBase.get_time(disc)
    name = getfield(metadata.pdesys, :name)
    pdesys = metadata.pdesys
    alleqs = vcat(disc_state.eqs, unique(disc_state.bceqs))
    alldepvarsdisc = vec(reduce(vcat, vec(unique(reduce(vcat, vec.(values(discvars)))))))

    defaults = merge(ModelingToolkit.defaults(pdesys), Dict(u0))
    ps_raw = ModelingToolkit.get_ps(pdesys)
    ps_raw = ps_raw === nothing || ps_raw === SciMLBase.NullParameters() ? Num[] : ps_raw
    # Separate Pair objects (parameter => value) from plain symbolic parameters
    ps = Num[]
    for p in ps_raw
        if p isa Pair
            push!(ps, p.first)
            defaults[p.first] = p.second
        else
            push!(ps, p)
        end
    end

    # Add unit correction constants for spatial variables with units
    for x in s.x̄
        uc = make_unit_constant(x)
        if uc !== nothing
            ps = vcat(ps, [uc])
        end
    end

    checks = true
    try
        if t === nothing
            eqs = map(eq -> 0 ~ eq.rhs - eq.lhs, alleqs)
            sys = System(
                eqs, alldepvarsdisc, ps, defaults = defaults, name = name,
                metadata = [PDEBase.ProblemTypeCtx => metadata], checks = checks
            )
            return sys, nothing
        else
            sys = System(
                alleqs, t, alldepvarsdisc, ps, defaults = defaults, name = name,
                metadata = [PDEBase.ProblemTypeCtx => metadata], checks = checks
            )
            return sys, tspan
        end
    catch e
        println("The system of equations is:")
        println()
        println("Discretization failed, please post an issue on https://github.com/SciML/MethodOfLines.jl with the failing code and system at low point count.")
        println()
        rethrow(e)
    end
end
