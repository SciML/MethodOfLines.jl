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

function _interface_physical_coords(b, grid1, grid2)
    if isupper(b)
        return grid1[end], grid2[1]
    else
        return grid1[1], grid2[end]
    end
end

function _interface_coords_aligned(coord1, coord2, grid1, grid2)
    scale = max(abs(grid1[end] - grid1[1]), abs(grid2[end] - grid2[1]))
    return isapprox(coord1, coord2; atol = sqrt(eps(float(one(scale)))) * scale)
end

function PDEBase.check_boundarymap(
        boundarymap, v::PDEBase.VariableMap, discretization::MOLFiniteDifference
    )
    return _check_interface_boundarymap(boundarymap, discretization)
end

# Kept for backwards compatibility with direct 2-arg calls; the PDEBase
# discretization pipeline invokes the 3-arg hook above.
function PDEBase.check_boundarymap(boundarymap, discretization::MOLFiniteDifference)
    return _check_interface_boundarymap(boundarymap, discretization)
end

function _check_interface_boundarymap(boundarymap, discretization::MOLFiniteDifference)
    bs = filter_interfaces(flatten_vardict(boundarymap))
    scheme = discretization.advection_scheme
    for b in bs
        dx1 = discretization.dxs[Num(b.x)]
        dx2 = discretization.dxs[Num(b.x2)]
        if dx1 isa AbstractVector && dx2 isa AbstractVector
            if scheme isa UpwindScheme && scheme.order > 1
                throw(ArgumentError("UpwindScheme(order=$(scheme.order)) is not yet supported with interface or periodic boundary conditions on nonuniform grids, please use the default first order `UpwindScheme()`."))
            end
            isequal(b.x, b.x2) && continue
            coord1, coord2 = _interface_physical_coords(b, dx1, dx2)
            if !_interface_coords_aligned(coord1, coord2, dx1, dx2)
                throw(
                    ArgumentError(
                        "The physical coordinates at the interface between $(b.x) and $(b.x2) must match for nonuniform grids, got $coord1 and $coord2 at the interface. Please ensure the grids align at the interface boundary. Note that cross-domain periodic (ring) topologies are not supported on nonuniform grids."
                    )
                )
            end
        elseif dx1 isa AbstractVector || dx2 isa AbstractVector
            throw(ArgumentError("The interface between $(b.x) and $(b.x2) mixes a scalar step size with a nonuniform grid vector, please supply the same kind of grid on both sides."))
        elseif dx1 != dx2
            throw(ArgumentError("The step size of the connected variables $(b.x) and $(b.x2) must be the same."))
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
    PDEBase.check_boundarymap(boundarymap, v, discretization)

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
