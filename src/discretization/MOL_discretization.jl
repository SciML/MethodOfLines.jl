# Method of lines discretization scheme

function interface_errors(depvars, indvars, discretization)
    for x in indvars
        @assert findfirst(dxs -> isequal(x, dxs[1].val), discretization.dxs) !== nothing "Variable $x has no step size"
    end
end

function SciMLBase.symbolic_discretize(
    pdesys::PDESystem,
    discretization::MethodOfLines.MOLFiniteDifference{G},
) where {G}
    pdeeqs = [eq.lhs - eq.rhs ~ 0 for eq in pdesys.eqs]
    bcs = pdesys.bcs
    domain = pdesys.domain

    t = discretization.time

    depvar_ops = map(x -> operation(x.val), pdesys.depvars)
    # Get all dependent variables in the correct type
    alldepvars = get_all_depvars(pdesys, depvar_ops)
    alldepvars = filter(u -> !any(map(x -> x isa Number, arguments(u))), alldepvars)
    # Get all independent variables in the correct type, removing time from the list
    allindvars = remove(
        collect(
            filter(
                x -> !(x isa Number),
                reduce(
                    union,
                    filter(xs -> (!isequal(xs, [t])), map(arguments, alldepvars)),
                ),
            ),
        ),
        t,
    )
    #@show allindvars, typeof.(allindvars)

    interface_errors(alldepvars, allindvars, discretization)
    # @show alldepvars
    # @show allindvars

    # Get tspan
    tspan = nothing
    # Check that inputs make sense
    if t !== nothing
        tdomain = pdesys.domain[findfirst(d -> isequal(t.val, d.variables), pdesys.domain)]
        @assert tdomain.domain isa DomainSets.Interval
        tspan = (DomainSets.infimum(tdomain.domain), DomainSets.supremum(tdomain.domain))
    end
    alleqs = []
    bceqs = []
    # * We wamt to do this in 2 passes
    # * First parse the system and BCs, replacing with DiscreteVariables and DiscreteDerivatives
    # * periodic parameters get type info on whether they are periodic or not, and if they join up to any other parameters
    # * Then we can do the actual discretization by recursively indexing in to the DiscreteVariables

    # Create discretized space and variables
    s = DiscreteSpace(domain, alldepvars, allindvars, discretization)
    # Generate finite difference weights
    derivweights = DifferentialDiscretizer(pdesys, s, discretization)
    # Create a map of each variable to their boundary conditions and get the initial condition
    boundarymap, u0 = BoundaryHandler(pdesys.bcs, s, depvar_ops, tspan, derivweights)
    # Generate a map of each variable to whether it is periodic in a given direction
    pmap = PeriodicMap(boundarymap, s)
    # Get the interior and variable to solve for each equation
    interiormap = InteriorMap(pdeeqs, boundarymap, s)

    # Loop over equations: different space, grid, independent variables etc for each equation
    # a slightly more efficient approach would be to group equations that have the same
    # independent variables
    for pde in pdeeqs
        # Read the dependent variables on both sides of the equation
        depvars_lhs = get_depvars(pde.lhs, depvar_ops)
        depvars_rhs = get_depvars(pde.rhs, depvar_ops)
        depvars = collect(depvars_lhs ∪ depvars_rhs)
        depvars = filter(u -> !any(map(x -> x isa Number, arguments(u))), depvars)

        # Read the independent variables
        # ignore if the only argument is [t]
        indvars = Set(filter(xs -> !isequal(xs, [t]), map(arguments, depvars)))
        # get all parameters in the equation
        allx̄ = Set(
            filter(
                !isempty,
                map(
                    u -> filter(x -> t === nothing || !isequal(x, t.val), arguments(u)),
                    depvars,
                ),
            ),
        )
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

            args = params(eqvar, s)
            indexmap = Dict([args[i] => i for i = 1:length(args)])

            # Handle boundary values appearing in the equation by creating functions that map each point on the interior to the correct replacement rule
            # Generate replacement rule gen closures for the boundary values like u(t, 1)
            boundaryvalfuncs =
                generate_boundary_val_funcs(s, depvars, boundarymap, indexmap, derivweights)

            # Generate the boundary conditions for the correct variable
            for boundary in reduce(vcat, collect(values(boundarymap[operation(eqvar)])))
                generate_bc_eqs!(bceqs, s, boundaryvalfuncs, interiormap, boundary)
            end

            # Set invalid corner points to zero
            generate_corner_eqs!(bceqs, s, interiormap, pde)

            # Generate the equations for the interior points
            pdeeqs = discretize_equation(
                pde,
                interiormap.I[pde],
                eqvar,
                depvars,
                s,
                derivweights,
                indexmap,
                boundaryvalfuncs,
                pmap,
            )

            push!(alleqs, pdeeqs)
        end
    end

    u0 = !isempty(u0) ? reduce(vcat, u0) : u0
    bceqs = reduce(vcat, bceqs)
    alleqs = reduce(vcat, alleqs)
    alldepvarsdisc = unique(reduce(vcat, vec.(values(s.discvars))))

    # Finalize
    defaults =
        pdesys.ps === nothing || pdesys.ps === SciMLBase.NullParameters() ? u0 :
        vcat(u0, pdesys.ps)
    ps =
        pdesys.ps === nothing || pdesys.ps === SciMLBase.NullParameters() ? Num[] :
        first.(pdesys.ps)
    # Combine PDE equations and BC equations
    try
        if t === nothing
            # At the time of writing, NonlinearProblems require that the system of equations be in this form:
            # 0 ~ ...
            # Thus, before creating a NonlinearSystem we normalize the equations s.t. the lhs is zero.
            eqs = map(eq -> 0 ~ eq.rhs - eq.lhs, vcat(alleqs, unique(bceqs)))
            sys = NonlinearSystem(
                eqs,
                vec(reduce(vcat, vec(alldepvarsdisc))),
                ps,
                defaults = Dict(defaults),
                name = pdesys.name,
            )
            return sys, nothing
        else
            # * In the end we have reduced the problem to a system of equations in terms of Dt that can be solved by an ODE solver.

            sys = ODESystem(
                vcat(alleqs, unique(bceqs)),
                t,
                vec(reduce(vcat, vec(alldepvarsdisc))),
                ps,
                defaults = Dict(defaults),
                name = pdesys.name,
            )
            return sys, tspan
        end
    catch e
        println("The system of equations is:")
        println(vcat(alleqs, unique(bceqs)))
        println()
        println(
            "Discretization failed, please post an issue on https://github.com/SciML/MethodOfLines.jl with the failing code and system at low point count.",
        )
        println()
        rethrow(e)
    end
end

function discretize_equation(
    pde,
    interior,
    eqvar,
    depvars,
    s,
    derivweights,
    indexmap,
    boundaryvalfuncs,
    pmap::PeriodicMap{hasperiodic},
) where {hasperiodic}
    return vec(
        map(interior) do II
            boundaryrules = mapreduce(f -> f(II), vcat, boundaryvalfuncs)
            rules = vcat(
                generate_finite_difference_rules(
                    II,
                    s,
                    depvars,
                    pde,
                    derivweights,
                    pmap,
                    indexmap,
                ),
                boundaryrules,
                valmaps(s, eqvar, depvars, II, indexmap),
            )
            substitute(pde.lhs, rules) ~ substitute(pde.rhs, rules)
        end,
    )
end

function SciMLBase.discretize(
    pdesys::PDESystem,
    discretization::MethodOfLines.MOLFiniteDifference,
)
    sys, tspan = SciMLBase.symbolic_discretize(pdesys, discretization)
    try
        simpsys = structural_simplify(sys)
        if tspan === nothing
            return prob = NonlinearProblem(
                simpsys,
                ones(length(simpsys.states));
                discretization.kwargs...,
            )
        else
            return prob = ODEProblem(simpsys, Pair[], tspan; discretization.kwargs...)
        end
    catch e
        error_analysis(sys, e)
    end
end

function get_discrete(pdesys, discretization)
    domain = pdesys.domain

    t = discretization.time

    depvar_ops = map(x -> operation(x.val), pdesys.depvars)
    # Get all dependent variables in the correct type
    alldepvars = get_all_depvars(pdesys, depvar_ops)
    alldepvars = filter(u -> !any(map(x -> x isa Number, arguments(u))), alldepvars)
    # Get all independent variables in the correct type, removing time from the list
    allindvars = remove(
        collect(
            filter(
                x -> !(x isa Number),
                reduce(
                    union,
                    filter(xs -> (!isequal(xs, [t])), map(arguments, alldepvars)),
                ),
            ),
        ),
        t,
    )
    #@show allindvars, typeof.(allindvars)

    interface_errors(alldepvars, allindvars, discretization)
    # @show alldepvars
    # @show allindvars

    # Get tspan
    tspan = nothing
    # Check that inputs make sense
    if t !== nothing
        tdomain = pdesys.domain[findfirst(d -> isequal(t.val, d.variables), pdesys.domain)]
        @assert tdomain.domain isa DomainSets.Interval
        tspan = (DomainSets.infimum(tdomain.domain), DomainSets.supremum(tdomain.domain))
    end
    alleqs = []
    bceqs = []

    # Create discretized space and variables
    s = DiscreteSpace(domain, alldepvars, allindvars, discretization)

    return Dict(
        vcat([Num(x) => s.grid[x] for x in s.x̄], [Num(u) => s.discvars[u] for u in s.ū]),
    )
end

function ModelingToolkit.ODEFunctionExpr(
    pdesys::PDESystem,
    discretization::MethodOfLines.MOLFiniteDifference,
)
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
        println(
            "Discretization failed, please post an issue on https://github.com/SciML/MethodOfLines.jl with the failing code and system at low point count.",
        )
        println()
        rethrow(e)
    end
end

function generate_code(
    pdesys::PDESystem,
    discretization::MethodOfLines.MOLFiniteDifference,
    filename = "code.jl",
)
    code = ODEFunctionExpr(pdesys, discretization)
    rm(filename)
    open(filename, "a") do io
        println(io, code)
    end
end
