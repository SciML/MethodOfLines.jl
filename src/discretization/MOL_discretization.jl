# Method of lines discretization scheme

function interface_errors(depvars, indvars, discretization)
    for x in indvars
        @assert findfirst(dxs -> isequal(x, dxs[1].val), discretization.dxs) !== nothing "Variable $x has no step size"
    end
end

function SciMLBase.symbolic_discretize(pdesys::PDESystem, discretization::MethodOfLines.MOLFiniteDifference{G}) where G
    pdeeqs = pdesys.eqs isa Vector ? pdesys.eqs : [pdesys.eqs]
    bcs = pdesys.bcs
    domain = pdesys.domain
    
    t = discretization.time

    depvar_ops = map(x->operation(x.val),pdesys.depvars)
    # Get all dependent variables in the correct type
    alldepvars = get_all_depvars(pdesys, depvar_ops)
    # Get all independent variables in the correct type, removing time from the list
    allindvars = remove(collect(reduce(union, filter(xs->!isequal(xs, [t]), map(arguments, alldepvars)))), t)
    
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
    # Generate finite difference weights
    derivweights = DifferentialDiscretizer(pdesys, s, discretization)
    # Create a map of each variable to their boundary conditions and get the initial condition
    boundarymap, u0 = BoundaryHandler(pdesys.bcs, s, depvar_ops, tspan, derivweights)
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
        
        # Read the independent variables
        # ignore if the only argument is [t]
        indvars = Set(filter(xs->!isequal(xs, [t]), map(arguments, depvars)))
        allx̄ = Set(filter(!isempty, map(u->filter(x-> t === nothing || !isequal(x, t.val), arguments(u)), depvars)))
        if isempty(allx̄) 
            rules = varmaps(s, depvars, CartesianIndex())
            
            push!(alleqs, substitute(pde.lhs, rules) ~ substitute(pde.rhs, rules))
        else
            # make sure there is only one set of independent variables per equation
            @assert length(allx̄) == 1
            pdex̄ = first(allx̄)
            @assert length(indvars) == 1
            
            interior = interiormap.I[pde]
            eqvar = interiormap.var[pde]
            # Generate the boundary conditions for the correct variable
            for boundary in boundarymap[operation(eqvar)]
                generate_bc_rules!(bceqs, derivweights, s, interiormap, boundary)
            end
            #@show interior
            
            # Set invalid corner points to zero
            generate_corner_eqs!(bceqs, s, interiormap, pde)

            pdeeqs = vec(map(interior) do II
                #@show II
                rules = vcat(generate_finite_difference_rules(II, s, depvars, pde, derivweights), valmaps(s, eqvar, depvars, II))
                substitute(pde.lhs,rules) ~ substitute(pde.rhs,rules)
            end)
            
            push!(alleqs,pdeeqs)
        end
    end
    
    u0 = !isempty(u0) ? reduce(vcat, u0) : u0
    bceqs = reduce(vcat, bceqs)
    alleqs = reduce(vcat, alleqs)
    alldepvarsdisc = unique(reduce(vcat, vec.(values(s.discvars))))
        
    # Finalize
    defaults = pdesys.ps === nothing || pdesys.ps === SciMLBase.NullParameters() ? u0 : vcat(u0,pdesys.ps)
    ps = pdesys.ps === nothing || pdesys.ps === SciMLBase.NullParameters() ? Num[] : first.(pdesys.ps)
    # Combine PDE equations and BC equations
    try 
        if t === nothing
            # At the time of writing, NonlinearProblems require that the system of equations be in this form:
            # 0 ~ ...
            # Thus, before creating a NonlinearSystem we normalize the equations s.t. the lhs is zero.
            eqs = map(eq -> 0 ~ eq.rhs - eq.lhs, vcat(alleqs, unique(bceqs)))
            #getfield.(alldepvarsdisc, [:val])
            sys = NonlinearSystem(eqs, vec(reduce(vcat, vec(alldepvarsdisc))), ps, defaults=Dict(defaults),name=pdesys.name)
            return sys, nothing
        else
            # * In the end we have reduced the problem to a system of equations in terms of Dt that can be solved by the `solve` method.
            #println(vcat(alleqs, unique(bceqs)))
            #println(Dict(defaults))#
            # println(vec(reduce(vcat, vec(alldepvarsdisc))))
            # println(ps)
            # println(tspan)#
            # println(typeof.(vcat(alleqs, unique(bceqs))))
            sys = ODESystem(vcat(alleqs, unique(bceqs)), t, vec(reduce(vcat, vec(alldepvarsdisc))), ps, defaults=Dict(defaults), name=pdesys.name)
            return sys, tspan
        end
    catch e
        println("The system of equations is:")
        println(vcat(alleqs, unique(bceqs)))
        println()
        println("The defaults are:")
        println(defaults)
        println()
        println("Discretization failed, please post an issue on https://github.com/SciML/MethodOfLines.jl with the failing code and system at low point count.")
        println()
        rethrow(e)
    end
end

function SciMLBase.discretize(pdesys::PDESystem,discretization::MethodOfLines.MOLFiniteDifference)
    sys, tspan = SciMLBase.symbolic_discretize(pdesys, discretization)
    if tspan === nothing
        return prob = NonlinearProblem(sys, ones(length(sys.states)))
    else
        simpsys = structural_simplify(sys)
        return prob = ODEProblem(simpsys,Pair[],tspan)
    end
end