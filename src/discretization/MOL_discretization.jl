# Method of lines discretization scheme

function SciMLBase.symbolic_discretize(pdesys::PDESystem, discretization::MethodOfLines.MOLFiniteDifference{G}) where G
    pdeeqs = pdesys.eqs isa Vector ? pdesys.eqs : [pdesys.eqs]
    bcs = pdesys.bcs
    domain = pdesys.domain
    depvars = pdesys.dvs
    indvars = pdesys.ivs
    
    t = discretization.time
    # Get tspan
    tspan = nothing
    # Check that inputs make sense
    if t !== nothing
        tdomain = pdesys.domain[findfirst(d->isequal(t.val, d.variables), pdesys.domain)]
        @assert tdomain.domain isa DomainSets.Interval
        tspan = (DomainSets.infimum(tdomain.domain), DomainSets.supremum(tdomain.domain))
    end

    indvars = filter(x-> t === nothing || !isequal(x, t.val), indvars)
    
    alleqs = []
    bceqs = []
    # Create discretized space and variables
    s = DiscreteSpace(domain, depvars, indvars, discretization)
    # Generate finite difference weights
    derivweights = DifferentialDiscretizer(pdesys, pdesys.bcs, s, discretization)
    # Create a map of each variable to their boundary conditions
    boundarymap, u0 = BoundaryHandler(pdesys.bcs, s, depvar_ops, tspan, derivweights)

    interiormap = InteriorMap(pdeeqs, boundarymap, s)

    # Loop over equations: different space, grid, independent variables etc for each equation
    # a slightly more efficient approach would be to group equations that have the same
    # independent variables
    for pde in pdeeqs
        # Read the dependent variables on both sides of the equation
        depvars_lhs = get_depvars(pde.lhs, depvar_ops)
        depvars_rhs = get_depvars(pde.rhs, depvar_ops)
        depvars = collect(depvars_lhs ∪ depvars_rhs)
        
        # Read the independent variables,JZ346595D
        # ignore if the only argument is [t]
        allindvars = Set(filter(xs->!isequal(xs, [t]), map(arguments, depvars)))
        allx̄ = Set(filter(!isempty, map(u->filter(x-> t === nothing || !isequal(x, t.val), arguments(u)), depvars)))
        if isempty(allx̄)
            push!(alleqs, pde)
            push!(alldepvarsdisc, depvars)
        else
            # make sure there is only one set of independent variables per equation
            @assert length(allx̄) == 1
            @assert length(allindvars) == 1
            
            # Generate the boundary conditions for the correct variable
            for boundary in boundarymap[operation(interiormap.var[pde])]
                generate_bc_rules!(bceqs, derivweights, s, interiormap, boundary)
            end
            
            interior = interiormap.I[pde]
            
            # Set invalid corner points to zero
            generate_corner_eqs!(bceqs, s, interior, interiormap.var[pde])

            pdeeqs = vec(map(interior) do II
                rules = vcat(generate_finite_difference_rules(II, s, pde, derivweights), valmaps(s, II))
                substitute(pde.lhs,rules) ~ substitute(pde.rhs,rules)
            end)
            
            push!(alleqs,pdeeqs)
        end
    end
    
    u0 = !isempty(u0) ? reduce(vcat, u0) : u0
    bceqs = reduce(vcat, bceqs)
    alleqs = reduce(vcat, alleqs)
    alldepvarsdisc = reduce(vcat, values(s.discvars))
    
    # Finalize
    defaults = pdesys.ps === nothing || pdesys.ps === SciMLBase.NullParameters() ? u0 : vcat(u0,pdesys.ps)
    ps = pdesys.ps === nothing || pdesys.ps === SciMLBase.NullParameters() ? Num[] : first.(pdesys.ps)
    # Combine PDE equations and BC equations
    if t === nothing
        # At the time of writing, NonlinearProblems require that the system of equations be in this form:
        # 0 ~ ...
        # Thus, before creating a NonlinearSystem we normalize the equations s.t. the lhs is zero.
        eqs = map(eq -> 0 ~ eq.rhs - eq.lhs, vcat(alleqs, unique(bceqs)))
        sys = NonlinearSystem(eqs, vec(reduce(vcat, vec(alldepvarsdisc))), ps, defaults=Dict(defaults),name=pdesys.name)
        return sys, nothing
    else
        # * In the end we have reduced the problem to a system of equations in terms of Dt that can be solved by the `solve` method.
        println(vcat(alleqs, unique(bceqs)))
         println(Dict(defaults))#
        # println(vec(reduce(vcat, vec(alldepvarsdisc))))
        # println(ps)
        # println(tspan)#
        # println(typeof.(vcat(alleqs, unique(bceqs))))
        sys = ODESystem(vcat(alleqs, unique(bceqs)), t, vec(reduce(vcat, vec(alldepvarsdisc))), ps, defaults=Dict(defaults), name=pdesys.name)
        return sys, tspan
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