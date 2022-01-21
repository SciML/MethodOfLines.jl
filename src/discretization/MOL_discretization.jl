# Method of lines discretization scheme

function SciMLBase.symbolic_discretize(pdesys::PDESystem, discretization::MethodOfLines.MOLFiniteDifference{G}) where G
    pdeeqs = pdesys.eqs isa Vector ? pdesys.eqs : [pdesys.eqs]
    bcs = pdesys.bcs
    domain = pdesys.domain
    
    grid_align = discretization.grid_align
    t = discretization.time
    # Get tspan
    tspan = nothing
    # Check that inputs make sense
    if t !== nothing
        tdomain = pdesys.domain[findfirst(d->isequal(t.val, d.variables), pdesys.domain)]
        @assert tdomain.domain isa DomainSets.Interval
        tspan = (DomainSets.infimum(tdomain.domain), DomainSets.supremum(tdomain.domain))
    end

    depvar_ops = map(x->operation(x.val),pdesys.depvars)

    u0 = []
    bceqs = []
    alleqs = []
    alldepvarsdisc = []
    # Loop over equations: different space, grid, independent variables etc for each equation
    # a slightly more efficient approach would be to group equations that have the same
    # independent variables
    for pde in pdeeqs
        # Read the dependent variables on both sides of the equation
        depvars_lhs = get_depvars(pde.lhs, depvar_ops)
        depvars_rhs = get_depvars(pde.rhs, depvar_ops)
        depvars = collect(depvars_lhs ∪ depvars_rhs)

        # Read the independent variables,
        # ignore if the only argument is [t]
        allindvars = Set(filter(xs->!isequal(xs, [t]), map(arguments, depvars)))
        allx̄ = Set(filter(!isempty, map(u->filter(x-> t === nothing || !isequal(x, t.val), arguments(u)), depvars)))
        if isempty(allx̄)
            push!(alleqs, pde)
            push!(alldepvarsdisc, depvars)
            for bc in bcs
                if any(u->isequal(bc.lhs, operation(u)(tspan[1])), depvars)
                    push!(u0, operation(bc.lhs)(t) => bc.rhs)
                end
            end
        else
            # make sure there is only one set of independent variables per equation
            @assert length(allx̄) == 1
            x̄ = first(allx̄)
            @assert length(allindvars) == 1
            indvars = first(allindvars)
            
            # Get the grid
            s = DiscreteSpace(domain, depvars, indvars, x̄, discretization)
            
            # Get the finite difference weights
            derivweights = DifferentialDiscretizer(pde, s, discretization)

            # Get the boundary conditions
            BoundaryHandler!!(u0, bceqs, pdesys.bcs, s, depvar_ops, tspan, derivweights)

            # Find the indexes on the interior
            interior = Iinterior(s)

            # Discretize the equation on the interior
            pdeeqs = vec(map(interior) do II
                rules = vcat(generate_finite_difference_rules(II, s, pde, derivweights), valmaps(s, II))
                substitute(pde.lhs,rules) ~ substitute(pde.rhs,rules)
            end)
            
            push!(alleqs,pdeeqs)
            push!(alldepvarsdisc, reduce(vcat, values(s.discvars)))
        end
    end
    
    u0 = !isempty(u0) ? reduce(vcat, u0) : u0
    bceqs = reduce(vcat, bceqs)
    alleqs = reduce(vcat, alleqs)
    alldepvarsdisc = unique(reduce(vcat, alldepvarsdisc))
    
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
        println(alleqs, bceqs)
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