# Method of lines discretization scheme

@enum GridAlign center_align edge_align

struct MOLFiniteDifference{T,T2} <: DiffEqBase.AbstractDiscretization
    dxs::T
    time::T2
    approx_order::Int
    grid_align::GridAlign
end

# Constructors. If no order is specified, both upwind and centered differences will be 2nd order
function MOLFiniteDifference(dxs, time=nothing; upwind_order = 1, centered_order = 2, grid_align=center_align)
    
    if discretization.centered_order % 2 != 0
        throw(ArgumentError("Discretization centered_order must be even, given $(discretization.centered_order)"))
    end
    return MOLFiniteDifference(dxs, time, approx_order, grid_align)
end

function SciMLBase.symbolic_discretize(pdesys::PDESystem, discretization::MethodOfLines.MOLFiniteDifference)
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
        allnottime = Set(filter(!isempty, map(u->filter(x-> t === nothing || !isequal(x, t.val), arguments(u)), depvars)))
        if isempty(allnottime)
            push!(alleqs, pde)
            push!(alldepvarsdisc, depvars)
            for bc in bcs
                if any(u->isequal(bc.lhs, operation(u)(tspan[1])), depvars)
                    push!(u0, operation(bc.lhs)(t) => bc.rhs)
                end
            end
        else
            # make sure there is only one set of independent variables per equation
            @assert length(allnottime) == 1
            nottime = first(allnottime)
            @assert length(allindvars) == 1
            indvars = first(allindvars)
            nspace = length(nottime)

            s = DiscreteSpace(pdesys.domain, depvars, indvars, nottime, discretization)

            derivweights = DifferentialDiscretizer(pde, s, discretization)

            interior = s.Igrid[[2:(length(grid[x])-1) for x in s.nottime]]

            ### PDE EQUATIONS ###
            # Create a stencil in the required dimension centered around 0
            # e.g. (-1,0,1) for 2nd order, (-2,-1,0,1,2) for 4th order, etc
            # For all cartesian indices in the interior, generate finite difference rules
            pdeeqs = vec(map(interior) do II
                rules = generate_finite_difference_rules(II, s, pde, derivweights)
                substitute(pde.lhs,rules) ~ substitute(pde.rhs,rules)
            end)
            
            generate_u0_and_bceqs!!(u0, bceqs, pdesys.bcs, s, depvar_ops, tspan, discretization)
            push!(alleqs,pdeeqs)
            push!(alldepvarsdisc, reduce(vcat, s.discvars))
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