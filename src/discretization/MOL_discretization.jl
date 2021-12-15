# Method of lines discretization scheme

@enum GridAlign center_align edge_align

struct MOLFiniteDifference{T,T2} <: DiffEqBase.AbstractDiscretization
    dxs::T
    time::T2
    upwind_order::Int
    centered_order::Int
    grid_align::GridAlign
end

# Constructors. If no order is specified, both upwind and centered differences will be 2nd order
MOLFiniteDifference(dxs, time=nothing; upwind_order = 1, centered_order = 2, grid_align=center_align) =
    MOLFiniteDifference(dxs, time, upwind_order, centered_order, grid_align)

function SciMLBase.symbolic_discretize(pdesys::PDESystem, discretization::MethodOfLines.MOLFiniteDifference)
    pdeeqs = pdesys.eqs isa Vector ? pdesys.eqs : [pdesys.eqs]
    bcs = pdesys.bcs
    domain = pdesys.domain

    grid_align = discretization.grid_align
    t = discretization.time
    # Get tspan
    tspan = nothing
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

            s = DiscreteSpace(pdesys.domain, depvars, indvars, nottime, grid_align, discretization)

            #---- Count Boundary Equations --------------------
            # Count the number of boundary equations that lie at the spatial boundary on
            # both the left and right side. This will be used to determine number of
            # interior equations s.t. we have a balanced system of equations.
            
            # get the depvar boundary terms for given depvar and indvar index.
            get_depvarbcs(depvar, i) = substitute.((depvar,),get_edgevals(s, i))

            # return the counts of the boundary-conditions that reference the "left" and
            # "right" edges of the given independent variable. Note that we return the
            # max of the count for each depvar in the system of equations.
            @inline function get_bc_counts(i)
                left = 0
                right = 0
                for depvar in s.vars
                    depvaredges = get_depvarbcs(depvar, i)
                    counts = [map(x->occursin(x, bc.lhs), depvaredges) for bc in pdesys.bcs]
                    left = max(left, sum([c[1] for c in counts]))
                    right = max(right, sum([c[2] for c in counts]))
                end
                return [left, right]
            end
            #--------------------------------------------------
            # * The stencil is the tappoints of the finite difference operator, relative to the current index
            stencil(j, order) = CartesianIndices(Tuple(map(x -> -x:x, (1:nspace.==j) * (order÷2))))
            
            # TODO: Generalize central difference handling to allow higher even order derivatives
            # The central neighbour indices should add the stencil to II, unless II is too close
            # to an edge in which case we need to shift away from the edge
            
            
            # Calculate buffers
            #TODO: Update this when allowing higher order derivatives to correctly deal with boundary upwinding
            interior = s.Igrid[[let bcs = get_bc_counts(i)
                                    (1 + first(bcs)):length(g)-last(bcs)
                                    end
                                    for (i,g) in enumerate(s.grid)]...]

            ### PDE EQUATIONS ###
            # Create a stencil in the required dimension centered around 0
            # e.g. (-1,0,1) for 2nd order, (-2,-1,0,1,2) for 4th order, etc
            if discretization.centered_order % 2 != 0
                throw(ArgumentError("Discretization centered_order must be even, given $(discretization.centered_order)"))
            end
            # For all cartesian indices in the interior, generate finite difference rules
            eqs = vec(map(interior) do II
                rules = generate_finite_difference_rules(II, s, pde, discretization)
                substitute(pde.lhs,rules) ~ substitute(pde.rhs,rules)
            end)
            
            generate_u0_and_bceqs!!(u0, bceqs, pdesys.bcs, s, depvar_ops, tspan, discretization)
            push!(alleqs,eqs)
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