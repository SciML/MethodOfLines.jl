using MethodOfLines, Test
using ModelingToolkit


@testset "Discretization of variables, center aligned, uniform grid" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    # 1D PDE and boundary conditions
    eq  = Dt(u(t,x)) ~ Dxx(u(t,x))
    bcs = [u(0,x) ~ cos(x),
           u(t,0) ~ exp(-t),
           u(t,Float64(π)) ~ -exp(-t)]

    # Space and time domains
    domains = [t ∈ Interval(0.0,1.0),
               x ∈ Interval(0.0,Float64(π))]

    # PDE system
    @named pdesys = PDESystem(eq,bcs,domains,[t,x],[u(t,x)])

    # Method of lines discretization
    dx = Float64(π/30)
    order = 2
    
    discretization = MOLFiniteDifference([x=>dx],t)

    # Prepare symbolic variables
    pdeeqs = pdesys.eqs
    bcs = pdesys.bcs
    domain = pdesys.domain

    grid_align = discretization.grid_align
    t = discretization.time
    # Get tspan
    tspan = nothing
    if t !== nothing
        tdomain = domain[findfirst(d->isequal(t.val, d.variables), domain)]
        @assert tdomain.domain isa DomainSets.Interval
        tspan = (DomainSets.infimum(tdomain.domain), DomainSets.supremum(tdomain.domain))
    end

    depvar_ops = map(x->operation(x.val),pdesys.depvars)

    pde = first(pdeeqs)

    # Read the dependent variables on both sides of the equation
    depvars_lhs = get_depvars(pde.lhs, depvar_ops)
    depvars_rhs = get_depvars(pde.rhs, depvar_ops)
    depvars = collect(depvars_lhs ∪ depvars_rhs)
    # Read the independent variables,
    # ignore if the only argument is [t]
    allindvars = Set(filter(xs->!isequal(xs, [t]), map(arguments, depvars)))
    allnottime = Set(filter(!isempty, map(u->filter(x-> t === nothing || !isequal(x, t.val), arguments(u)), depvars)))

    # make sure there is only one set of independent variables per equation
    @assert length(allnottime) == 1
    nottime = first(allnottime)
    @assert length(allindvars) == 1
    indvars = first(allindvars)
    nspace = length(nottime)

    # Discretize space

    s = DiscreteSpace(domain, depvars, indvars, nottime, grid_align, discretization)
    
    # Calculate grid points for test
    disc_x = 0.0:dx:Float64(π)

    @test all(s.nottime[1] .== disc_x)
end