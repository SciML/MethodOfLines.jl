using ModelingToolkit, MethodOfLines, DomainSets, Test, Symbolics, SymbolicUtils, LinearAlgebra

@parameters x, t 
@variables u(..)

Dx(d) = Differential(x)^d
Dt = Differential(t)

t_min= 0.
t_max = 2.0
x_min = 0.
x_max = 20.0 

dx = 1.0

domains = [t ∈ Interval(t_min, t_max), x ∈ Interval(x_min, x_max)]


@testset "Test 01: Cartesian derivative discretization" begin
    weights = []
    push!(weights, ([-0.5,0,0.5], [1.,-2.,1.], [-1/2,1.,0.,-1.,1/2]))
    push!(weights, ([1/12, -2/3,0,2/3,-1/12], [-1/12,4/3,-5/2,4/3,-1/12], [1/8,-1.,13/8,0.,-13/8,1.,-1/8]))
    for d in 1:3
        for (i,a) in enumerate([2,4])
            pde = Dt(u(t,x)) ~ Dx(d)(u(t,x))
            bcs = [u(0,x) ~ cos(x), u(t,0) ~ exp(-t), u(t,Float64(π)) ~ -exp(-t)]

            @named pdesys = PDESystem(pde,bcs,domains,[t,x],[u(t,x)])

            # Test centered order 
            disc = MOLFiniteDifference([x=>dx], t; approx_order=a)

            depvar_ops = map(x->operation(x.val),pdesys.depvars)

            depvars_lhs = MethodOfLines.get_depvars(pde.lhs, depvar_ops)
            depvars_rhs = MethodOfLines.get_depvars(pde.rhs, depvar_ops)
            depvars = collect(depvars_lhs ∪ depvars_rhs)
            # Read the independent variables,
            # ignore if the only argument is [t]
            indvars = first(Set(filter(xs->!isequal(xs, [t]), map(arguments, depvars))))
            x̄ = first(Set(filter(!isempty, map(u->filter(x-> t === nothing || !isequal(x, t.val), arguments(u)), depvars))))

            s = MethodOfLines.DiscreteSpace(domains, depvars, indvars, x̄, disc)

            derivweights = MethodOfLines.DifferentialDiscretizer(pde, s, disc)
            
            II = s.Igrid[10]

            I1 = MethodOfLines.unitindices(1)[1]

            rules = MethodOfLines.generate_finite_difference_rules(II, s, pde, derivweights)

            disc_pde=substitute(pde.lhs,rules) ~ substitute(pde.rhs,rules)

            @test disc_pde.rhs == dot(weights[i][d], s.discvars[depvars[1]][[II + j*I1 for j in MethodOfLines.half_range(length(weights[i][d]))]])
        end
    end
end