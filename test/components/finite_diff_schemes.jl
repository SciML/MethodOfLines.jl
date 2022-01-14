using ModelingToolkit, MethodOfLines, DomainSets, Test

@parameters x, t 
@variables u(..)

indvars = [x, t]
nottime = [x]
depvars = [u]

Dx(d) = Differential(x)^d
Dt = Differential(t)

t_min= 0.
t_max = 2.0
x_min = 0.
x_max = 20.0

dx = 1.0
order = 2

domains = [t ∈ Interval(t_min, t_max), x ∈ Interval(x_min, x_max)]


@testset "Test 01: Cartesian derivative discretization" begin
    weights = []
    push!(weights, ([-0.5,0,0.5], [1.,-2.,1.], [-1/2,1.,0.,-1.,1/2]))
    push!(weights, ([1/12, -2/3,0,2/3,-1/12], [-1/12,4/3,-5/2,4/3,-1/12], [1/8,-1.,13/8,0.,-13/8,1.,-1/8]))
    for d in 1:3
        for a in [2,4]
            pde = Dt(u) ~ Dx(d)(u)
            # Test centered order
            disc = MOLFiniteDifference([x=>dx], t; centered_order=order)

            s = MethodOfLines.DiscreteSpace(domains, depvars, indvars, nottime, disc)

            derivweights = MethodOfLines.DifferentialDiscretizer(pde, s, disc)
            
            II = s.Igrid[10]

            rules = MethodOfLines.generate_finite_difference_rules(II, s, pde, derivweights)
            @show rules
            disc_pde=substitute(pde.lhs,rules) ~ substitute(pde.rhs,rules)
            @show disc_pde
        end
    end
end