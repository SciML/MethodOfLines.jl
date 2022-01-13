using ModelingToolkit, MethodOfLines, DomainSets, Test

@parameters x, t 
@variables u(..)

indvars = [x, t]
nottime = [x]
depvars = [u]

Dx = Differential(x)
Dt = Differential(t)

t_min= 0.
t_max = 2.0
x_min = 0.
x_max = 2.

dx = 0.1
order = 2

domains = [t ∈ Interval(t_min, t_max), x ∈ Interval(x_min, x_max)]


@testset "Test 01: discretization of variables, center aligned grid" begin
    pde = Dt(u) ~ Dx(u)
    # Test centered order
    disc = MOLFiniteDifference([x=>dx], t; centered_order=order)

    s = MethodOfLines.DiscreteSpace(domains, depvars, indvars, nottime, disc)

    derivweights = MethodOfLines.DifferentialDiscretizer(pde, s, disc)
    
    II = s.Igrid[10]

    rules = MethodOfLines.generate_finite_difference_rules(II, s, pde, derivweights)
    disc_pde=substitute(pde.lhs,rules) ~ substitute(pde.rhs,rules)
end