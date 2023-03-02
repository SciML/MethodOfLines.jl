using ModelingToolkit, MethodOfLines, LinearAlgebra, Test, OrdinaryDiffEq, DomainSets

# Parameters, variables, and derivatives
@parameters t x1 x2
@variables c1(..)
@variables c2(..)
Dt = Differential(t)

Dx1 = Differential(x1)
Dxx1 = Dx1^2

Dx2 = Differential(x2)
Dxx2 = Dx2^2

D1(c) = 1 + c/10
D2(c) = 1/10 + c/10

eqs = [Dt(c1(t, x1)) ~ Dx1(D1(c1(t, x1))*Dx1(c1(t, x1))),
       Dt(c2(t, x2)) ~ Dx2(D2(c2(t, x2))*Dx2(c2(t, x2)))]

bcs = [c1(0, x1) ~ -x1 * (x1 - 1) * sin(x1),
         c2(0, x2) ~ -x2 * (x2 - 1) * sin(x2),
         c1(t, 0) ~ 0,
         c1(t, 0.5) ~ c2(t, 0.5),
        -D1(c1(t, 0.5))*Dx1(c1(t, 0.5)) ~ -D2(c2(t, 0.5))*Dx2(c2(t, 0.5)),
         c2(t, 1) ~ 0]

domains = [t ∈ Interval(0.0, 1.0),
           x1 ∈ Interval(0.0, 0.5),
           x2 ∈ Interval(0.5, 1.0)]

@named pdesys = PDESystem(eqs, bcs, domains,
                          [t, x1, x2], [c1(t, x1), c2(t, x2)])

l = 10

discretization = MOLFiniteDifference([x1 => l, x2 => l], t)

prob = discretize(pdesys, discretization)

sol = solve(prob, Tsit5(), saveat=0.1)

x1_sol = sol[x1]
x2_sol = sol[x2]
t_sol = sol[t]
solc1 = sol[c1(t, x1)]
solc2 = sol[c2(t, x2)]

solc = vcat(solc1[end, :], solc2[end, 2:end])
