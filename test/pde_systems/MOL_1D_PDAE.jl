# 1D diffusion problem

# Packages and inclusions
using ModelingToolkit,MethodOfLines,LinearAlgebra,Test,OrdinaryDiffEq, DomainSets
using ModelingToolkit: Differential

# Tests
@testset "Dt(u(t,x)) ~ Dxx(u(t,x)), 0 ~ Dxx(v(t,x)) + sin(x), Dirichlet BCs" begin
    # Method of Manufactured Solutions
    u_exact = (x,t) -> exp.(-t) * cos.(x)
    v_exact = (x,t) -> exp.(-t) * sin.(x)

    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..) v(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Dx^2

    # 1D PDE and boundary conditions
    eqs = [Dt(u(t,x)) ~ Dxx(u(t,x)),
           0 ~ Dxx(v(t,x)) + exp(-t)*sin(x)]
    bcs = [u(0,x) ~ cos(x),
           v(0,x) ~ sin(x),
           u(t,0) ~ exp(-t),
           Dx(u(t,1)) ~ -exp(-t) * sin(1),
           Dx(v(t,0)) ~ exp(-t),
           v(t,1) ~ exp(-t) * sin(1)]

    # Space and time domains
    domains = [t ∈ Interval(0.0,1.0),
               x ∈ Interval(0.0,1.0)]

    # PDE system
    @named pdesys = PDESystem(eqs,bcs,domains,[t,x],[u(t,x),v(t,x)])

    # Method of lines discretization
    l = 20
    dx = range(0.0,1.0,length=l)
    dx_ = dx[2] - dx[1]
    order = 2
    discretization = MOLFiniteDifference([x=>dx_],t)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys,discretization)
    # Solve ODE problem
    sol = solve(prob,Rodas4(),saveat=0.1)

    x_sol = sol[x]
    t_sol = sol[t]

    solu = sol[u(t,x)]
    solv = sol[v(t,x)]

    # Test against exact solution
    for i in 1:length(t_sol)
        @test all(isapprox.(u_exact(x_sol, t_sol[i]), solu[i, :], atol=0.01))
        @test all(isapprox.(v_exact(x_sol, t_sol[i]), solv[i, :], atol=0.01))
    end
end
