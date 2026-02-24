using ModelingToolkit, MethodOfLines, LinearAlgebra, Test, OrdinaryDiffEq, DomainSets
using ModelingToolkit: Differential
using DynamicQuantities


# Method of Manufactured Solutions
u_exact = (x, t) -> exp.(-t) * cos.(x)

# Parameters, variables, and derivatives
@parameters begin
    t, [unit = u"s"]
    x, [unit = u"m"]
    D_diff=1.0, [description = "Diffusion coefficient", unit = u"m^2/s"]
end   
@variables u(..) [unit = u"kg/m^3"]
Dt = Differential(t)
Dxx = Differential(x)^2

# 1D PDE and boundary conditions
eq = Dt(u(t, x)) ~ D_diff * Dxx(u(t, x))
bcs = [
    u(0, x) ~ cos(x),
    u(t, 0) ~ exp(-t),
    u(t, Float64(π)) ~ -exp(-t),
]

# Space and time domains
domains = [
    t ∈ Interval(0.0, 1.0),
    x ∈ Interval(0.0, Float64(π)),
]

# PDE system
@named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)], [D_diff])

# Method of lines discretization
dx = range(0.0, Float64(π), length = 30)
dx_ = dx[2] - dx[1]

order = 2
discretization = MOLFiniteDifference([x => dx_], t)
discretization_edge = MOLFiniteDifference([x => dx_], t; grid_align = edge_align)
# Explicitly specify order of centered difference
discretization_centered = MOLFiniteDifference([x => dx_], t; approx_order = order)
# Higher order centered difference
discretization_approx_order4 = MOLFiniteDifference([x => dx_], t; approx_order = 4)

for disc in [
        discretization, discretization_edge,
        discretization_centered, discretization_approx_order4,
    ]
    # Convert the PDE problem into an ODE problem
    # Here we are disabling unit checks because MOL does not currently
    # handle units correctly. In the future, we should add unit handling 
    # and re-enable these checks.
    prob = discretize(pdesys, disc; system_kwargs = [:checks => ~ModelingToolkit.CheckUnits])

    # Solve ODE problem      # Solve ODE problem
    sol = solve(prob, Tsit5(), saveat = 0.1)

    x_disc = sol[x][2:(end - 1)]
    t_disc = sol[t]
    u_approx = sol[u(t, x)][:, 2:(end - 1)]

    # Test against exact solution
    for i in 1:length(sol)
        exact = u_exact(x_disc, t_disc[i])
        @test all(isapprox.(u_approx[i, :], exact, atol = 0.01))
    end
end
