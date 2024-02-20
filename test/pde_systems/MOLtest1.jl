using ModelingToolkit, MethodOfLines, LinearAlgebra, OrdinaryDiffEq
using ModelingToolkit: operation, istree, arguments
using DomainSets
using NonlinearSolve
using StableRNGs
using Test

# # Define some variables
@testset "Heat Equation 1D 2 variables" begin
    @parameters t x
    @variables u(..) v(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2
    eqs = [Dt(u(t, x)) ~ Dxx(u(t, x)),
        Dt(v(t, x)) ~ Dxx(v(t, x))]
    bcs = [u(0, x) ~ -x * (x - 1) * sin(x),
        v(0, x) ~ -x * (x - 1) * sin(x),
        u(t, 0) ~ 0.0, u(t, 1) ~ 0.0,
        v(t, 0) ~ 0.0, v(t, 1) ~ 0.0]

    domains = [t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, 1.0)]

    @named pdesys = PDESystem(eqs, bcs, domains, [t, x], [u(t, x), v(t, x)])
    discretization = MOLFiniteDifference([x => 0.1], t; grid_align = edge_align)
    prob = discretize(pdesys, discretization) # This gives an ODEProblem since it's time-dependent
    sol = solve(prob, Tsit5())
end

@testset "Heat Equation 2D 1 variable" begin
    @parameters t x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    Dt = Differential(t)
    t_min = 0.0
    t_max = 2.0
    x_min = 0.0
    x_max = 2.0
    y_min = 0.0
    y_max = 2.0

    # 3D PDE
    eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y))

    analytic_sol_func(t, x, y) = exp(x + y) * cos(x + y + 4t)
    # Initial and boundary conditions
    bcs = [u(t_min, x, y) ~ analytic_sol_func(t_min, x, y),
        u(t, x_min, y) ~ analytic_sol_func(t, x_min, y),
        u(t, x_max, y) ~ analytic_sol_func(t, x_max, y),
        u(t, x, y_min) ~ analytic_sol_func(t, x, y_min),
        u(t, x, y_max) ~ analytic_sol_func(t, x, y_max)]

    # Space and time domains
    domains = [t ∈ Interval(t_min, t_max),
        x ∈ Interval(x_min, x_max),
        y ∈ Interval(y_min, y_max)]
    @named pdesys = PDESystem([eq], bcs, domains, [t, x, y], [u(t, x, y)])
    # Method of lines discretization
    dx = 0.1
    dy = 0.2
    discretization = MOLFiniteDifference([x => dx, y => dy], t)
    prob = ModelingToolkit.discretize(pdesys, discretization)
    sol = solve(prob, Tsit5())
end

# Diffusion in a sphere
@testset "Spherical Diffusion" begin
    @parameters t r
    @variables u(..)
    Dt = Differential(t)
    Dr = Differential(r)
    Drr = Dr^2
    eq = Dt(u(t, r)) ~ 1 / r^2 * Dr(r^2 * Dr(u(t, r)))
    bcs = [u(0, r) ~ -r * (r - 1) * sin(r),
        Dr(u(t, 0)) ~ 0.0, u(t, 1) ~ sin(1)]

    domains = [t ∈ Interval(0.0, 1.0),
        r ∈ Interval(0.0, 1.0)]

    @named pdesys = PDESystem(eq, bcs, domains, [t, r], [u(t, r)])
    discretization = MOLFiniteDifference([r => 0.1], t)
    prob = discretize(pdesys, discretization) # This gives an ODEProblem since it's time-dependent
    sol = solve(prob, Tsit5())
end

@testset "RHS = 0" begin
    @parameters x, t
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dx2 = Differential(x)^2
    Dx3 = Differential(x)^3
    Dx4 = Differential(x)^4

    α = 1
    β = 4
    γ = 1
    eq = Dt(u(x, t)) + u(x, t) * Dx(u(x, t)) + α * Dx2(u(x, t)) + β * Dx3(u(x, t)) + γ * Dx4(u(x, t)) ~ 0

    du(x, t; z = -x / 2 + t) = 15 / 2 * (tanh(z) + 1) * (3 * tanh(z) - 1) * sech(z)^2

    bcs = [u(x, 0) ~ x^2,
        Dx(u(-10, t)) ~ du(-10, t),
        Dx(u(10, t)) ~ du(10, t)]

    # Space and time domains
    domains = [x ∈ Interval(-10.0, 10.0),
        t ∈ Interval(0.0, 0.5)]
    # Discretization
    dx = 0.4
    dt = 0.2

    discretization = MOLFiniteDifference([x => dx], t; approx_order = 4)
    @named pdesys = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])
    prob = discretize(pdesys, discretization)
end

@testset "Wave Equation" begin
    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..)
    Dt = Differential(t)             # Required in ICs and equation
    Dtt = Differential(t)^2           # required in equation
    Dxxxx = Differential(x)^4       # required in equation
    Dxx = Differential(x)^2         # required in BCs
    # some parameters
    EI = 291.6667
    m = 1.3850
    c = 1.0
    p = 0.01
    L = 2.0

    # 1D PDE and boundaru conditions
    eq = m * Dtt(u(t, x)) + c * Dt(u(t, x)) + EI * Dxxxx(u(t, x)) ~ 0

    ic_bc = [u(0, x) ~ (p * x * (x^3 + L^3 - 2 * L * x^2) / (24 * EI)), #for all 0 < u < L
        Dt(u(0, x)) ~ 0.0,        # for all 0 < u < L
        u(t, 0) ~ 0.0,            # for all t > 0,, displacement zero at u=0
        u(t, 2) ~ 0.0,            # for all t > 0,, displacement zero at u=L
        Dxx(u(t, 0)) ~ 0.0,       # for all t > 0,, curvature zero at u=0
        Dxx(u(t, 2)) ~ 0.0]       # for all t > 0,, curvature zero at u=L

    # Space and time domains
    domains = [t ∈ Interval(0.0, 2.0),
        x ∈ Interval(0.0, L)]

    dt = 0.1   # dt related to saving the data.. not actual dt
    # PDE sustem
    @named pdesys = PDESystem(eq, ic_bc, domains, [t, x], [u(t, x)])

    # Method of lines discretization
    dx = 0.1
    order = 2
    discretization = MOLFiniteDifference([x => dx, t => dt], approx_order = order)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys, discretization)

    # Solve the ODE problem
    sol = NonlinearSolve.solve(prob, NewtonRaphson())
end

@testset "Rearranged Robin" begin
    @parameters x t

    @variables c(..)

    ∂t = Differential(t)
    ∂x = Differential(x)
    ∂²x = Differential(x)^2

    D₀ = 1.0
    R = 0.5
    cₑ = 2.5
    ℓ = 2.0
    α = 1 / 3

    diff_eq = ∂t(c(x, t)) ~ ∂x((D₀ + α * c(x, t)) * ∂x(c(x, t)))

    bcs = [c(x, 0) ~ 0.0,         # initial condition
        R * (D₀ + α * c(0, t)) * ∂x(c(0, t)) ~ c(0, t) - cₑ, # Robin
        ∂x(c(ℓ, t)) ~ 0.0]   # no flux

    domains = [t ∈ Interval(0.0, 10.0),
        x ∈ Interval(0.0, ℓ)]

    @named pdesys = PDESystem(diff_eq, bcs, domains, [x, t], [c(x, t)])

    # Method of lines discretization
    dx = 0.1
    order = 2
    discretization = MOLFiniteDifference([x => dx], t, approx_order = order)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys, discretization)

    sol = solve(prob, Rodas4())
end

@testset "Array u" begin
    # Dependencies
    N = 6 # number of dependent variables

    # Variables, parameters, and derivatives
    @parameters x
    @variables u[1:N](..)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    # Domain edges
    x_min = 0.0
    x_max = 1.0

    # Discretization parameters
    dx = 0.1
    order = 2

    #u = collect(u)

    # Equations
    eqs = Vector{ModelingToolkit.Equation}(undef, N)
    for i in 1:N
        eqs[i] = Dxx(u[i](x)) ~ u[i](x)
    end

    # Initial and boundary conditions
    bcs = Vector{ModelingToolkit.Equation}(undef, 2 * N)
    for i in 1:N
        bcs[i] = Dx(u[i](x_min)) ~ 0.0
    end

    for i in 1:N
        bcs[i + N] = u[i](x_max) ~ rand(StableRNG(0),)
    end

    # Space and time domains
    domains = [x ∈ Interval(x_min, x_max)]

    # PDE system
    @named pdesys = PDESystem(eqs, bcs, domains, [x], collect([u[i](x) for i in 1:N]))

    # Method of lines discretization
    discretization = MOLFiniteDifference([x => dx], nothing, approx_order = order)
    prob = ModelingToolkit.discretize(pdesys, discretization)

    # # Solution of the ODE system
    sol = NonlinearSolve.solve(prob, NewtonRaphson())
end

@testset "Non-Uniform Grid" begin
    @parameters x y t
    @variables u(..) v(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dy = Differential(y)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    ∇²(u) = Dxx(u) + Dyy(u)

    brusselator_f(x, y, t) = (((x - 0.3)^2 + (y - 0.6)^2) <= 0.1^2) * (t >= 1.1) * 5.0

    x_min = y_min = t_min = 0.0
    x_max = y_max = 1.0
    t_max = 11.5

    α = 10.0

    u0(x, y, t) = 22(y * (1 - y))^(3 / 2)
    v0(x, y, t) = 27(x * (1 - x))^(3 / 2)

    eq = [
        Dt(u(x, y, t)) ~ 1.0 + Dy(v(x, y, t)) * u(x, y, t)^2 - 4.4 * u(x, y, t) +
                         α * ∇²(u(x, y, t)) + brusselator_f(x, y, t),
        Dt(v(x, y, t)) ~ -3.4 * Dx(u(x, y, t)) - v(x, y, t) * u(x, y, t)^2 +
                         α * ∇²(v(x, y, t))]

    domains = [x ∈ Interval(x_min, x_max),
        y ∈ Interval(y_min, y_max),
        t ∈ Interval(t_min, t_max)]

    bcs = [u(x, y, 0) ~ u0(x, y, 0),
        u(0, y, t) ~ 0.0,
        u(1, y, t) ~ 0.0,
        u(x, 0, t) ~ 0.0,
        u(x, 1, t) ~ 0.0,
        v(x, y, 0) ~ v0(x, y, 0),
        v(0, y, t) ~ 0.0,
        v(1, y, t) ~ 0.0,
        v(x, 0, t) ~ 0.0,
        v(x, 1, t) ~ 0.0]

    @named pdesys = PDESystem(eq, bcs, domains, [x, y, t], [u(x, y, t), v(x, y, t)])

    # Method of lines discretization
    N = 5

    dx = [0.0, cumsum(rand(StableRNG(0), 0.05:0.01:0.2, 5))...]
    if dx[end] < 1.0
        push!(dx, 1.0)
    end

    dy = [0.0, cumsum(rand(StableRNG(0), 0.05:0.01:0.2, 5))...]
    if dy[end] < 1.0
        push!(dy, 1.0)
    end

    order = 2

    discretization = MOLFiniteDifference([x => dx, y => dy], t, approx_order = order)

    prob = discretize(pdesys, discretization)

    sol = solve(prob, TRBDF2())
end

@testset "2D variable connected to 1D variable at boundary #33" begin
    @parameters t x r
    @variables u(..) v(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2
    Dr = Differential(r)
    Drr = Differential(r)^2

    s = u(t, x) + v(t, x, 1)

    eqs = [Dt(u(t, x)) ~ Dxx(u(t, x)) + s,
        Dt(v(t, x, r)) ~ Drr(v(t, x, r))]
    bcs = [u(0, x) ~ 1,
        v(0, x, r) ~ 1,
        Dx(u(t, 0)) ~ 0.0, Dx(u(t, 1)) ~ 0.0,
        Dr(v(t, x, 0)) ~ 0.0, Dr(v(t, x, 1)) ~ s]

    domains = [t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, 1.0),
        r ∈ Interval(0.0, 1.0)]

    @named pdesys = PDESystem(eqs, bcs, domains, [t, x, r], [u(t, x), v(t, x, r)])

    # Method of lines discretization
    dx = 0.1
    dr = 0.1
    order = 2
    discretization = MOLFiniteDifference([x => dx, r => dr], t, approx_order = order)

    # Convert the PDE problem into an ODE problem
    prob = discretize(pdesys, discretization)

    sol = solve(prob, Tsit5())
end
