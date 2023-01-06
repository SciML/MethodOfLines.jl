using ModelingToolkit, MethodOfLines, LinearAlgebra, OrdinaryDiffEq
using ModelingToolkit: operation, istree, arguments
using DomainSets
using NonlinearSolve
using StableRNGs
using Test

# # Define some variables



@testset "Testing discretization of varied systems" begin
    @parameters x t

    @variables c(..)

    ∂t = Differential(t)
    ∂x = Differential(x)
    ∂²x = Differential(x)^2

    D₀ = 1.5
    α = 0.15
    χ = 1.2
    R = 0.1
    cₑ = 2.0
    ℓ = 1.0
    Δx = 0.1

    bcs = [
        # initial condition
        c(x, 0) ~ 0.0,
        # Robin BC
        ∂x(c(0.0, t)) / (1 + exp(α * (c(0.0, t) - χ))) * R * D₀ + cₑ - c(0.0, t) ~ 0.0,
        # no flux BC
        ∂x(c(ℓ, t)) ~ 0.0]


    # define space-time plane
    domains = [x ∈ Interval(0.0, ℓ), t ∈ Interval(0.0, 5.0)]

    @testset "Test 01: ∂t(c(x, t)) ~ ∂x(D * ∂x(c(x, t)))" begin
        D = D₀ / (1.0 + exp(α * (c(x, t) - χ)))
        diff_eq = ∂t(c(x, t)) ~ ∂x(D * ∂x(c(x, t)))
        @named pdesys = PDESystem(diff_eq, bcs, domains, [x, t], [c(x, t)])
        discretization = MOLFiniteDifference([x => Δx], t)
    end

    @testset "Test 02: ∂t(c(x, t)) ~ ∂x(D * ∂x(c(x, t)))" begin
        D = 1.0 / (1.0 + exp(α * (c(x, t) - χ)))
        diff_eq = ∂t(c(x, t)) ~ ∂x(D * ∂x(c(x, t)))
        @named pdesys = PDESystem(diff_eq, bcs, domains, [x, t], [c(x, t)])
        discretization = MOLFiniteDifference([x => Δx], t)
    end

    @testset "Test 03: ∂t(c(x, t)) ~ ∂x(1.0 / (1.0/D₀ + exp(α * (c(x, t) - χ))/D₀) * ∂x(c(x, t)))" begin
        diff_eq = ∂t(c(x, t)) ~ ∂x(1.0 / (1.0 / D₀ + exp(α * (c(x, t) - χ)) / D₀) * ∂x(c(x, t)))
        @named pdesys = PDESystem(diff_eq, bcs, domains, [x, t], [c(x, t)])
        discretization = MOLFiniteDifference([x => Δx], t)
    end

    @testset "Test 04: ∂t(c(x, t)) ~ ∂x(D₀ / (1.0 + exp(α * (c(x, t) - χ))) * ∂x(c(x, t)))" begin
        diff_eq = ∂t(c(x, t)) ~ ∂x(D₀ / (1.0 + exp(α * (c(x, t) - χ))) * ∂x(c(x, t)))
        @named pdesys = PDESystem(diff_eq, bcs, domains, [x, t], [c(x, t)])
        discretization = MOLFiniteDifference([x => Δx], t)
    end

    @testset "Test 05: ∂t(c(x, t)) ~ ∂x(1/x * ∂x(c(x, t)))" begin
        diff_eq = ∂t(c(x, t)) ~ ∂x(1 / x * ∂x(c(x, t)))
        @named pdesys = PDESystem(diff_eq, bcs, domains, [x, t], [c(x, t)])
        discretization = MOLFiniteDifference([x => Δx], t)
    end

    @testset "Test 06: ∂t(c(x, t)) ~ ∂x(x*∂x(c(x, t)))/c(x,t)" begin
        diff_eq = ∂t(c(x, t)) ~ ∂x(x * ∂x(c(x, t))) / c(x, t)
        @named pdesys = PDESystem(diff_eq, bcs, domains, [x, t], [c(x, t)])
        discretization = MOLFiniteDifference([x => Δx], t)
    end

    @testset "Test 07: ∂t(c(x, t)) ~ ∂x(1/(1+c(x,t)) ∂x(c(x, t)))" begin
        diff_eq = ∂t(c(x, t)) ~ ∂x(1 / (1 + c(x, t)) * ∂x(c(x, t)))
        @named pdesys = PDESystem(diff_eq, bcs, domains, [x, t], [c(x, t)])
        discretization = MOLFiniteDifference([x => Δx], t)
    end

    @testset "Test 08: ∂t(c(x, t)) ~ c(x, t) * ∂x(c(x,t) * ∂x(c(x, t)))" begin
        diff_eq = ∂t(c(x, t)) ~ c(x, t) * ∂x(c(x, t) * ∂x(c(x, t)))
        @named pdesys = PDESystem(diff_eq, bcs, domains, [x, t], [c(x, t)])
        discretization = MOLFiniteDifference([x => Δx], t)
    end

    @testset "Test 09: ∂t(c(x, t)) ~ c(x, t) * ∂x(c(x,t) * ∂x(c(x, t)))/(1+c(x,t))" begin
        diff_eq = c(x, t) * ∂x(c(x, t) * ∂x(c(x, t))) / (1 + c(x, t))
        @named pdesys = PDESystem(diff_eq, bcs, domains, [x, t], [c(x, t)])
        discretization = MOLFiniteDifference([x => Δx], t)
    end

    @testset "Test 10: ∂t(c(x, t)) ~ c(x, t) * ∂x(c(x,t) * ∂x(c(x, t)))/(1+c(x,t))" begin
        diff_eq = c(x, t) * ∂x(c(x, t)^(-1) * ∂x(c(x, t)))
        @named pdesys = PDESystem(diff_eq, bcs, domains, [x, t], [c(x, t)])
        discretization = MOLFiniteDifference([x => Δx], t)
    end

    @testset "Test 11: ∂t(c(x, t)) ~ ∂x(1/(1+c(x,t)^2) ∂x(c(x, t)))" begin
        diff_eq = ∂t(c(x, t)) ~ ∂x(1 / (1 + c(x, t)^2) * ∂x(c(x, t)))
        @named pdesys = PDESystem(diff_eq, bcs, domains, [x, t], [c(x, t)])
        discretization = MOLFiniteDifference([x => Δx], t)
    end

end

@testset "Nonlinlap with flux interface boundary condition" begin
    @parameters t x1 x2
    @variables c1(..)
    @variables c2(..)
    Dt = Differential(t)

    Dx1 = Differential(x1)
    Dxx1 = Dx1^2

    Dx2 = Differential(x2)
    Dxx2 = Dx2^2

    D1(c) = 1 + c / 10
    D2(c) = 1 / 10 + c / 10

    eqs = [Dt(c1(t, x1)) ~ Dx1(D1(c1(t, x1)) * Dx1(c1(t, x1))),
        Dt(c2(t, x2)) ~ Dx2(D2(c2(t, x2)) * Dx2(c2(t, x2)))]

    bcs = [c1(0, x1) ~ 1 + cospi(2 * x1),
        c2(0, x2) ~ 1 + cospi(2 * x2),
        Dx1(c1(t, 0)) ~ 0,
        c1(t, 0.5) ~ c2(t, 0.5),
        -D1(c1(t, 0.5)) * Dx1(c1(t, 0.5)) ~ -D2(c2(t, 0.5)) * Dx2(c2(t, 0.5)),
        Dx2(c2(t, 1)) ~ 0]

    domains = [t ∈ Interval(0.0, 0.15),
        x1 ∈ Interval(0.0, 0.5),
        x2 ∈ Interval(0.5, 1.0)]

    @named pdesys = PDESystem(eqs, bcs, domains,
        [t, x1, x2], [c1(t, x1), c2(t, x2)])

    l = 40

    disc = MOLFiniteDifference([x1 => l, x2 => l], t)

    prob = discretize(pdesys, disc)

    sol = solve(prob, FBDF(), saveat=0.01)

    x1_sol = sol[x1]
    x2_sol = sol[x2]
    t_sol = sol[t]
    solc1 = sol[c1(t, x1)]
    solc2 = sol[c2(t, x2)]

    solc = hcat(solc1[:, :], solc2[:, 2:end])

    @test sol.retcode == :Success
end

@testset "Another boundaries appearing in equations case" begin

    g = 9.81

    @parameters x z t
    @variables φ(..) φ̃(..) η(..)

    Dt = Differential(t)
    Dx = Differential(x)
    Dz = Differential(z)
    Dxx = Differential(x)^2
    Dzz = Differential(z)^2

    eqs = [Dxx(φ(t, x, z)) + Dzz(φ(t, x, z)) ~ 0,
        Dt(φ̃(t, x)) ~ -g * η(t, x),
        Dt(η(t, x)) ~ Dz(φ(t, x, 1.0))
    ]

    bcs = [
        φ(0, x, z) ~ 0,
        φ̃(0.0, x) ~ 0.0,
        η(0.0, x) ~ cos(2 * π * x),
        φ(t, x, 1.0) ~ φ̃(t, x),
        Dx(φ(t, 0.0, z)) ~ 0.0,
        Dx(φ(t, 1.0, z)) ~ 0.0,
        Dz(φ(t, x, 0.0)) ~ 0.0,
        Dx(φ̃(t, 0.0)) ~ 0.0,
        Dx(φ̃(t, 1.0)) ~ 0.0,
        Dx(η(t, 0.0)) ~ 0.0,
        Dx(η(t, 1.0)) ~ 0.0,
    ]

    domains = [x ∈ Interval(0.0, 1.0),
        z ∈ Interval(0.0, 1.0),
        t ∈ Interval(0.0, 10.0)]

    @named pdesys = PDESystem(eqs, bcs, domains, [t, x, z],
        [φ(t, x, z), φ̃(t, x), η(t, x)])


    dx = 0.1
    dz = 0.1
    order = 2

    discretization = MOLFiniteDifference([x => dx, z => dz], t,
        approx_order=order,
        grid_align=center_align)

    println("Discretization:")
    prob = discretize(pdesys, discretization)
end

@testset "Integrals in BCs" begin
    β = 0.0005
    γ = 0.25
    amin = 0.0
    amax = 40.0

    @parameters t a
    @variables S(..) I(..) R(..)
    Dt = Differential(t)
    Da = Differential(a)
    Ia = Integral(a in DomainSets.ClosedInterval(amin, amax))


    eqs = [Dt(S(t)) ~ -β * S(t) * Ia(I(a, t)),
        Dt(I(a, t)) + Da(I(a, t)) ~ -γ * I(a, t),
        Dt(R(t)) ~ γ * Ia(I(a, t))]

    bcs = [
        S(0) ~ 990.0,
        I(0, t) ~ β * S(t) * Ia(I(a, t)),
        I(a, 0) ~ 10.0 / 40.0,
        R(0) ~ 0.0
    ]

    domains = [t ∈ (0.0, 40.0), a ∈ (0.0, 40.0)]

    @named pde_system = PDESystem(eqs, bcs, domains, [a, t], [S(t), I(a, t), R(t)])

    da = 40
    discretization = MOLFiniteDifference([a => da], t)

    prob = MethodOfLines.discretize(pde_system, discretization)

    sol = solve(prob, FBDF())

end
