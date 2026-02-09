# 1D diffusion problem with units
# Test for https://github.com/SciML/MethodOfLines.jl/issues/511
# Verifies that finite difference discretization handles spatial variables with units correctly.

using ModelingToolkit, MethodOfLines, LinearAlgebra, Test, OrdinaryDiffEq, DomainSets,
    DynamicQuantities
using ModelingToolkit: Differential

@testset "1D Diffusion with units - symbolic_discretize" begin
    # Parameters, variables, and derivatives with units
    @parameters t [unit = u"s"]
    @parameters x [unit = u"m"]
    @parameters D_diff [unit = u"m^2/s"]
    @variables u(..) [unit = u"kg/m^3"]

    # Reference constants for dimensionally-correct expressions
    @constants begin
        u_ref = 1.0, [unit = u"kg/m^3"]
        x_ref = 1.0, [unit = u"m"]
        t_ref = 1.0, [unit = u"s"]
    end

    Dt = Differential(t)
    Dxx = Differential(x)^2

    # 1D PDE: diffusion equation
    eq = Dt(u(t, x)) ~ D_diff * Dxx(u(t, x))

    # Non-zero boundary conditions using symbolic constants for correct units
    # Boundary locations must be literal numbers, not symbolic expressions
    bcs = [
        u(0, x) ~ u_ref * cos(x / x_ref),
        u(t, 0) ~ u_ref * exp(-t / t_ref),
        u(t, 1) ~ u_ref * exp(-t / t_ref) * cos(1.0),
    ]

    # Space and time domains
    domains = [
        t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, 1.0),
    ]

    # PDE system
    @named pdesys = PDESystem(
        eq, bcs, domains, [t, x], [u(t, x)],
        [D_diff, u_ref, x_ref, t_ref],
        defaults = Dict(D_diff => 1.0, u_ref => 1.0, x_ref => 1.0, t_ref => 1.0)
    )

    # Method of lines discretization
    dx = 0.05
    discretization = MOLFiniteDifference([x => dx], t)

    # symbolic_discretize should succeed without unit errors.
    # Unit checking is performed during System construction (checks=true),
    # so reaching this point without error means units are consistent.
    sys, tspan = symbolic_discretize(pdesys, discretization)
    @test sys isa System
    @test tspan !== nothing
end

@testset "1D Diffusion with units - full solve" begin
    # Method of manufactured solutions with units
    # u_exact(t, x) = exp(-t) * cos(x) [in kg/m^3]
    # Dt(u) = -exp(-t)*cos(x), D*Dxx(u) = -D*exp(-t)*cos(x)
    # For D=1: Dt(u) = D*Dxx(u)

    @parameters t [unit = u"s"]
    @parameters x [unit = u"m"]
    @parameters D_diff [unit = u"m^2/s"]
    @variables u(..) [unit = u"kg/m^3"]

    @constants begin
        u_ref = 1.0, [unit = u"kg/m^3"]
        x_ref = 1.0, [unit = u"m"]
        t_ref = 1.0, [unit = u"s"]
    end

    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq = Dt(u(t, x)) ~ D_diff * Dxx(u(t, x))

    L = Float64(π)

    bcs = [
        u(0, x) ~ u_ref * cos(x / x_ref),
        u(t, 0) ~ u_ref * exp(-t / t_ref),
        u(t, L) ~ u_ref * exp(-t / t_ref) * cos(L),
    ]

    domains = [
        t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, L),
    ]

    @named pdesys = PDESystem(
        eq, bcs, domains, [t, x], [u(t, x)],
        [D_diff, u_ref, x_ref, t_ref],
        defaults = Dict(D_diff => 1.0, u_ref => 1.0, x_ref => 1.0, t_ref => 1.0)
    )

    dx_val = L / 29
    discretization = MOLFiniteDifference([x => dx_val], t)

    prob = discretize(pdesys, discretization)
    sol = solve(prob, Tsit5(), saveat = 0.1)

    u_exact = (x_val, t_val) -> exp(-t_val) * cos(x_val)

    x_disc = sol[x]
    t_disc = sol[t]

    # Test against exact solution
    for i in 1:length(t_disc)
        for j in 1:length(x_disc)
            @test isapprox(sol[u(t, x)][i, j], u_exact(x_disc[j], t_disc[i]), atol = 0.05)
        end
    end
end

@testset "unit_correct utilities" begin
    # Test make_unit_constant returns nothing for unitless variables
    @parameters x_no_unit
    @test MethodOfLines.make_unit_constant(x_no_unit) === nothing

    # Test make_unit_constant returns a constant for variables with units
    @parameters x_unit [unit = u"m"]
    uc = MethodOfLines.make_unit_constant(x_unit)
    @test uc !== nothing
    @test Symbolics.getdefaultval(uc) == 1.0

    # Test unit_correct with empty map (no-op)
    unit_map = Dict{Any, Any}()
    @test MethodOfLines.unit_correct(42, x_unit, 2, unit_map) == 42

    # Test unit_correct with populated map
    unit_map[x_unit] = uc
    result = MethodOfLines.unit_correct(42, x_unit, 2, unit_map)
    @test result !== 42  # Should have been divided by uc^2
end
