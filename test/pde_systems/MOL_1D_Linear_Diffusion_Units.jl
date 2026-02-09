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

@testset "1D Diffusion with units - Neumann BC symbolic_discretize" begin
    # Test that Neumann boundary conditions with units discretize correctly.
    # This verifies the unit_correct fix applied to boundary_value_maps in generate_bc_eqs.jl.
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
    Dx = Differential(x)
    Dxx = Differential(x)^2

    eq = Dt(u(t, x)) ~ D_diff * Dxx(u(t, x))

    # Neumann BC at x=0 (flux condition), Dirichlet at x=1
    bcs = [
        u(0, x) ~ u_ref * cos(x / x_ref),
        -D_diff * Dx(u(t, 0.0)) ~ u_ref * x_ref / t_ref,  # Neumann BC with correct units [kg/(m^2*s)]
        u(t, 1.0) ~ u_ref * exp(-t / t_ref) * cos(1.0),
    ]

    domains = [
        t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem(
        eq, bcs, domains, [t, x], [u(t, x)],
        [D_diff, u_ref, x_ref, t_ref],
        defaults = Dict(D_diff => 1.0, u_ref => 1.0, x_ref => 1.0, t_ref => 1.0)
    )

    dx = 0.05
    discretization = MOLFiniteDifference([x => dx], t)

    # symbolic_discretize should succeed without unit errors for Neumann BCs
    sys, tspan = symbolic_discretize(pdesys, discretization)
    @test sys isa System
    @test tspan !== nothing
end

@testset "1D Diffusion with units - Neumann BC full solve" begin
    # Heat equation with Neumann BC at x=0 and Dirichlet BC at x=L.
    # Steady state: d²T/dz² = 0 => T(x) = a*x + b
    # BC: -λ*dT/dx|_{x=0} = h => a = -h/λ
    # BC: T(L) = T_L => b = T_L + h*L/λ
    # T(x) = T_L + h*(L-x)/λ

    @parameters t [unit = u"s"]
    @parameters x [unit = u"m"]
    @variables T(..) [unit = u"K"]

    @parameters λ_cond [unit = u"W/(m*K)"]
    @parameters c_cap [unit = u"J/(m^3*K)"]
    @parameters h_flux [unit = u"W/m^2"]
    @parameters T_right [unit = u"K"]
    @parameters T_init [unit = u"K"]

    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    L = 0.3

    eq = c_cap * Dt(T(t, x)) ~ λ_cond * Dxx(T(t, x))

    bcs = [
        T(0, x) ~ T_init,                         # Initial condition
        -λ_cond * Dx(T(t, 0.0)) ~ h_flux,         # Neumann BC at left
        T(t, L) ~ T_right,                         # Dirichlet BC at right
    ]

    domains = [
        t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, L),
    ]

    λ_val = 1.5
    h_val = 50.0
    T_R = 290.0

    @named pdesys = PDESystem(
        eq, bcs, domains, [t, x], [T(t, x)],
        [λ_cond => λ_val, c_cap => 2.0e6, h_flux => h_val, T_right => T_R, T_init => T_R]
    )

    dx_val = L / 30
    discretization = MOLFiniteDifference([x => dx_val], t)

    prob = discretize(pdesys, discretization)

    # Solve for long time to reach steady state
    prob2 = remake(prob; tspan = (0.0, 100000.0))
    sol = solve(prob2, Tsit5())

    T_mat = sol[T(t, x)]
    T_final = T_mat[end, :]

    # Analytical steady state: T(x) = T_R + h*(L-x)/λ
    x_disc = sol[x]
    for j in 1:length(x_disc)
        T_analytical = T_R + h_val * (L - x_disc[j]) / λ_val
        @test isapprox(T_final[j], T_analytical, rtol = 0.05)
    end

    # Surface temperature should match analytical solution
    T_surface_analytical = T_R + h_val * L / λ_val
    @test isapprox(T_final[1], T_surface_analytical, rtol = 0.05)
end

@testset "1D Diffusion with units - Neumann BC both sides" begin
    # Test with Neumann BCs on both sides (zero flux = insulated boundaries).
    # With zero flux everywhere, temperature should remain at initial value.

    @parameters t [unit = u"s"]
    @parameters x [unit = u"m"]
    @variables T(..) [unit = u"K"]

    @parameters λ_cond [unit = u"W/(m*K)"]
    @parameters c_cap [unit = u"J/(m^3*K)"]
    @parameters T_init [unit = u"K"]
    @parameters zero_grad [unit = u"K/m"]

    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    L = 1.0

    eq = c_cap * Dt(T(t, x)) ~ λ_cond * Dxx(T(t, x))

    bcs = [
        T(0, x) ~ T_init,                    # Initial condition
        Dx(T(t, 0.0)) ~ zero_grad,           # Zero flux at left
        Dx(T(t, L)) ~ zero_grad,             # Zero flux at right
    ]

    domains = [
        t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, L),
    ]

    T_init_val = 300.0

    @named pdesys = PDESystem(
        eq, bcs, domains, [t, x], [T(t, x)],
        [λ_cond => 1.0, c_cap => 2.0e6, T_init => T_init_val, zero_grad => 0.0]
    )

    dx_val = L / 20
    discretization = MOLFiniteDifference([x => dx_val], t)

    prob = discretize(pdesys, discretization)
    sol = solve(prob, Tsit5(), saveat = 0.1)

    T_mat = sol[T(t, x)]

    # All temperatures should remain at initial value
    T_final = T_mat[end, :]
    for j in 1:length(T_final)
        @test isapprox(T_final[j], T_init_val, rtol = 1e-3)
    end

    # Energy conservation: sum of temperatures should be constant
    T_initial = T_mat[1, :]
    @test isapprox(sum(T_initial), sum(T_final), rtol = 1e-6)
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
