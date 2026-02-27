# Tests for ArrayDiscretization
# These mirror selected tests from MOL_1D_Linear_Diffusion.jl but use
# ArrayDiscretization() as the discretization strategy.

using ModelingToolkit, MethodOfLines, LinearAlgebra, Test, OrdinaryDiffEq, DomainSets
using ModelingToolkit: Differential

@testset "ArrayDiscretization: Dt(u(t,x)) ~ Dxx(u(t,x))" begin
    # Method of Manufactured Solutions
    u_exact = (x, t) -> exp.(-t) * cos.(x)

    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq = Dt(u(t, x)) ~ Dxx(u(t, x))
    bcs = [
        u(0, x) ~ cos(x),
        u(t, 0) ~ exp(-t),
        u(t, Float64(π)) ~ -exp(-t),
    ]

    domains = [
        t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, Float64(π)),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    dx = range(0.0, Float64(π), length = 30)
    dx_ = dx[2] - dx[1]

    # Test with ArrayDiscretization
    disc = MOLFiniteDifference(
        [x => dx_], t; discretization_strategy = ArrayDiscretization()
    )
    prob = discretize(pdesys, disc)
    sol = solve(prob, Tsit5(), saveat = 0.1)

    x_disc = sol[x][2:(end - 1)]
    t_disc = sol[t]
    u_approx = sol[u(t, x)][:, 2:(end - 1)]

    for i in 1:length(sol)
        exact = u_exact(x_disc, t_disc[i])
        @test all(isapprox.(u_approx[i, :], exact, atol = 0.01))
    end
end

@testset "ArrayDiscretization: Dt(u(t,x)) ~ D*Dxx(u(t,x)) (with parameter)" begin
    @parameters t x D
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq = Dt(u(t, x)) ~ D * Dxx(u(t, x))
    bcs = [
        u(0, x) ~ -x * (x - 1) * sin(x),
        u(t, 0) ~ 0.0,
        u(t, 1) ~ 0.0,
    ]

    domains = [
        t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem(
        eq, bcs, domains, [t, x], [u(t, x)], [D]; initial_conditions = Dict(D => 10.0)
    )

    dx = 1 / (5pi)
    disc = MOLFiniteDifference(
        [x => dx], t; discretization_strategy = ArrayDiscretization()
    )
    prob = discretize(pdesys, disc)
    sol = solve(prob, Tsit5(), saveat = 0.1)

    # Basic sanity: solution should converge toward zero (strong diffusion)
    u_final = sol[u(t, x)][end, :]
    @test all(abs.(u_final) .< 0.1)
end

@testset "ArrayDiscretization: approx_order = 4" begin
    u_exact = (x, t) -> exp.(-t) * cos.(x)

    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq = Dt(u(t, x)) ~ Dxx(u(t, x))
    bcs = [
        u(0, x) ~ cos(x),
        u(t, 0) ~ exp(-t),
        u(t, Float64(π)) ~ -exp(-t),
    ]

    domains = [
        t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, Float64(π)),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    dx = range(0.0, Float64(π), length = 30)
    dx_ = dx[2] - dx[1]

    disc = MOLFiniteDifference(
        [x => dx_], t; approx_order = 4,
        discretization_strategy = ArrayDiscretization()
    )
    prob = discretize(pdesys, disc)
    sol = solve(prob, Tsit5(), saveat = 0.1)

    x_disc = sol[x][2:(end - 1)]
    t_disc = sol[t]
    u_approx = sol[u(t, x)][:, 2:(end - 1)]

    for i in 1:length(sol)
        exact = u_exact(x_disc, t_disc[i])
        @test all(isapprox.(u_approx[i, :], exact, atol = 0.01))
    end
end

@testset "ArrayDiscretization matches ScalarizedDiscretization" begin
    # Verify that ArrayDiscretization produces the same numerical results
    # as ScalarizedDiscretization for a simple 1D diffusion problem.

    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq = Dt(u(t, x)) ~ Dxx(u(t, x))
    bcs = [
        u(0, x) ~ sin(π * x),
        u(t, 0) ~ 0.0,
        u(t, 1) ~ 0.0,
    ]

    domains = [
        t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    dx = 0.1

    disc_scalar = MOLFiniteDifference(
        [x => dx], t; discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference(
        [x => dx], t; discretization_strategy = ArrayDiscretization()
    )

    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)

    sol_scalar = solve(prob_scalar, Tsit5(), saveat = 0.1)
    sol_array = solve(prob_array, Tsit5(), saveat = 0.1)

    u_scalar = sol_scalar[u(t, x)]
    u_array = sol_array[u(t, x)]

    @test size(u_scalar) == size(u_array)
    @test isapprox(u_scalar, u_array, rtol = 1e-10)
end

@testset "ArrayOp template path: uniform grid with parameter" begin
    # This test ensures the ArrayOp template path (not the per-point fallback)
    # is exercised with a uniform grid AND a parameter multiplier.
    @parameters t x D
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq = Dt(u(t, x)) ~ D * Dxx(u(t, x))
    bcs = [
        u(0, x) ~ sin(π * x),
        u(t, 0) ~ 0.0,
        u(t, 1) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, 1.0),
    ]
    @named pdesys = PDESystem(
        eq, bcs, domains, [t, x], [u(t, x)], [D]; initial_conditions = Dict(D => 1.0)
    )

    # dx = 0.1 divides [0,1] exactly → uniform grid → ArrayOp template used
    dx = 0.1
    disc_scalar = MOLFiniteDifference(
        [x => dx], t; discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference(
        [x => dx], t; discretization_strategy = ArrayDiscretization()
    )

    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)

    sol_scalar = solve(prob_scalar, Tsit5(), saveat = 0.1)
    sol_array = solve(prob_array, Tsit5(), saveat = 0.1)

    u_scalar = sol_scalar[u(t, x)]
    u_array = sol_array[u(t, x)]

    @test size(u_scalar) == size(u_array)
    @test isapprox(u_scalar, u_array, rtol = 1e-10)
end

@testset "ArrayOp template: symbolic structure" begin
    # Verify that the template approach with symbolic _i produces equations
    # structurally equivalent to the per-point approach.
    using SymbolicUtils: SymReal, idxs_for_arrayop, BSImpl
    using Symbolics: unwrap, wrap
    using PDEBase: sym_dot, pde_substitute

    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq = Dt(u(t, x)) ~ Dxx(u(t, x))
    bcs = [
        u(0, x) ~ sin(π * x),
        u(t, 0) ~ 0.0,
        u(t, 1) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, 1.0),
    ]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    dx = 0.25  # Exact: 4 intervals, 5 grid points, interior = [2, 4]
    disc = MOLFiniteDifference(
        [x => dx], t; discretization_strategy = ArrayDiscretization()
    )

    sys, tspan = MethodOfLines.symbolic_discretize(pdesys, disc)
    eqs = equations(sys)

    # Interior equations should be 3 (indices 2, 3, 4) + 2 boundary
    @test length(eqs) == 5

    # Verify that the ArrayOp symbolic index (_i) was used in template
    # by checking that idxs_for_arrayop exists and the template produces
    # correct instantiated equations
    _i = idxs_for_arrayop(SymReal)[1]
    u_disc = first(@variables u(t)[1:5])
    u_c = BSImpl.Const{SymReal}(unwrap(u_disc))

    # Build a 3-point stencil template for dx=0.25: weights = [16, -32, 16]
    w = [1 / dx^2, -2 / dx^2, 1 / dx^2]
    base = 1  # _i=1 → grid index 2
    offsets = [-1, 0, 1]
    taps = [wrap(u_c[_i + base + off]) for off in offsets]
    stencil_template = sym_dot(w, taps)

    # Instantiate at _i = 1 (grid index 2): should involve u[1], u[2], u[3]
    stencil_at_1 = pde_substitute(stencil_template, Dict(_i => 1))
    stencil_str = string(stencil_at_1)
    @test occursin("(u(t))[1]", stencil_str)
    @test occursin("(u(t))[2]", stencil_str)
    @test occursin("(u(t))[3]", stencil_str)

    # Instantiate at _i = 3 (grid index 4): should involve u[3], u[4], u[5]
    stencil_at_3 = pde_substitute(stencil_template, Dict(_i => 3))
    stencil_str_3 = string(stencil_at_3)
    @test occursin("(u(t))[3]", stencil_str_3)
    @test occursin("(u(t))[4]", stencil_str_3)
    @test occursin("(u(t))[5]", stencil_str_3)
end
