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
    # Verify that the ArrayOp path produces a single array equation for the
    # centred interior region instead of N scalar equations.
    using SymbolicUtils: SymReal, idxs_for_arrayop, BSImpl
    using SymbolicUtils
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

    dx = 0.25  # 4 intervals, 5 grid points, interior = [2, 3, 4]
    disc = MOLFiniteDifference(
        [x => dx], t; discretization_strategy = ArrayDiscretization()
    )

    sys, tspan = MethodOfLines.symbolic_discretize(pdesys, disc)
    eqs = equations(sys)

    # With ArrayOp: 1 array equation (3 interior points) + 2 BC equations = 3
    @test length(eqs) == 3

    # Verify that at least one equation contains an ArrayOp
    has_arrayop = any(eqs) do eq
        lhs_raw = unwrap(eq.lhs)
        rhs_raw = unwrap(eq.rhs)
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(lhs_raw)) ||
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(rhs_raw))
    end
    @test has_arrayop

    # The ArrayOp equation should scalarize to 3 interior equations
    using ModelingToolkit.ModelingToolkitBase: flatten_equations
    flat = flatten_equations(eqs)
    @test length(flat) == 5  # 3 interior + 2 BCs

    # Verify stencil structure: the ArrayOp index mechanism works correctly
    _i = idxs_for_arrayop(SymReal)[1]
    u_disc_test = first(@variables u(t)[1:5])
    u_c = BSImpl.Const{SymReal}(unwrap(u_disc_test))

    w = [1 / dx^2, -2 / dx^2, 1 / dx^2]
    base = 1
    offsets = [-1, 0, 1]
    taps = [wrap(u_c[_i + base + off]) for off in offsets]
    stencil_template = sym_dot(w, taps)

    # Instantiate at _i = 1 (grid index 2): should involve u[1], u[2], u[3]
    stencil_at_1 = pde_substitute(stencil_template, Dict(_i => 1))
    stencil_str = string(stencil_at_1)
    @test occursin("(u(t))[1]", stencil_str)
    @test occursin("(u(t))[2]", stencil_str)
    @test occursin("(u(t))[3]", stencil_str)
end

# ─── Phase 2: Tests for per-point fallback (non-templateable PDEs) ───────────

@testset "ArrayDiscretization: Upwind convection (Burgers)" begin
    # Inviscid Burgers equation with UpwindScheme — exercises the per-point
    # fallback path because the PDE has an odd-order derivative (Dx).
    @parameters x t
    @variables u(..)
    Dx = Differential(x)
    Dt = Differential(t)

    analytic_u(t, x) = x / (t + 1)

    eq = Dt(u(t, x)) ~ -u(t, x) * Dx(u(t, x))

    bcs = [
        u(0, x) ~ x,
        u(t, 0.0) ~ analytic_u(t, 0.0),
        u(t, 1.0) ~ analytic_u(t, 1.0),
    ]

    domains = [
        t ∈ Interval(0.0, 6.0),
        x ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    disc = MOLFiniteDifference([x => 0.05], t;
        advection_scheme = UpwindScheme(),
        discretization_strategy = ArrayDiscretization()
    )
    prob = discretize(pdesys, disc)
    sol = solve(prob, Tsit5())

    x_disc = sol[x]
    solu = sol[u(t, x)]

    for (i, t_val) in enumerate(sol.t)
        u_analytic = analytic_u.([t_val], x_disc)
        @test all(isapprox.(u_analytic, solu[i, :], atol = 1.0e-3))
    end
end

@testset "ArrayDiscretization: Nonlinear diffusion" begin
    # Nonlinear diffusion Dt(u) ~ Dx(u^(-1) * Dx(u)) — exercises the per-point
    # fallback because this uses the nonlinear Laplacian scheme.
    @parameters t x
    @variables u(..)
    Dx = Differential(x)
    Dt = Differential(t)
    c = 1.0
    a = 1.0

    analytic_sol_func(t, x) = 2.0 * (c + t) / (a + x)^2

    eq = Dt(u(t, x)) ~ Dx(u(t, x)^(-1) * Dx(u(t, x)))

    bcs = [
        u(0.0, x) ~ analytic_sol_func(0.0, x),
        u(t, 0.0) ~ analytic_sol_func(t, 0.0),
        u(t, 2.0) ~ analytic_sol_func(t, 2.0),
    ]

    domains = [
        t ∈ Interval(0.0, 2.0),
        x ∈ Interval(0.0, 2.0),
    ]

    @named pdesys = PDESystem([eq], bcs, domains, [t, x], [u(t, x)])

    disc = MOLFiniteDifference([x => 0.01], t;
        discretization_strategy = ArrayDiscretization()
    )
    prob = discretize(pdesys, disc)
    sol = solve(prob, Rosenbrock32())
    @test SciMLBase.successful_retcode(sol)

    x_disc = sol[x]
    asf = [analytic_sol_func(2.0, x_val) for x_val in x_disc]
    sol′ = sol[u(t, x)]
    @test asf ≈ sol′[end, :] atol = 0.1
end

@testset "ArrayDiscretization: 2D diffusion" begin
    # 2D diffusion Dt(u) ~ Dxx(u) + Dyy(u) — now uses the N-D template path
    # for the centred-stencil interior region.
    @parameters t x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    Dt = Differential(t)

    analytic_sol_func(t, x, y) = exp(x + y) * cos(x + y + 4t)

    eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y))

    bcs = [
        u(0.0, x, y) ~ analytic_sol_func(0.0, x, y),
        u(t, 0.0, y) ~ analytic_sol_func(t, 0.0, y),
        u(t, 2.0, y) ~ analytic_sol_func(t, 2.0, y),
        u(t, x, 0.0) ~ analytic_sol_func(t, x, 0.0),
        u(t, x, 2.0) ~ analytic_sol_func(t, x, 2.0),
    ]

    domains = [
        t ∈ Interval(0.0, 2.0),
        x ∈ Interval(0.0, 2.0),
        y ∈ Interval(0.0, 2.0),
    ]

    @named pdesys = PDESystem([eq], bcs, domains, [t, x, y], [u(t, x, y)])

    disc = MOLFiniteDifference([x => 0.1, y => 0.2], t;
        approx_order = 4,
        discretization_strategy = ArrayDiscretization()
    )
    prob = discretize(pdesys, disc)
    sol = solve(prob, Tsit5())

    r_space_x = sol[x]
    r_space_y = sol[y]
    asf = [analytic_sol_func(2.0, X, Y) for X in r_space_x, Y in r_space_y]
    asf[1, 1] = asf[1, end] = asf[end, 1] = asf[end, end] = 0.0

    sol′ = sol[u(t, x, y)]
    @test asf ≈ sol′[end, :, :] atol = 0.4
end

# --- Phase 3: N-D template tests --------------------------------------------

@testset "ArrayDiscretization: 2D diffusion template matches scalar" begin
    # 2D diffusion with uniform grid and even-order derivatives only.
    # Should use the N-D template path. Compare Array vs Scalar solutions.
    @parameters t x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    Dt = Differential(t)

    eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y))

    bcs = [
        u(0.0, x, y) ~ sin(pi * x) * sin(pi * y),
        u(t, 0.0, y) ~ 0.0,
        u(t, 1.0, y) ~ 0.0,
        u(t, x, 0.0) ~ 0.0,
        u(t, x, 1.0) ~ 0.0,
    ]

    domains = [
        t ∈ Interval(0.0, 0.5),
        x ∈ Interval(0.0, 1.0),
        y ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem([eq], bcs, domains, [t, x, y], [u(t, x, y)])

    dx = 0.1
    dy = 0.1

    disc_scalar = MOLFiniteDifference([x => dx, y => dy], t;
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([x => dx, y => dy], t;
        discretization_strategy = ArrayDiscretization()
    )

    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)

    sol_scalar = solve(prob_scalar, Tsit5(), saveat = 0.1)
    sol_array = solve(prob_array, Tsit5(), saveat = 0.1)

    u_scalar = sol_scalar[u(t, x, y)]
    u_array = sol_array[u(t, x, y)]

    @test size(u_scalar) == size(u_array)
    @test isapprox(u_scalar, u_array, rtol = 1e-10)
end

@testset "ArrayOp template: 2D symbolic structure" begin
    # Small 2D grid: verify that the ArrayOp path produces a single array
    # equation that flattens to the correct number of scalar equations.
    using SymbolicUtils
    using Symbolics: unwrap

    @parameters t x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    Dt = Differential(t)

    eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y))

    bcs = [
        u(0.0, x, y) ~ sin(pi * x) * sin(pi * y),
        u(t, 0.0, y) ~ 0.0,
        u(t, 1.0, y) ~ 0.0,
        u(t, x, 0.0) ~ 0.0,
        u(t, x, 1.0) ~ 0.0,
    ]

    domains = [
        t ∈ Interval(0.0, 0.5),
        x ∈ Interval(0.0, 1.0),
        y ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem([eq], bcs, domains, [t, x, y], [u(t, x, y)])

    # dx=dy=0.25 => 5 grid points per dim, interior [2,4] x [2,4] = 3x3
    dx = 0.25
    disc = MOLFiniteDifference([x => dx, y => dx], t;
        discretization_strategy = ArrayDiscretization()
    )

    sys, tspan = MethodOfLines.symbolic_discretize(pdesys, disc)
    eqs = equations(sys)

    # Verify at least one equation contains an ArrayOp
    has_arrayop = any(eqs) do eq
        rhs_raw = unwrap(eq.rhs)
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(rhs_raw))
    end
    @test has_arrayop

    # After flattening, should have at least 9 interior equations
    using ModelingToolkit.ModelingToolkitBase: flatten_equations
    flat = flatten_equations(eqs)
    @test length(flat) >= 9
end

@testset "ArrayDiscretization: 3D diffusion matches scalar" begin
    # 3D diffusion on a coarse grid. Verifies the 3-index template path.
    @parameters t x y z
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    Dzz = Differential(z)^2
    Dt = Differential(t)

    eq = Dt(u(t, x, y, z)) ~ Dxx(u(t, x, y, z)) + Dyy(u(t, x, y, z)) + Dzz(u(t, x, y, z))

    bcs = [
        u(0.0, x, y, z) ~ sin(pi * x) * sin(pi * y) * sin(pi * z),
        u(t, 0.0, y, z) ~ 0.0,
        u(t, 1.0, y, z) ~ 0.0,
        u(t, x, 0.0, z) ~ 0.0,
        u(t, x, 1.0, z) ~ 0.0,
        u(t, x, y, 0.0) ~ 0.0,
        u(t, x, y, 1.0) ~ 0.0,
    ]

    domains = [
        t ∈ Interval(0.0, 0.1),
        x ∈ Interval(0.0, 1.0),
        y ∈ Interval(0.0, 1.0),
        z ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem([eq], bcs, domains, [t, x, y, z], [u(t, x, y, z)])

    d = 0.25  # coarse grid for speed

    disc_scalar = MOLFiniteDifference([x => d, y => d, z => d], t;
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([x => d, y => d, z => d], t;
        discretization_strategy = ArrayDiscretization()
    )

    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)

    sol_scalar = solve(prob_scalar, Tsit5(), saveat = 0.05)
    sol_array = solve(prob_array, Tsit5(), saveat = 0.05)

    u_scalar = sol_scalar[u(t, x, y, z)]
    u_array = sol_array[u(t, x, y, z)]

    @test size(u_scalar) == size(u_array)
    @test isapprox(u_scalar, u_array, rtol = 1e-10)
end

# --- Phase 4: Upwind and mixed derivative ArrayOp tests ----------------------

@testset "ArrayOp template: upwind Burgers symbolic structure" begin
    # Burgers equation with UpwindScheme on uniform grid should use the
    # ArrayOp path (not per-point fallback) and produce IfElse expressions.
    using SymbolicUtils
    using Symbolics: unwrap
    using ModelingToolkit.ModelingToolkitBase: flatten_equations

    @parameters x t
    @variables u(..)
    Dx = Differential(x)
    Dt = Differential(t)

    eq = Dt(u(t, x)) ~ -u(t, x) * Dx(u(t, x))

    bcs = [
        u(0, x) ~ x,
        u(t, 0.0) ~ 0.0,
        u(t, 1.0) ~ 1.0 / (t + 1),
    ]

    domains = [
        t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    dx = 0.1
    disc = MOLFiniteDifference([x => dx], t;
        advection_scheme = UpwindScheme(),
        discretization_strategy = ArrayDiscretization()
    )

    sys, tspan = MethodOfLines.symbolic_discretize(pdesys, disc)
    eqs = equations(sys)

    # Should have ArrayOp equations (not N scalar equations)
    has_arrayop = any(eqs) do eq
        lhs_raw = unwrap(eq.lhs)
        rhs_raw = unwrap(eq.rhs)
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(lhs_raw)) ||
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(rhs_raw))
    end
    @test has_arrayop

    # After flattening, should produce the right number of interior equations
    flat = flatten_equations(eqs)
    @test length(flat) >= 9  # interior points + BCs
end

@testset "ArrayDiscretization: Upwind Burgers ArrayOp matches scalar" begin
    # Compare ArrayOp path vs scalar path for Burgers with UpwindScheme.
    @parameters x t
    @variables u(..)
    Dx = Differential(x)
    Dt = Differential(t)

    analytic_u(t, x) = x / (t + 1)

    eq = Dt(u(t, x)) ~ -u(t, x) * Dx(u(t, x))

    bcs = [
        u(0, x) ~ x,
        u(t, 0.0) ~ analytic_u(t, 0.0),
        u(t, 1.0) ~ analytic_u(t, 1.0),
    ]

    domains = [
        t ∈ Interval(0.0, 2.0),
        x ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    disc_scalar = MOLFiniteDifference([x => 0.05], t;
        advection_scheme = UpwindScheme(),
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([x => 0.05], t;
        advection_scheme = UpwindScheme(),
        discretization_strategy = ArrayDiscretization()
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

@testset "ArrayDiscretization: 2D mixed derivative ArrayOp matches scalar" begin
    # PDE with mixed cross-derivative on uniform 2D grid.
    # Dt(u) ~ Dxx(u) + Dxy(u) + Dyy(u)
    @parameters t x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    Dxy = Differential(x) * Differential(y)
    Dt = Differential(t)

    eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dxy(u(t, x, y)) + Dyy(u(t, x, y))

    bcs = [
        u(0.0, x, y) ~ sin(pi * x) * sin(pi * y),
        u(t, 0.0, y) ~ 0.0,
        u(t, 1.0, y) ~ 0.0,
        u(t, x, 0.0) ~ 0.0,
        u(t, x, 1.0) ~ 0.0,
    ]

    domains = [
        t ∈ Interval(0.0, 0.5),
        x ∈ Interval(0.0, 1.0),
        y ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem([eq], bcs, domains, [t, x, y], [u(t, x, y)])

    dx = 0.1

    disc_scalar = MOLFiniteDifference([x => dx, y => dx], t;
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([x => dx, y => dx], t;
        discretization_strategy = ArrayDiscretization()
    )

    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)

    sol_scalar = solve(prob_scalar, Tsit5(), saveat = 0.1)
    sol_array = solve(prob_array, Tsit5(), saveat = 0.1)

    u_scalar = sol_scalar[u(t, x, y)]
    u_array = sol_array[u(t, x, y)]

    @test size(u_scalar) == size(u_array)
    @test isapprox(u_scalar, u_array, rtol = 1e-10)
end

# --- Phase 5: Nonlinear Laplacian ArrayOp tests ------------------------------

@testset "Nonlinlap ArrayOp matches scalar" begin
    # 1D nonlinear diffusion Dt(u) ~ Dx(u^(-1) * Dx(u)) on uniform grid.
    # Compare ArrayDiscretization (which should now use the ArrayOp path)
    # against ScalarizedDiscretization.
    @parameters t x
    @variables u(..)
    Dx = Differential(x)
    Dt = Differential(t)
    c = 1.0
    a = 1.0

    analytic_sol_func(t, x) = 2.0 * (c + t) / (a + x)^2

    eq = Dt(u(t, x)) ~ Dx(u(t, x)^(-1) * Dx(u(t, x)))

    bcs = [
        u(0.0, x) ~ analytic_sol_func(0.0, x),
        u(t, 0.0) ~ analytic_sol_func(t, 0.0),
        u(t, 2.0) ~ analytic_sol_func(t, 2.0),
    ]

    domains = [
        t ∈ Interval(0.0, 2.0),
        x ∈ Interval(0.0, 2.0),
    ]

    @named pdesys = PDESystem([eq], bcs, domains, [t, x], [u(t, x)])

    dx = 0.05

    disc_scalar = MOLFiniteDifference([x => dx], t;
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([x => dx], t;
        discretization_strategy = ArrayDiscretization()
    )

    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)

    sol_scalar = solve(prob_scalar, Rosenbrock32(), saveat = 0.5)
    sol_array = solve(prob_array, Rosenbrock32(), saveat = 0.5)

    u_scalar = sol_scalar[u(t, x)]
    u_array = sol_array[u(t, x)]

    @test size(u_scalar) == size(u_array)
    @test isapprox(u_scalar, u_array, rtol = 1e-10)
end

@testset "Nonlinlap ArrayOp symbolic structure" begin
    # Verify that the ArrayOp path produces a single array equation (not
    # per-point fallback) for a nonlinear Laplacian on uniform grid.
    using SymbolicUtils
    using Symbolics: unwrap
    using ModelingToolkit.ModelingToolkitBase: flatten_equations

    @parameters t x
    @variables u(..)
    Dx = Differential(x)
    Dt = Differential(t)

    eq = Dt(u(t, x)) ~ Dx(u(t, x)^(-1) * Dx(u(t, x)))

    bcs = [
        u(0.0, x) ~ 2.0 / (1.0 + x)^2,
        u(t, 0.0) ~ 2.0 * (1.0 + t),
        u(t, 2.0) ~ 2.0 * (1.0 + t) / 9.0,
    ]

    domains = [
        t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, 2.0),
    ]

    @named pdesys = PDESystem([eq], bcs, domains, [t, x], [u(t, x)])

    dx = 0.25
    disc = MOLFiniteDifference([x => dx], t;
        discretization_strategy = ArrayDiscretization()
    )

    sys, tspan = MethodOfLines.symbolic_discretize(pdesys, disc)
    eqs = equations(sys)

    # Should have ArrayOp equations (not N scalar equations)
    has_arrayop = any(eqs) do eq
        lhs_raw = unwrap(eq.lhs)
        rhs_raw = unwrap(eq.rhs)
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(lhs_raw)) ||
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(rhs_raw))
    end
    @test has_arrayop

    # After flattening, should produce the right number of equations
    flat = flatten_equations(eqs)
    # dx=0.25 on [0,2] gives 9 grid points, interior has some points
    # (exact count depends on boundary frame)
    @test length(flat) >= 5
end

@testset "Nonlinlap ArrayOp with analytical solution" begin
    # Verify the nonlinear Laplacian ArrayOp produces correct solutions by
    # comparing against the known analytical solution.
    @parameters t x
    @variables u(..)
    Dx = Differential(x)
    Dt = Differential(t)
    c = 1.0
    a = 1.0

    analytic_sol_func(t, x) = 2.0 * (c + t) / (a + x)^2

    eq = Dt(u(t, x)) ~ Dx(u(t, x)^(-1) * Dx(u(t, x)))

    bcs = [
        u(0.0, x) ~ analytic_sol_func(0.0, x),
        u(t, 0.0) ~ analytic_sol_func(t, 0.0),
        u(t, 2.0) ~ analytic_sol_func(t, 2.0),
    ]

    domains = [
        t ∈ Interval(0.0, 2.0),
        x ∈ Interval(0.0, 2.0),
    ]

    @named pdesys = PDESystem([eq], bcs, domains, [t, x], [u(t, x)])

    disc = MOLFiniteDifference([x => 0.05], t;
        discretization_strategy = ArrayDiscretization()
    )
    prob = discretize(pdesys, disc)
    sol = solve(prob, Rosenbrock32())
    @test SciMLBase.successful_retcode(sol)

    x_disc = sol[x]
    asf = [analytic_sol_func(2.0, x_val) for x_val in x_disc]
    sol′ = sol[u(t, x)]
    @test asf ≈ sol′[end, :] atol = 0.1
end

@testset "Nonlinlap ArrayOp higher-order (approx_order=4)" begin
    # Verify the nonlinear Laplacian ArrayOp works with 4th order accuracy.
    @parameters t x
    @variables u(..)
    Dx = Differential(x)
    Dt = Differential(t)
    c = 1.0
    a = 1.0

    analytic_sol_func(t, x) = 2.0 * (c + t) / (a + x)^2

    eq = Dt(u(t, x)) ~ Dx(u(t, x)^(-1) * Dx(u(t, x)))

    bcs = [
        u(0.0, x) ~ analytic_sol_func(0.0, x),
        u(t, 0.0) ~ analytic_sol_func(t, 0.0),
        u(t, 2.0) ~ analytic_sol_func(t, 2.0),
    ]

    domains = [
        t ∈ Interval(0.0, 2.0),
        x ∈ Interval(0.0, 2.0),
    ]

    @named pdesys = PDESystem([eq], bcs, domains, [t, x], [u(t, x)])

    dx = 0.05

    disc_scalar = MOLFiniteDifference([x => dx], t;
        approx_order = 4,
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([x => dx], t;
        approx_order = 4,
        discretization_strategy = ArrayDiscretization()
    )

    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)

    sol_scalar = solve(prob_scalar, Rosenbrock32(), saveat = 0.5)
    sol_array = solve(prob_array, Rosenbrock32(), saveat = 0.5)

    u_scalar = sol_scalar[u(t, x)]
    u_array = sol_array[u(t, x)]

    @test size(u_scalar) == size(u_array)
    @test isapprox(u_scalar, u_array, rtol = 1e-10)
end

# --- Phase 6: Spherical Laplacian ArrayOp tests ------------------------------

@testset "Spherical ArrayOp matches scalar" begin
    # Spherical diffusion Dt(u) ~ 1/r^2 * Dr(r^2 * Dr(u)) on uniform grid.
    # Compare ArrayDiscretization against ScalarizedDiscretization.
    @parameters t r
    @variables u(..)
    Dt = Differential(t)
    Dr = Differential(r)

    eq = Dt(u(t, r)) ~ 1 / r^2 * Dr(r^2 * Dr(u(t, r)))

    bcs = [
        u(0, r) ~ sin(r) / r,
        Dr(u(t, 0)) ~ 0,
        u(t, 1) ~ exp(-t) * sin(1),
    ]

    domains = [
        t ∈ Interval(0.0, 1.0),
        r ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, r], [u(t, r)])

    dr = 0.1

    disc_scalar = MOLFiniteDifference([r => dr], t;
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([r => dr], t;
        discretization_strategy = ArrayDiscretization()
    )

    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)

    sol_scalar = solve(prob_scalar, Rodas4(), saveat = 0.1)
    sol_array = solve(prob_array, Rodas4(), saveat = 0.1)

    u_scalar = sol_scalar[u(t, r)]
    u_array = sol_array[u(t, r)]

    @test size(u_scalar) == size(u_array)
    @test isapprox(u_scalar, u_array, rtol = 1e-10)
end

@testset "Spherical ArrayOp symbolic structure" begin
    # Verify that the spherical Laplacian uses the ArrayOp path (not per-point).
    using SymbolicUtils
    using Symbolics: unwrap
    using ModelingToolkit.ModelingToolkitBase: flatten_equations

    @parameters t r
    @variables u(..)
    Dt = Differential(t)
    Dr = Differential(r)

    eq = Dt(u(t, r)) ~ 1 / r^2 * Dr(r^2 * Dr(u(t, r)))

    bcs = [
        u(0, r) ~ sin(r) / r,
        Dr(u(t, 0)) ~ 0,
        u(t, 1) ~ exp(-t) * sin(1),
    ]

    domains = [
        t ∈ Interval(0.0, 1.0),
        r ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, r], [u(t, r)])

    dr = 0.1
    disc = MOLFiniteDifference([r => dr], t;
        discretization_strategy = ArrayDiscretization()
    )

    sys, tspan = MethodOfLines.symbolic_discretize(pdesys, disc)
    eqs = equations(sys)

    # Should have ArrayOp equations
    has_arrayop = any(eqs) do eq
        lhs_raw = unwrap(eq.lhs)
        rhs_raw = unwrap(eq.rhs)
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(lhs_raw)) ||
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(rhs_raw))
    end
    @test has_arrayop

    flat = flatten_equations(eqs)
    @test length(flat) >= 5
end

@testset "Spherical ArrayOp with coefficient" begin
    # Spherical diffusion with coefficient: Dt(u) ~ 4/r^2 * Dr(r^2 * Dr(u)).
    # Compare ArrayDiscretization against ScalarizedDiscretization.
    @parameters t r
    @variables u(..)
    Dt = Differential(t)
    Dr = Differential(r)

    eq = Dt(u(t, r)) ~ 4 / r^2 * Dr(r^2 * Dr(u(t, r)))

    bcs = [
        u(0, r) ~ sin(r) / r,
        Dr(u(t, 0)) ~ 0,
        u(t, 1) ~ exp(-4t) * sin(1),
    ]

    domains = [
        t ∈ Interval(0.0, 1.0),
        r ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, r], [u(t, r)])

    dr = 0.1

    disc_scalar = MOLFiniteDifference([r => dr], t;
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([r => dr], t;
        discretization_strategy = ArrayDiscretization()
    )

    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)

    sol_scalar = solve(prob_scalar, Tsit5(), saveat = 0.1)
    sol_array = solve(prob_array, Tsit5(), saveat = 0.1)

    u_scalar = sol_scalar[u(t, r)]
    u_array = sol_array[u(t, r)]

    @test size(u_scalar) == size(u_array)
    @test isapprox(u_scalar, u_array, rtol = 1e-10)
end

@testset "Spherical ArrayOp higher-order (approx_order=4)" begin
    # Spherical diffusion with 4th order accuracy.
    @parameters t r
    @variables u(..)
    Dt = Differential(t)
    Dr = Differential(r)

    eq = Dt(u(t, r)) ~ 1 / r^2 * Dr(r^2 * Dr(u(t, r)))

    bcs = [
        u(0, r) ~ sin(r) / r,
        Dr(u(t, 0)) ~ 0,
        u(t, 1) ~ exp(-t) * sin(1),
    ]

    domains = [
        t ∈ Interval(0.0, 1.0),
        r ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, r], [u(t, r)])

    dr = 0.1

    disc_scalar = MOLFiniteDifference([r => dr], t;
        approx_order = 4,
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([r => dr], t;
        approx_order = 4,
        discretization_strategy = ArrayDiscretization()
    )

    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)

    sol_scalar = solve(prob_scalar, Rodas4(), saveat = 0.1)
    sol_array = solve(prob_array, Rodas4(), saveat = 0.1)

    u_scalar = sol_scalar[u(t, r)]
    u_array = sol_array[u(t, r)]

    @test size(u_scalar) == size(u_array)
    @test isapprox(u_scalar, u_array, rtol = 1e-10)
end

# ===========================================================================
# Phase 7: Non-uniform grid ArrayOp tests
# ===========================================================================

using StableRNGs

@testset "Non-uniform ArrayOp: 1D diffusion matches scalar" begin
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

    # Non-uniform grid: perturb interior points
    dx = collect(range(0.0, Float64(π), length = 30))
    dx[2:(end - 1)] .= dx[2:(end - 1)] .+
        rand(StableRNG(0), [0.001, -0.001], length(dx[2:(end - 1)]))

    disc_scalar = MOLFiniteDifference([x => dx], t;
        discretization_strategy = ScalarizedDiscretization())
    disc_array = MOLFiniteDifference([x => dx], t;
        discretization_strategy = ArrayDiscretization())

    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)

    sol_scalar = solve(prob_scalar, Tsit5(), saveat = 0.1)
    sol_array = solve(prob_array, Tsit5(), saveat = 0.1)

    u_scalar = sol_scalar[u(t, x)]
    u_array = sol_array[u(t, x)]

    @test size(u_scalar) == size(u_array)
    @test isapprox(u_scalar, u_array, rtol = 1e-10)
end

@testset "Non-uniform ArrayOp: symbolic structure" begin
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

    # Non-uniform grid
    dx = collect(range(0.0, Float64(π), length = 30))
    dx[2:(end - 1)] .= dx[2:(end - 1)] .+
        rand(StableRNG(0), [0.001, -0.001], length(dx[2:(end - 1)]))

    disc = MOLFiniteDifference([x => dx], t;
        discretization_strategy = ArrayDiscretization())

    sys, tspan = MethodOfLines.symbolic_discretize(pdesys, disc)
    eqs = equations(sys)

    using Symbolics: unwrap
    using SymbolicUtils

    # With ArrayOp: 1 array equation (28 interior) + 2 BC = 3 equations
    @test length(eqs) == 3

    # Verify that at least one equation contains an ArrayOp
    has_arrayop = any(eqs) do eq
        lhs_raw = unwrap(eq.lhs)
        rhs_raw = unwrap(eq.rhs)
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(lhs_raw)) ||
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(rhs_raw))
    end
    @test has_arrayop

    # After flattening, should have 30 equations (28 interior + 2 boundary)
    using ModelingToolkit.ModelingToolkitBase: flatten_equations
    flat = flatten_equations(eqs)
    @test length(flat) == 30
end

@testset "Non-uniform ArrayOp: 1D diffusion with coefficient" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    D_coeff = 2.0
    eq = Dt(u(t, x)) ~ D_coeff * Dxx(u(t, x))
    bcs = [
        u(0, x) ~ cos(x),
        u(t, 0) ~ exp(-D_coeff * t),
        u(t, Float64(π)) ~ -exp(-D_coeff * t),
    ]

    domains = [
        t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, Float64(π)),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    dx = collect(range(0.0, Float64(π), length = 30))
    dx[2:(end - 1)] .= dx[2:(end - 1)] .+
        rand(StableRNG(42), [0.002, -0.002], length(dx[2:(end - 1)]))

    disc_scalar = MOLFiniteDifference([x => dx], t;
        discretization_strategy = ScalarizedDiscretization())
    disc_array = MOLFiniteDifference([x => dx], t;
        discretization_strategy = ArrayDiscretization())

    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)

    sol_scalar = solve(prob_scalar, Tsit5(), saveat = 0.1)
    sol_array = solve(prob_array, Tsit5(), saveat = 0.1)

    u_scalar = sol_scalar[u(t, x)]
    u_array = sol_array[u(t, x)]

    @test size(u_scalar) == size(u_array)
    @test isapprox(u_scalar, u_array, rtol = 1e-10)
end

@testset "Non-uniform ArrayOp: higher order (approx_order=4)" begin
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

    dx = collect(range(0.0, Float64(π), length = 30))
    dx[2:(end - 1)] .= dx[2:(end - 1)] .+
        rand(StableRNG(0), [0.001, -0.001], length(dx[2:(end - 1)]))

    disc_scalar = MOLFiniteDifference([x => dx], t; approx_order = 4,
        discretization_strategy = ScalarizedDiscretization())
    disc_array = MOLFiniteDifference([x => dx], t; approx_order = 4,
        discretization_strategy = ArrayDiscretization())

    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)

    sol_scalar = solve(prob_scalar, Tsit5(), saveat = 0.1)
    sol_array = solve(prob_array, Tsit5(), saveat = 0.1)

    u_scalar = sol_scalar[u(t, x)]
    u_array = sol_array[u(t, x)]

    @test size(u_scalar) == size(u_array)
    @test isapprox(u_scalar, u_array, rtol = 1e-10)
end

@testset "Non-uniform ArrayOp: 2D diffusion" begin
    @parameters t x y
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y))

    bcs = [
        u(0, x, y) ~ cos(x) * cos(y),
        u(t, 0, y) ~ exp(-2t) * cos(y),
        u(t, Float64(π), y) ~ -exp(-2t) * cos(y),
        u(t, x, 0) ~ exp(-2t) * cos(x),
        u(t, x, Float64(π)) ~ -exp(-2t) * cos(x),
    ]

    domains = [
        t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, Float64(π)),
        y ∈ Interval(0.0, Float64(π)),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])

    # Non-uniform grids in both dimensions
    dx = collect(range(0.0, Float64(π), length = 12))
    dx[2:(end - 1)] .= dx[2:(end - 1)] .+
        rand(StableRNG(1), [0.005, -0.005], length(dx[2:(end - 1)]))
    dy = collect(range(0.0, Float64(π), length = 12))
    dy[2:(end - 1)] .= dy[2:(end - 1)] .+
        rand(StableRNG(2), [0.005, -0.005], length(dy[2:(end - 1)]))

    disc_scalar = MOLFiniteDifference([x => dx, y => dy], t;
        discretization_strategy = ScalarizedDiscretization())
    disc_array = MOLFiniteDifference([x => dx, y => dy], t;
        discretization_strategy = ArrayDiscretization())

    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)

    sol_scalar = solve(prob_scalar, Tsit5(), saveat = 0.2)
    sol_array = solve(prob_array, Tsit5(), saveat = 0.2)

    u_scalar = sol_scalar[u(t, x, y)]
    u_array = sol_array[u(t, x, y)]

    @test size(u_scalar) == size(u_array)
    @test isapprox(u_scalar, u_array, rtol = 1e-10)
end

# ===========================================================================
# Phase 8: Non-uniform upwind ArrayOp tests
# ===========================================================================

@testset "Non-uniform upwind: scalar path bug fix verification" begin
    # 1D advection (Burgers) with UpwindScheme on a non-uniform grid.
    # Verify the scalar path produces correct results after the offside bug fix.
    # Use Burgers equation u*Dx(u) so the wind direction matters.
    @parameters x t
    @variables u(..)
    Dx = Differential(x)
    Dt = Differential(t)

    analytic_u(t, x) = x / (t + 1)

    eq = Dt(u(t, x)) ~ -u(t, x) * Dx(u(t, x))

    xs = sort([0.0; [0.1i + 0.01 * (-1)^i for i in 1:9]; 1.0])

    bcs = [
        u(0, x) ~ x,
        u(t, 0.0) ~ analytic_u(t, 0.0),
        u(t, 1.0) ~ analytic_u(t, 1.0),
    ]

    domains = [
        t ∈ Interval(0.0, 2.0),
        x ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    disc_scalar = MOLFiniteDifference([x => xs], t;
        advection_scheme = UpwindScheme(),
        discretization_strategy = ScalarizedDiscretization()
    )
    prob_scalar = discretize(pdesys, disc_scalar)
    sol_scalar = solve(prob_scalar, Tsit5())
    @test SciMLBase.successful_retcode(sol_scalar)

    x_disc = sol_scalar[x]
    solu = sol_scalar[u(t, x)]

    # On a coarse 11-point grid with first-order upwind, numerical diffusion is
    # significant, so use a generous tolerance.
    for (i, t_val) in enumerate(sol_scalar.t)
        u_analytic = analytic_u.([t_val], x_disc)
        @test all(isapprox.(u_analytic, solu[i, :], atol = 0.5))
    end
end

@testset "Non-uniform upwind: ArrayOp matches scalar" begin
    # 1D advection with UpwindScheme on non-uniform grid.
    # Compare ArrayDiscretization vs ScalarizedDiscretization.
    @parameters x t
    @variables u(..)
    Dx = Differential(x)
    Dt = Differential(t)

    analytic_u(t, x) = x / (t + 1)

    eq = Dt(u(t, x)) ~ -u(t, x) * Dx(u(t, x))

    xs = sort([0.0; [0.1i + 0.01 * (-1)^i for i in 1:9]; 1.0])

    bcs = [
        u(0, x) ~ x,
        u(t, 0.0) ~ analytic_u(t, 0.0),
        u(t, 1.0) ~ analytic_u(t, 1.0),
    ]

    domains = [
        t ∈ Interval(0.0, 2.0),
        x ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    disc_scalar = MOLFiniteDifference([x => xs], t;
        advection_scheme = UpwindScheme(),
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([x => xs], t;
        advection_scheme = UpwindScheme(),
        discretization_strategy = ArrayDiscretization()
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

@testset "Non-uniform upwind: symbolic structure" begin
    # Verify ArrayOp equations are produced (not per-point fallback).
    using SymbolicUtils
    using Symbolics: unwrap
    using ModelingToolkit.ModelingToolkitBase: flatten_equations

    @parameters x t
    @variables u(..)
    Dx = Differential(x)
    Dt = Differential(t)

    eq = Dt(u(t, x)) ~ -u(t, x) * Dx(u(t, x))

    xs = sort([0.0; [0.1i + 0.01 * (-1)^i for i in 1:9]; 1.0])

    bcs = [
        u(0, x) ~ x,
        u(t, 0.0) ~ 0.0,
        u(t, 1.0) ~ 1.0 / (t + 1),
    ]

    domains = [
        t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    disc = MOLFiniteDifference([x => xs], t;
        advection_scheme = UpwindScheme(),
        discretization_strategy = ArrayDiscretization()
    )

    sys, tspan = MethodOfLines.symbolic_discretize(pdesys, disc)
    eqs = equations(sys)

    # Should have ArrayOp equations (not N scalar equations)
    has_arrayop = any(eqs) do eq
        lhs_raw = unwrap(eq.lhs)
        rhs_raw = unwrap(eq.rhs)
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(lhs_raw)) ||
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(rhs_raw))
    end
    @test has_arrayop

    # After flattening, should produce interior + BC equations
    flat = flatten_equations(eqs)
    @test length(flat) >= 9  # interior points + BCs
end

@testset "Non-uniform upwind: with parameter" begin
    # Dt(u) ~ -v_param * Dx(u) with UpwindScheme and non-uniform grid.
    @parameters x t v_param
    @variables u(..)
    Dx = Differential(x)
    Dt = Differential(t)

    eq = Dt(u(t, x)) ~ -v_param * Dx(u(t, x))

    xs = sort([0.0; [0.1i + 0.01 * (-1)^i for i in 1:9]; 1.0])

    bcs = [
        u(0, x) ~ sin(π * x),
        u(t, 0.0) ~ 0.0,
        u(t, 1.0) ~ 0.0,
    ]

    domains = [
        t ∈ Interval(0.0, 0.5),
        x ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)], [v_param];
        initial_conditions = Dict(v_param => 0.5))

    disc_scalar = MOLFiniteDifference([x => xs], t;
        advection_scheme = UpwindScheme(),
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([x => xs], t;
        advection_scheme = UpwindScheme(),
        discretization_strategy = ArrayDiscretization()
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

@testset "Non-uniform upwind: advection-diffusion" begin
    # Dt(u) ~ -v * Dx(u) + D * Dxx(u) combining non-uniform upwind (odd order)
    # and non-uniform centered (even order).
    @parameters x t
    @variables u(..)
    Dx = Differential(x)
    Dxx = Differential(x)^2
    Dt = Differential(t)

    v = 0.5
    D_coeff = 0.1

    eq = Dt(u(t, x)) ~ -v * Dx(u(t, x)) + D_coeff * Dxx(u(t, x))

    xs = sort([0.0; [0.1i + 0.01 * (-1)^i for i in 1:9]; 1.0])

    bcs = [
        u(0, x) ~ sin(π * x),
        u(t, 0.0) ~ 0.0,
        u(t, 1.0) ~ 0.0,
    ]

    domains = [
        t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    disc_scalar = MOLFiniteDifference([x => xs], t;
        advection_scheme = UpwindScheme(),
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([x => xs], t;
        advection_scheme = UpwindScheme(),
        discretization_strategy = ArrayDiscretization()
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

# ===========================================================================
# Phase 9: Non-uniform nonlinear Laplacian, spherical Laplacian, and mixed
#           cross-derivative ArrayOp tests
# ===========================================================================

# --- 9a: Non-uniform nonlinear Laplacian ---

@testset "Non-uniform Nonlinlap ArrayOp matches scalar" begin
    # 1D nonlinear diffusion Dt(u) ~ Dx(u^(-1) * Dx(u)) on non-uniform grid.
    @parameters t x
    @variables u(..)
    Dx = Differential(x)
    Dt = Differential(t)
    c = 1.0
    a = 1.0

    analytic_sol_func(t, x) = 2.0 * (c + t) / (a + x)^2

    eq = Dt(u(t, x)) ~ Dx(u(t, x)^(-1) * Dx(u(t, x)))

    bcs = [
        u(0.0, x) ~ analytic_sol_func(0.0, x),
        u(t, 0.0) ~ analytic_sol_func(t, 0.0),
        u(t, 2.0) ~ analytic_sol_func(t, 2.0),
    ]

    domains = [
        t ∈ Interval(0.0, 2.0),
        x ∈ Interval(0.0, 2.0),
    ]

    @named pdesys = PDESystem([eq], bcs, domains, [t, x], [u(t, x)])

    # Non-uniform grid
    xs = collect(range(0.0, 2.0, length = 41))
    xs[2:(end - 1)] .+= rand(StableRNG(42), [0.001, -0.001], length(xs) - 2)

    disc_scalar = MOLFiniteDifference([x => xs], t;
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([x => xs], t;
        discretization_strategy = ArrayDiscretization()
    )

    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)

    sol_scalar = solve(prob_scalar, Rosenbrock32(), saveat = 0.5)
    sol_array = solve(prob_array, Rosenbrock32(), saveat = 0.5)

    u_scalar = sol_scalar[u(t, x)]
    u_array = sol_array[u(t, x)]

    @test size(u_scalar) == size(u_array)
    @test isapprox(u_scalar, u_array, rtol = 1e-10)
end

@testset "Non-uniform Nonlinlap ArrayOp symbolic structure" begin
    # Verify that the ArrayOp path produces array equations on a non-uniform grid.
    using SymbolicUtils
    using Symbolics: unwrap
    using ModelingToolkit.ModelingToolkitBase: flatten_equations

    @parameters t x
    @variables u(..)
    Dx = Differential(x)
    Dt = Differential(t)

    eq = Dt(u(t, x)) ~ Dx(u(t, x)^(-1) * Dx(u(t, x)))

    bcs = [
        u(0.0, x) ~ 2.0 / (1.0 + x)^2,
        u(t, 0.0) ~ 2.0 * (1.0 + t),
        u(t, 2.0) ~ 2.0 * (1.0 + t) / 9.0,
    ]

    domains = [
        t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, 2.0),
    ]

    @named pdesys = PDESystem([eq], bcs, domains, [t, x], [u(t, x)])

    xs = collect(range(0.0, 2.0, length = 20))
    xs[2:(end - 1)] .+= rand(StableRNG(43), [0.001, -0.001], length(xs) - 2)

    disc = MOLFiniteDifference([x => xs], t;
        discretization_strategy = ArrayDiscretization()
    )

    sys, tspan = MethodOfLines.symbolic_discretize(pdesys, disc)
    eqs = equations(sys)

    # Should have ArrayOp equations (not per-point fallback)
    has_arrayop = any(eqs) do eq
        lhs_raw = unwrap(eq.lhs)
        rhs_raw = unwrap(eq.rhs)
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(lhs_raw)) ||
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(rhs_raw))
    end
    @test has_arrayop

    flat = flatten_equations(eqs)
    @test length(flat) >= 5
end

@testset "Non-uniform Nonlinlap ArrayOp analytical solution" begin
    # Verify the nonlinear Laplacian ArrayOp produces correct solutions on
    # a non-uniform grid by comparing to the analytical solution.
    @parameters t x
    @variables u(..)
    Dx = Differential(x)
    Dt = Differential(t)
    c = 1.0
    a = 1.0

    analytic_sol_func(t, x) = 2.0 * (c + t) / (a + x)^2

    eq = Dt(u(t, x)) ~ Dx(u(t, x)^(-1) * Dx(u(t, x)))

    bcs = [
        u(0.0, x) ~ analytic_sol_func(0.0, x),
        u(t, 0.0) ~ analytic_sol_func(t, 0.0),
        u(t, 2.0) ~ analytic_sol_func(t, 2.0),
    ]

    domains = [
        t ∈ Interval(0.0, 2.0),
        x ∈ Interval(0.0, 2.0),
    ]

    @named pdesys = PDESystem([eq], bcs, domains, [t, x], [u(t, x)])

    xs = collect(range(0.0, 2.0, length = 41))
    xs[2:(end - 1)] .+= rand(StableRNG(44), [0.001, -0.001], length(xs) - 2)

    disc = MOLFiniteDifference([x => xs], t;
        discretization_strategy = ArrayDiscretization()
    )

    prob = discretize(pdesys, disc)
    sol = solve(prob, Rosenbrock32(), saveat = 0.5)

    x_disc = sol[x]
    t_disc = sol[t]
    u_approx = sol[u(t, x)]

    for i in eachindex(t_disc)
        exact = [analytic_sol_func(t_disc[i], xi) for xi in x_disc]
        @test isapprox(u_approx[i, :], exact, atol = 0.05)
    end
end

# --- 9b: Non-uniform spherical Laplacian ---

@testset "Non-uniform Spherical ArrayOp matches scalar" begin
    # Spherical diffusion Dt(u) ~ 1/r^2 * Dr(r^2 * Dr(u)) on non-uniform grid.
    @parameters t r
    @variables u(..)
    Dt = Differential(t)
    Dr = Differential(r)

    eq = Dt(u(t, r)) ~ 1 / r^2 * Dr(r^2 * Dr(u(t, r)))

    bcs = [
        u(0, r) ~ sin(r) / r,
        Dr(u(t, 0)) ~ 0,
        u(t, 1) ~ exp(-t) * sin(1),
    ]

    domains = [
        t ∈ Interval(0.0, 1.0),
        r ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, r], [u(t, r)])

    rs = collect(range(0.0, 1.0, length = 11))
    rs[2:(end - 1)] .+= rand(StableRNG(45), [0.001, -0.001], length(rs) - 2)

    disc_scalar = MOLFiniteDifference([r => rs], t;
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([r => rs], t;
        discretization_strategy = ArrayDiscretization()
    )

    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)

    sol_scalar = solve(prob_scalar, Rodas4(), saveat = 0.1)
    sol_array = solve(prob_array, Rodas4(), saveat = 0.1)

    u_scalar = sol_scalar[u(t, r)]
    u_array = sol_array[u(t, r)]

    @test size(u_scalar) == size(u_array)
    @test isapprox(u_scalar, u_array, rtol = 1e-10)
end

@testset "Non-uniform Spherical ArrayOp with coefficient" begin
    # Spherical diffusion with coefficient: Dt(u) ~ 4/r^2 * Dr(r^2 * Dr(u))
    # on non-uniform grid.
    @parameters t r
    @variables u(..)
    Dt = Differential(t)
    Dr = Differential(r)

    eq = Dt(u(t, r)) ~ 4 / r^2 * Dr(r^2 * Dr(u(t, r)))

    bcs = [
        u(0, r) ~ sin(r) / r,
        Dr(u(t, 0)) ~ 0,
        u(t, 1) ~ exp(-4t) * sin(1),
    ]

    domains = [
        t ∈ Interval(0.0, 1.0),
        r ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, r], [u(t, r)])

    rs = collect(range(0.0, 1.0, length = 11))
    rs[2:(end - 1)] .+= rand(StableRNG(46), [0.001, -0.001], length(rs) - 2)

    disc_scalar = MOLFiniteDifference([r => rs], t;
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([r => rs], t;
        discretization_strategy = ArrayDiscretization()
    )

    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)

    sol_scalar = solve(prob_scalar, Rodas4(), saveat = 0.1)
    sol_array = solve(prob_array, Rodas4(), saveat = 0.1)

    u_scalar = sol_scalar[u(t, r)]
    u_array = sol_array[u(t, r)]

    @test size(u_scalar) == size(u_array)
    @test isapprox(u_scalar, u_array, rtol = 1e-10)
end

# --- 9c: Non-uniform mixed cross-derivatives ---

@testset "Non-uniform mixed derivative ArrayOp matches scalar" begin
    # 2D PDE with mixed cross-derivative on non-uniform grids.
    # Dt(u) ~ Dxx(u) + Dxy(u) + Dyy(u)
    @parameters t x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    Dxy = Differential(x) * Differential(y)
    Dt = Differential(t)

    eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dxy(u(t, x, y)) + Dyy(u(t, x, y))

    bcs = [
        u(0.0, x, y) ~ sin(pi * x) * sin(pi * y),
        u(t, 0.0, y) ~ 0.0,
        u(t, 1.0, y) ~ 0.0,
        u(t, x, 0.0) ~ 0.0,
        u(t, x, 1.0) ~ 0.0,
    ]

    domains = [
        t ∈ Interval(0.0, 0.5),
        x ∈ Interval(0.0, 1.0),
        y ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem([eq], bcs, domains, [t, x, y], [u(t, x, y)])

    xs = collect(range(0.0, 1.0, length = 11))
    xs[2:(end - 1)] .+= rand(StableRNG(47), [0.001, -0.001], length(xs) - 2)
    ys = collect(range(0.0, 1.0, length = 11))
    ys[2:(end - 1)] .+= rand(StableRNG(48), [0.001, -0.001], length(ys) - 2)

    disc_scalar = MOLFiniteDifference([x => xs, y => ys], t;
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([x => xs, y => ys], t;
        discretization_strategy = ArrayDiscretization()
    )

    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)

    sol_scalar = solve(prob_scalar, Tsit5(), saveat = 0.1)
    sol_array = solve(prob_array, Tsit5(), saveat = 0.1)

    u_scalar = sol_scalar[u(t, x, y)]
    u_array = sol_array[u(t, x, y)]

    @test size(u_scalar) == size(u_array)
    @test isapprox(u_scalar, u_array, rtol = 1e-10)
end

@testset "Non-uniform mixed derivative ArrayOp symbolic structure" begin
    # Verify the ArrayOp path is used for mixed cross-derivatives on non-uniform grids.
    using SymbolicUtils
    using Symbolics: unwrap
    using ModelingToolkit.ModelingToolkitBase: flatten_equations

    @parameters t x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    Dxy = Differential(x) * Differential(y)
    Dt = Differential(t)

    eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dxy(u(t, x, y)) + Dyy(u(t, x, y))

    bcs = [
        u(0.0, x, y) ~ sin(pi * x) * sin(pi * y),
        u(t, 0.0, y) ~ 0.0,
        u(t, 1.0, y) ~ 0.0,
        u(t, x, 0.0) ~ 0.0,
        u(t, x, 1.0) ~ 0.0,
    ]

    domains = [
        t ∈ Interval(0.0, 0.5),
        x ∈ Interval(0.0, 1.0),
        y ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem([eq], bcs, domains, [t, x, y], [u(t, x, y)])

    xs = collect(range(0.0, 1.0, length = 11))
    xs[2:(end - 1)] .+= rand(StableRNG(49), [0.001, -0.001], length(xs) - 2)
    ys = collect(range(0.0, 1.0, length = 11))
    ys[2:(end - 1)] .+= rand(StableRNG(50), [0.001, -0.001], length(ys) - 2)

    disc = MOLFiniteDifference([x => xs, y => ys], t;
        discretization_strategy = ArrayDiscretization()
    )

    sys, tspan = MethodOfLines.symbolic_discretize(pdesys, disc)
    eqs = equations(sys)

    has_arrayop = any(eqs) do eq
        lhs_raw = unwrap(eq.lhs)
        rhs_raw = unwrap(eq.rhs)
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(lhs_raw)) ||
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(rhs_raw))
    end
    @test has_arrayop

    flat = flatten_equations(eqs)
    @test length(flat) >= 5
end

# ==========================================================================
# Phase 10: WENO ArrayOp tests
# ==========================================================================

# --- 10a: WENO ArrayOp matches scalar ---

@testset "WENO ArrayOp basic linear convection matches scalar" begin
    # Dt(u) ~ -Dx(u) with WENO scheme, Dirichlet BCs.
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    eq = Dt(u(t, x)) ~ -Dx(u(t, x))

    bcs = [
        u(0, x) ~ sin(pi * x),
        u(t, 0) ~ 0.0,
        u(t, 1) ~ 0.0,
    ]

    domains = [
        t ∈ Interval(0.0, 0.3),
        x ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    dx = 0.05
    disc_scalar = MOLFiniteDifference([x => dx], t;
        advection_scheme = WENOScheme(),
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([x => dx], t;
        advection_scheme = WENOScheme(),
        discretization_strategy = ArrayDiscretization()
    )

    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)

    using OrdinaryDiffEq
    sol_scalar = solve(prob_scalar, SSPRK33(), dt = 0.005, saveat = 0.05)
    sol_array = solve(prob_array, SSPRK33(), dt = 0.005, saveat = 0.05)

    u_scalar = sol_scalar[u(t, x)]
    u_array = sol_array[u(t, x)]

    @test size(u_scalar) == size(u_array)
    @test isapprox(u_scalar, u_array, rtol = 1e-10)
end

@testset "WENO ArrayOp coefficient-multiplied matches scalar" begin
    # Dt(u) ~ -v*Dx(u) with WENO scheme, Dirichlet BCs.
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    eq = Dt(u(t, x)) ~ -2.0 * Dx(u(t, x))

    bcs = [
        u(0, x) ~ sin(pi * x),
        u(t, 0) ~ 0.0,
        u(t, 1) ~ 0.0,
    ]

    domains = [
        t ∈ Interval(0.0, 0.3),
        x ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    dx = 0.05
    disc_scalar = MOLFiniteDifference([x => dx], t;
        advection_scheme = WENOScheme(),
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([x => dx], t;
        advection_scheme = WENOScheme(),
        discretization_strategy = ArrayDiscretization()
    )

    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)

    using OrdinaryDiffEq
    sol_scalar = solve(prob_scalar, SSPRK33(), dt = 0.005, saveat = 0.05)
    sol_array = solve(prob_array, SSPRK33(), dt = 0.005, saveat = 0.05)

    u_scalar = sol_scalar[u(t, x)]
    u_array = sol_array[u(t, x)]

    @test size(u_scalar) == size(u_array)
    @test isapprox(u_scalar, u_array, rtol = 1e-10)
end

@testset "WENO ArrayOp mixed advection+diffusion matches scalar" begin
    # Dt(u) ~ -Dx(u) + D*Dxx(u) with WENO advection + centered diffusion.
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    D_coeff = 0.01
    eq = Dt(u(t, x)) ~ -Dx(u(t, x)) + D_coeff * Dxx(u(t, x))

    bcs = [
        u(0, x) ~ sin(pi * x),
        u(t, 0) ~ 0.0,
        u(t, 1) ~ 0.0,
    ]

    domains = [
        t ∈ Interval(0.0, 0.3),
        x ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    dx = 0.05
    disc_scalar = MOLFiniteDifference([x => dx], t;
        advection_scheme = WENOScheme(),
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([x => dx], t;
        advection_scheme = WENOScheme(),
        discretization_strategy = ArrayDiscretization()
    )

    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)

    using OrdinaryDiffEq
    sol_scalar = solve(prob_scalar, SSPRK33(), dt = 0.005, saveat = 0.05)
    sol_array = solve(prob_array, SSPRK33(), dt = 0.005, saveat = 0.05)

    u_scalar = sol_scalar[u(t, x)]
    u_array = sol_array[u(t, x)]

    @test size(u_scalar) == size(u_array)
    @test isapprox(u_scalar, u_array, rtol = 1e-10)
end

# --- 10b: WENO ArrayOp symbolic structure ---

@testset "WENO ArrayOp symbolic structure" begin
    # Verify the ArrayOp path is used for WENO scheme.
    using SymbolicUtils
    using Symbolics: unwrap
    using ModelingToolkit.ModelingToolkitBase: flatten_equations

    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    eq = Dt(u(t, x)) ~ -Dx(u(t, x))

    bcs = [
        u(0, x) ~ sin(pi * x),
        u(t, 0) ~ 0.0,
        u(t, 1) ~ 0.0,
    ]

    domains = [
        t ∈ Interval(0.0, 0.3),
        x ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    dx = 0.05
    disc = MOLFiniteDifference([x => dx], t;
        advection_scheme = WENOScheme(),
        discretization_strategy = ArrayDiscretization()
    )

    sys, tspan = MethodOfLines.symbolic_discretize(pdesys, disc)
    eqs = equations(sys)

    has_arrayop = any(eqs) do eq
        lhs_raw = unwrap(eq.lhs)
        rhs_raw = unwrap(eq.rhs)
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(lhs_raw)) ||
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(rhs_raw))
    end
    @test has_arrayop

    flat = flatten_equations(eqs)
    @test length(flat) >= 5
end

# --- 10c: WENO with higher-order odd derivatives ---

@testset "WENO ArrayOp with 3rd-order derivative matches scalar" begin
    # PDE with Dxxx alongside WENO 1st-order: Dt(u) ~ -Dx(u) + 0.01*Dxxx(u)
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxxx = Differential(x)^3

    eq = Dt(u(t, x)) ~ -Dx(u(t, x)) + 0.01 * Dxxx(u(t, x))

    bcs = [
        u(0, x) ~ sin(pi * x),
        u(t, 0) ~ 0.0,
        u(t, 1) ~ 0.0,
    ]

    domains = [
        t ∈ Interval(0.0, 0.3),
        x ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    dx = 0.05
    disc_scalar = MOLFiniteDifference([x => dx], t;
        advection_scheme = WENOScheme(),
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([x => dx], t;
        advection_scheme = WENOScheme(),
        discretization_strategy = ArrayDiscretization()
    )

    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)

    using OrdinaryDiffEq
    sol_scalar = solve(prob_scalar, SSPRK33(), dt = 0.005, saveat = 0.05)
    sol_array = solve(prob_array, SSPRK33(), dt = 0.005, saveat = 0.05)

    u_scalar = sol_scalar[u(t, x)]
    u_array = sol_array[u(t, x)]

    @test size(u_scalar) == size(u_array)
    @test isapprox(u_scalar, u_array, rtol = 1e-10)
end

# ─── Phase 11: Periodic BCs, Multi-Variable Systems, and Neumann/Robin BC Tests ───

# --- 11a: Periodic BC centered derivative ---

@testset "Periodic BC: centered diffusion ArrayOp matches scalar" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    L = 2.0
    eq = Dt(u(t, x)) ~ Dxx(u(t, x))

    bcs = [
        u(0, x) ~ sin(2π * x / L),
        u(t, 0) ~ u(t, L),  # Periodic BC
    ]

    domains = [
        t ∈ Interval(0.0, 0.5),
        x ∈ Interval(0.0, L),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    dx = 0.1
    disc_scalar = MOLFiniteDifference([x => dx], t;
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([x => dx], t;
        discretization_strategy = ArrayDiscretization()
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

# --- 11b: Periodic BC WENO advection ---

@testset "Periodic BC: WENO advection ArrayOp matches scalar" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    L = 2.0
    eq = Dt(u(t, x)) ~ -Dx(u(t, x))

    bcs = [
        u(0, x) ~ sin(2π * x / L),
        u(t, 0) ~ u(t, L),  # Periodic BC
    ]

    domains = [
        t ∈ Interval(0.0, 0.3),
        x ∈ Interval(0.0, L),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    dx = 0.1
    disc_scalar = MOLFiniteDifference([x => dx], t;
        advection_scheme = WENOScheme(),
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([x => dx], t;
        advection_scheme = WENOScheme(),
        discretization_strategy = ArrayDiscretization()
    )

    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)

    using OrdinaryDiffEq
    sol_scalar = solve(prob_scalar, SSPRK33(), dt = 0.005, saveat = 0.05)
    sol_array = solve(prob_array, SSPRK33(), dt = 0.005, saveat = 0.05)

    u_scalar = sol_scalar[u(t, x)]
    u_array = sol_array[u(t, x)]

    @test size(u_scalar) == size(u_array)
    @test isapprox(u_scalar, u_array, rtol = 1e-10)
end

# --- 11c: Periodic BC upwind ---

@testset "Periodic BC: upwind advection ArrayOp matches scalar" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    L = 2.0
    eq = Dt(u(t, x)) ~ -Dx(u(t, x))

    bcs = [
        u(0, x) ~ sin(2π * x / L),
        u(t, 0) ~ u(t, L),  # Periodic BC
    ]

    domains = [
        t ∈ Interval(0.0, 0.3),
        x ∈ Interval(0.0, L),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    dx = 0.1
    disc_scalar = MOLFiniteDifference([x => dx], t;
        advection_scheme = UpwindScheme(),
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([x => dx], t;
        advection_scheme = UpwindScheme(),
        discretization_strategy = ArrayDiscretization()
    )

    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)

    sol_scalar = solve(prob_scalar, Tsit5(), saveat = 0.05)
    sol_array = solve(prob_array, Tsit5(), saveat = 0.05)

    u_scalar = sol_scalar[u(t, x)]
    u_array = sol_array[u(t, x)]

    @test size(u_scalar) == size(u_array)
    @test isapprox(u_scalar, u_array, rtol = 1e-10)
end

# --- 11d: Periodic BC symbolic structure ---

@testset "Periodic BC: ArrayDiscretization symbolic structure" begin
    # Periodic BCs use per-point fallback (not ArrayOp template) because index
    # aliasing (u[1] → u[N]) is incompatible with the ArrayOp template.
    # Verify that ArrayDiscretization still produces valid equations.
    using ModelingToolkit.ModelingToolkitBase: flatten_equations

    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    L = 2.0
    eq = Dt(u(t, x)) ~ Dxx(u(t, x))

    bcs = [
        u(0, x) ~ sin(2π * x / L),
        u(t, 0) ~ u(t, L),
    ]

    domains = [
        t ∈ Interval(0.0, 0.5),
        x ∈ Interval(0.0, L),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    dx = 0.25
    disc = MOLFiniteDifference([x => dx], t;
        discretization_strategy = ArrayDiscretization()
    )

    sys, tspan = MethodOfLines.symbolic_discretize(pdesys, disc)
    eqs = equations(sys)

    # Verify equations are produced (per-point fallback for periodic BCs)
    flat = flatten_equations(eqs)
    @test length(flat) >= 7  # At least N-2 interior + boundary equations
end

# --- 11e: Two coupled diffusion equations ---

@testset "Multi-variable: coupled diffusion ArrayOp matches scalar" begin
    @parameters t x
    @variables u(..) v(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eqs = [
        Dt(u(t, x)) ~ Dxx(u(t, x)) + v(t, x),
        Dt(v(t, x)) ~ Dxx(v(t, x)) + u(t, x),
    ]

    bcs = [
        u(0, x) ~ sin(π * x),
        v(0, x) ~ cos(π * x / 2) * sin(π * x),
        u(t, 0) ~ 0.0,
        u(t, 1) ~ 0.0,
        v(t, 0) ~ 0.0,
        v(t, 1) ~ 0.0,
    ]

    domains = [
        t ∈ Interval(0.0, 0.3),
        x ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem(eqs, bcs, domains, [t, x], [u(t, x), v(t, x)])

    dx = 0.05
    disc_scalar = MOLFiniteDifference([x => dx], t;
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([x => dx], t;
        discretization_strategy = ArrayDiscretization()
    )

    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)

    sol_scalar = solve(prob_scalar, Tsit5(), saveat = 0.05)
    sol_array = solve(prob_array, Tsit5(), saveat = 0.05)

    u_scalar = sol_scalar[u(t, x)]
    u_array = sol_array[u(t, x)]
    v_scalar = sol_scalar[v(t, x)]
    v_array = sol_array[v(t, x)]

    @test size(u_scalar) == size(u_array)
    @test isapprox(u_scalar, u_array, rtol = 1e-10)
    @test size(v_scalar) == size(v_array)
    @test isapprox(v_scalar, v_array, rtol = 1e-10)
end

# --- 11f: Advection-diffusion coupled system ---

@testset "Multi-variable: advection-diffusion coupled ArrayOp matches scalar" begin
    @parameters t x
    @variables u(..) v(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    eqs = [
        Dt(u(t, x)) ~ -Dx(u(t, x)) + 0.01 * Dxx(u(t, x)) + v(t, x),
        Dt(v(t, x)) ~ Dxx(v(t, x)) - u(t, x),
    ]

    bcs = [
        u(0, x) ~ sin(π * x),
        v(0, x) ~ 0.0,
        u(t, 0) ~ 0.0,
        u(t, 1) ~ 0.0,
        v(t, 0) ~ 0.0,
        v(t, 1) ~ 0.0,
    ]

    domains = [
        t ∈ Interval(0.0, 0.3),
        x ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem(eqs, bcs, domains, [t, x], [u(t, x), v(t, x)])

    dx = 0.05
    disc_scalar = MOLFiniteDifference([x => dx], t;
        advection_scheme = UpwindScheme(),
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([x => dx], t;
        advection_scheme = UpwindScheme(),
        discretization_strategy = ArrayDiscretization()
    )

    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)

    sol_scalar = solve(prob_scalar, Tsit5(), saveat = 0.05)
    sol_array = solve(prob_array, Tsit5(), saveat = 0.05)

    u_scalar = sol_scalar[u(t, x)]
    u_array = sol_array[u(t, x)]
    v_scalar = sol_scalar[v(t, x)]
    v_array = sol_array[v(t, x)]

    @test size(u_scalar) == size(u_array)
    @test isapprox(u_scalar, u_array, rtol = 1e-10)
    @test size(v_scalar) == size(v_array)
    @test isapprox(v_scalar, v_array, rtol = 1e-10)
end

# --- 11g: Multi-variable symbolic structure ---

@testset "Multi-variable: ArrayOp symbolic structure" begin
    using SymbolicUtils
    using Symbolics: unwrap
    using ModelingToolkit.ModelingToolkitBase: flatten_equations

    @parameters t x
    @variables u(..) v(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eqs = [
        Dt(u(t, x)) ~ Dxx(u(t, x)) + v(t, x),
        Dt(v(t, x)) ~ Dxx(v(t, x)) + u(t, x),
    ]

    bcs = [
        u(0, x) ~ sin(π * x),
        v(0, x) ~ 0.0,
        u(t, 0) ~ 0.0,
        u(t, 1) ~ 0.0,
        v(t, 0) ~ 0.0,
        v(t, 1) ~ 0.0,
    ]

    domains = [
        t ∈ Interval(0.0, 0.3),
        x ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem(eqs, bcs, domains, [t, x], [u(t, x), v(t, x)])

    dx = 0.25
    disc = MOLFiniteDifference([x => dx], t;
        discretization_strategy = ArrayDiscretization()
    )

    sys, tspan = MethodOfLines.symbolic_discretize(pdesys, disc)
    eqns = equations(sys)

    # Count ArrayOp equations — should have at least one for each variable
    arrayop_count = count(eqns) do eq
        lhs_raw = unwrap(eq.lhs)
        rhs_raw = unwrap(eq.rhs)
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(lhs_raw)) ||
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(rhs_raw))
    end
    @test arrayop_count >= 2
end

# --- 11h: Neumann BC diffusion ---

@testset "Neumann BC: diffusion ArrayOp matches scalar" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    eq = Dt(u(t, x)) ~ Dxx(u(t, x))

    bcs = [
        u(0, x) ~ cos(x),
        Dx(u(t, 0)) ~ 0.0,                   # Left Neumann
        u(t, Float64(π)) ~ -exp(-t),          # Right Dirichlet
    ]

    domains = [
        t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, Float64(π)),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    dx = range(0.0, Float64(π), length = 30)
    dx_ = dx[2] - dx[1]

    disc_scalar = MOLFiniteDifference([x => dx_], t;
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([x => dx_], t;
        discretization_strategy = ArrayDiscretization()
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

# --- 11i: Robin BC diffusion ---

@testset "Robin BC: diffusion ArrayOp matches scalar" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    eq = Dt(u(t, x)) ~ Dxx(u(t, x))

    bcs = [
        u(0, x) ~ sin(x),
        u(t, -1.0) + 3Dx(u(t, -1.0)) ~ exp(-t) * (sin(-1.0) + 3cos(-1.0)),  # Robin BC
        4u(t, 1.0) + Dx(u(t, 1.0)) ~ exp(-t) * (4sin(1.0) + cos(1.0)),      # Robin BC
    ]

    domains = [
        t ∈ Interval(0.0, 1.0),
        x ∈ Interval(-1.0, 1.0),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    dx = 0.1
    disc_scalar = MOLFiniteDifference([x => dx], t;
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([x => dx], t;
        discretization_strategy = ArrayDiscretization()
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

# --- 11j: Mixed BCs with upwind advection ---

@testset "Neumann + Dirichlet: upwind advection ArrayOp matches scalar" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    eq = Dt(u(t, x)) ~ -Dx(u(t, x)) + 0.1 * Dxx(u(t, x))

    bcs = [
        u(0, x) ~ sin(π * x),
        Dx(u(t, 0)) ~ π * exp(-t),          # Left Neumann
        u(t, 1) ~ 0.0,                       # Right Dirichlet
    ]

    domains = [
        t ∈ Interval(0.0, 0.5),
        x ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    dx = 0.05
    disc_scalar = MOLFiniteDifference([x => dx], t;
        advection_scheme = UpwindScheme(),
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([x => dx], t;
        advection_scheme = UpwindScheme(),
        discretization_strategy = ArrayDiscretization()
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

# --- 11k: Neumann/Robin symbolic structure ---

@testset "Neumann/Robin BC: ArrayOp symbolic structure" begin
    using SymbolicUtils
    using Symbolics: unwrap

    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    eq = Dt(u(t, x)) ~ Dxx(u(t, x))

    bcs = [
        u(0, x) ~ cos(x),
        Dx(u(t, 0)) ~ 0.0,
        u(t, Float64(π)) ~ -exp(-t),
    ]

    domains = [
        t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, Float64(π)),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    dx = 0.25
    disc = MOLFiniteDifference([x => dx], t;
        discretization_strategy = ArrayDiscretization()
    )

    sys, tspan = MethodOfLines.symbolic_discretize(pdesys, disc)
    eqs = equations(sys)

    has_arrayop = any(eqs) do eq
        lhs_raw = unwrap(eq.lhs)
        rhs_raw = unwrap(eq.rhs)
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(lhs_raw)) ||
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(rhs_raw))
    end
    @test has_arrayop
end

# ===========================================================================
# Phase 12: 2D/3D ArrayOp validation and Periodic BC ArrayOp tests
# ===========================================================================

# --- 12a: 2D upwind advection-diffusion ---

@testset "ArrayDiscretization: 2D upwind advection-diffusion matches scalar" begin
    @parameters t x y
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    eq = Dt(u(t, x, y)) ~ -Dx(u(t, x, y)) + 0.1 * (Dxx(u(t, x, y)) + Dyy(u(t, x, y)))

    bcs = [
        u(0.0, x, y) ~ sin(pi * x) * sin(pi * y),
        u(t, 0.0, y) ~ 0.0,
        u(t, 1.0, y) ~ 0.0,
        u(t, x, 0.0) ~ 0.0,
        u(t, x, 1.0) ~ 0.0,
    ]

    domains = [
        t ∈ Interval(0.0, 0.3),
        x ∈ Interval(0.0, 1.0),
        y ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem([eq], bcs, domains, [t, x, y], [u(t, x, y)])

    dx = 0.1

    disc_scalar = MOLFiniteDifference([x => dx, y => dx], t;
        advection_scheme = UpwindScheme(),
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([x => dx, y => dx], t;
        advection_scheme = UpwindScheme(),
        discretization_strategy = ArrayDiscretization()
    )

    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)

    sol_scalar = solve(prob_scalar, Tsit5(), saveat = 0.1)
    sol_array = solve(prob_array, Tsit5(), saveat = 0.1)

    u_scalar = sol_scalar[u(t, x, y)]
    u_array = sol_array[u(t, x, y)]

    @test size(u_scalar) == size(u_array)
    @test isapprox(u_scalar, u_array, rtol = 1e-10)
end

# --- 12b: 2D coupled system ---

@testset "ArrayDiscretization: 2D coupled diffusion matches scalar" begin
    @parameters t x y
    @variables u(..) v(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    eqs = [
        Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y)) + v(t, x, y),
        Dt(v(t, x, y)) ~ Dxx(v(t, x, y)) + Dyy(v(t, x, y)) + u(t, x, y),
    ]

    bcs = [
        u(0.0, x, y) ~ sin(pi * x) * sin(pi * y),
        v(0.0, x, y) ~ 0.0,
        u(t, 0.0, y) ~ 0.0,
        u(t, 1.0, y) ~ 0.0,
        u(t, x, 0.0) ~ 0.0,
        u(t, x, 1.0) ~ 0.0,
        v(t, 0.0, y) ~ 0.0,
        v(t, 1.0, y) ~ 0.0,
        v(t, x, 0.0) ~ 0.0,
        v(t, x, 1.0) ~ 0.0,
    ]

    domains = [
        t ∈ Interval(0.0, 0.3),
        x ∈ Interval(0.0, 1.0),
        y ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem(eqs, bcs, domains, [t, x, y], [u(t, x, y), v(t, x, y)])

    dx = 0.1

    disc_scalar = MOLFiniteDifference([x => dx, y => dx], t;
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([x => dx, y => dx], t;
        discretization_strategy = ArrayDiscretization()
    )

    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)

    sol_scalar = solve(prob_scalar, Tsit5(), saveat = 0.1)
    sol_array = solve(prob_array, Tsit5(), saveat = 0.1)

    u_scalar = sol_scalar[u(t, x, y)]
    u_array = sol_array[u(t, x, y)]
    v_scalar = sol_scalar[v(t, x, y)]
    v_array = sol_array[v(t, x, y)]

    @test size(u_scalar) == size(u_array)
    @test isapprox(u_scalar, u_array, rtol = 1e-10)
    @test size(v_scalar) == size(v_array)
    @test isapprox(v_scalar, v_array, rtol = 1e-10)
end

# --- 12c: 2D symbolic structure with ranges check ---

@testset "ArrayOp template: 2D symbolic structure with flattening count" begin
    using SymbolicUtils
    using Symbolics: unwrap
    using ModelingToolkit.ModelingToolkitBase: flatten_equations

    @parameters t x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    Dt = Differential(t)

    eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y))

    bcs = [
        u(0.0, x, y) ~ sin(pi * x) * sin(pi * y),
        u(t, 0.0, y) ~ 0.0,
        u(t, 1.0, y) ~ 0.0,
        u(t, x, 0.0) ~ 0.0,
        u(t, x, 1.0) ~ 0.0,
    ]

    domains = [
        t ∈ Interval(0.0, 0.5),
        x ∈ Interval(0.0, 1.0),
        y ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem([eq], bcs, domains, [t, x, y], [u(t, x, y)])

    # dx=dy=0.2 => 6 points per dim, interior [2,5] x [2,5] = 4x4 = 16
    dx = 0.2
    disc = MOLFiniteDifference([x => dx, y => dx], t;
        discretization_strategy = ArrayDiscretization()
    )

    sys, tspan = MethodOfLines.symbolic_discretize(pdesys, disc)
    eqs = equations(sys)

    # Verify ArrayOp presence
    has_arrayop = any(eqs) do eq
        rhs_raw = unwrap(eq.rhs)
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(rhs_raw))
    end
    @test has_arrayop

    # After flattening: 4*4 interior + boundary equations
    flat = flatten_equations(eqs)
    @test length(flat) >= 16  # At least the interior points
end

# --- 12d: Periodic BC ArrayOp symbolic structure (should use ArrayOp now) ---

@testset "Periodic BC: ArrayOp template used (not per-point fallback)" begin
    using SymbolicUtils
    using Symbolics: unwrap
    using ModelingToolkit.ModelingToolkitBase: flatten_equations

    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    L = 2.0
    eq = Dt(u(t, x)) ~ Dxx(u(t, x))

    bcs = [
        u(0, x) ~ sin(2π * x / L),
        u(t, 0) ~ u(t, L),  # Periodic BC
    ]

    domains = [
        t ∈ Interval(0.0, 0.5),
        x ∈ Interval(0.0, L),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    dx = 0.25
    disc = MOLFiniteDifference([x => dx], t;
        discretization_strategy = ArrayDiscretization()
    )

    sys, tspan = MethodOfLines.symbolic_discretize(pdesys, disc)
    eqs = equations(sys)

    # With periodic ArrayOp support, should now use ArrayOp (not per-point fallback)
    has_arrayop = any(eqs) do eq
        lhs_raw = unwrap(eq.lhs)
        rhs_raw = unwrap(eq.rhs)
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(lhs_raw)) ||
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(rhs_raw))
    end
    @test has_arrayop

    # After flattening, should have the correct number of equations
    flat = flatten_equations(eqs)
    @test length(flat) >= 7  # At least the interior + boundary
end

# --- 12e: Periodic BC centered diffusion with ArrayOp numerical verification ---

@testset "Periodic BC: centered diffusion ArrayOp numerical match" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    L = 2.0
    eq = Dt(u(t, x)) ~ Dxx(u(t, x))

    bcs = [
        u(0, x) ~ sin(2π * x / L),
        u(t, 0) ~ u(t, L),  # Periodic BC
    ]

    domains = [
        t ∈ Interval(0.0, 0.5),
        x ∈ Interval(0.0, L),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    dx = 0.1
    disc_scalar = MOLFiniteDifference([x => dx], t;
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([x => dx], t;
        discretization_strategy = ArrayDiscretization()
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

# --- 12f: Periodic WENO ArrayOp path ---

@testset "Periodic BC: WENO advection ArrayOp uses template" begin
    using SymbolicUtils
    using Symbolics: unwrap

    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    L = 2.0
    eq = Dt(u(t, x)) ~ -Dx(u(t, x))

    bcs = [
        u(0, x) ~ sin(2π * x / L),
        u(t, 0) ~ u(t, L),  # Periodic BC
    ]

    domains = [
        t ∈ Interval(0.0, 0.3),
        x ∈ Interval(0.0, L),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    dx = 0.1
    disc = MOLFiniteDifference([x => dx], t;
        advection_scheme = WENOScheme(),
        discretization_strategy = ArrayDiscretization()
    )

    sys, tspan = MethodOfLines.symbolic_discretize(pdesys, disc)
    eqs = equations(sys)

    # With periodic support, should use ArrayOp template
    has_arrayop = any(eqs) do eq
        lhs_raw = unwrap(eq.lhs)
        rhs_raw = unwrap(eq.rhs)
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(lhs_raw)) ||
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(rhs_raw))
    end
    @test has_arrayop
end

# --- 12g: Periodic upwind ArrayOp path ---

@testset "Periodic BC: upwind advection ArrayOp uses template" begin
    using SymbolicUtils
    using Symbolics: unwrap

    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    L = 2.0
    eq = Dt(u(t, x)) ~ -Dx(u(t, x))

    bcs = [
        u(0, x) ~ sin(2π * x / L),
        u(t, 0) ~ u(t, L),  # Periodic BC
    ]

    domains = [
        t ∈ Interval(0.0, 0.3),
        x ∈ Interval(0.0, L),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    dx = 0.1
    disc = MOLFiniteDifference([x => dx], t;
        advection_scheme = UpwindScheme(),
        discretization_strategy = ArrayDiscretization()
    )

    sys, tspan = MethodOfLines.symbolic_discretize(pdesys, disc)
    eqs = equations(sys)

    # With periodic support, should use ArrayOp template
    has_arrayop = any(eqs) do eq
        lhs_raw = unwrap(eq.lhs)
        rhs_raw = unwrap(eq.rhs)
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(lhs_raw)) ||
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(rhs_raw))
    end
    @test has_arrayop
end

# --- 12h: 2D periodic diffusion ---

@testset "Periodic BC: 2D diffusion ArrayOp matches scalar" begin
    @parameters t x y
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    Lx = 2.0
    Ly = 2.0
    eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y))

    bcs = [
        u(0, x, y) ~ sin(2π * x / Lx) * sin(2π * y / Ly),
        u(t, 0, y) ~ u(t, Lx, y),  # Periodic in x
        u(t, x, 0) ~ u(t, x, Ly),  # Periodic in y
    ]

    domains = [
        t ∈ Interval(0.0, 0.3),
        x ∈ Interval(0.0, Lx),
        y ∈ Interval(0.0, Ly),
    ]

    @named pdesys = PDESystem([eq], bcs, domains, [t, x, y], [u(t, x, y)])

    dx = 0.2

    disc_scalar = MOLFiniteDifference([x => dx, y => dx], t;
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([x => dx, y => dx], t;
        discretization_strategy = ArrayDiscretization()
    )

    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)

    sol_scalar = solve(prob_scalar, Tsit5(), saveat = 0.1)
    sol_array = solve(prob_array, Tsit5(), saveat = 0.1)

    u_scalar = sol_scalar[u(t, x, y)]
    u_array = sol_array[u(t, x, y)]

    @test size(u_scalar) == size(u_array)
    @test isapprox(u_scalar, u_array, rtol = 1e-10)
end
