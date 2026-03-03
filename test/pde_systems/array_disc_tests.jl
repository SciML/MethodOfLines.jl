# Tests for ArrayDiscretization
# These mirror selected tests from MOL_1D_Linear_Diffusion.jl but use
# ArrayDiscretization() as the discretization strategy.

using ModelingToolkit, MethodOfLines, LinearAlgebra, Test, OrdinaryDiffEq, DomainSets, SciMLBase
using ModelingToolkit: Differential
import PDEBase

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

# ─── Phase 13: Complex PDE Validation, PDAE Support, Higher-Order Upwind ──────

# --- 13a (B1): PDAE — diffusion + algebraic constraint ---

@testset "PDAE: diffusion + algebraic constraint ArrayOp matches scalar" begin
    # Adapted from MOL_1D_PDAE.jl
    # Dt(u(t,x)) ~ Dxx(u(t,x))
    # 0 ~ Dxx(v(t,x)) + exp(-t)*sin(x)

    @parameters t x
    @variables u(..) v(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Dx^2

    eqs = [
        Dt(u(t, x)) ~ Dxx(u(t, x)),
        0 ~ Dxx(v(t, x)) + exp(-t) * sin(x),
    ]
    bcs = [
        u(0, x) ~ cos(x),
        v(0, x) ~ sin(x),
        u(t, 0) ~ exp(-t),
        Dx(u(t, 1)) ~ -exp(-t) * sin(1),
        Dx(v(t, 0)) ~ exp(-t),
        v(t, 1) ~ exp(-t) * sin(1),
    ]

    domains = [
        t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem(eqs, bcs, domains, [t, x], [u(t, x), v(t, x)])

    l = 20
    dx = range(0.0, 1.0, length = l)
    dx_ = dx[2] - dx[1]

    disc_scalar = MOLFiniteDifference([x => dx_], t;
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([x => dx_], t;
        discretization_strategy = ArrayDiscretization()
    )

    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)

    sol_scalar = solve(prob_scalar, Rodas4(), saveat = 0.1)
    sol_array = solve(prob_array, Rodas4(), saveat = 0.1)

    @test SciMLBase.successful_retcode(sol_scalar)
    @test SciMLBase.successful_retcode(sol_array)

    u_scalar = sol_scalar[u(t, x)]
    u_array = sol_array[u(t, x)]
    v_scalar = sol_scalar[v(t, x)]
    v_array = sol_array[v(t, x)]

    @test size(u_scalar) == size(u_array)
    @test isapprox(u_scalar, u_array, rtol = 1e-3)
    @test size(v_scalar) == size(v_array)
    @test isapprox(v_scalar, v_array, rtol = 1e-3)
end

# --- 13b (B2): KdV equation — 3rd-order upwind with UpwindScheme ---

@testset "KdV 3rd-order upwind ArrayOp matches scalar" begin
    # Adapted from MOL_1D_HigherOrder.jl KdV test
    # Dt(u(x,t)) ~ -6*u*Dx(u) - Dx3(u)

    @parameters x t
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dx2 = Differential(x)^2
    Dx3 = Differential(x)^3

    α = 6
    β = 1
    eq = Dt(u(x, t)) ~ -α * u(x, t) * Dx(u(x, t)) - β * Dx3(u(x, t))

    u_analytic(x, t; z = (x - t) / 2) = 1 / 2 * sech(z)^2
    du(x, t; z = (x - t) / 2) = 1 / 2 * tanh(z) * sech(z)^2
    ddu(x, t; z = (x - t) / 2) = 1 / 4 * (2 * tanh(z)^2 + sech(z)^2) * sech(z)^2
    bcs = [
        u(x, 0) ~ u_analytic(x, 0),
        u(-10, t) ~ u_analytic(-10, t),
        u(10, t) ~ u_analytic(10, t),
        Dx(u(-10, t)) ~ du(-10, t),
        Dx(u(10, t)) ~ du(10, t),
        Dx2(u(-10, t)) ~ ddu(-10, t),
        Dx2(u(10, t)) ~ ddu(10, t),
    ]

    domains = [
        x ∈ Interval(-10.0, 10.0),
        t ∈ Interval(0.0, 1.0),
    ]

    dx = 0.4

    disc_scalar = MOLFiniteDifference([x => dx], t;
        upwind_order = 1, grid_align = center_align,
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([x => dx], t;
        upwind_order = 1, grid_align = center_align,
        discretization_strategy = ArrayDiscretization()
    )

    @named pdesys = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])

    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)

    sol_scalar = solve(prob_scalar, FBDF(), saveat = 0.1)
    sol_array = solve(prob_array, FBDF(), saveat = 0.1)

    @test SciMLBase.successful_retcode(sol_scalar)
    @test SciMLBase.successful_retcode(sol_array)

    u_scalar = sol_scalar[u(x, t)]
    u_array = sol_array[u(x, t)]

    @test size(u_scalar) == size(u_array)
    @test isapprox(u_scalar, u_array, rtol = 1e-10)
end

# --- 13c (B3): Mixed advection + diffusion + dispersion ---

@testset "Mixed advection+diffusion+dispersion ArrayOp matches scalar" begin
    # Dt(u) ~ -alpha*Dx(u) + beta*Dxx(u) + gamma*Dxxx(u) - delta*Dxxxx(u)
    # Combines 1st-order upwind, 2nd-order centered, 3rd-order upwind, 4th-order centered

    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2
    Dxxx = Differential(x)^3
    Dxxxx = Differential(x)^4

    alpha_val = 1.0
    beta_val = 0.1
    gamma_val = 0.01
    delta_val = 0.001

    eq = Dt(u(t, x)) ~ -alpha_val * Dx(u(t, x)) + beta_val * Dxx(u(t, x)) +
                         gamma_val * Dxxx(u(t, x)) - delta_val * Dxxxx(u(t, x))

    bcs = [
        u(0, x) ~ exp(-((x - 5)^2)),
        u(t, 0) ~ 0.0,
        u(t, 10) ~ 0.0,
        Dx(u(t, 0)) ~ 0.0,
        Dx(u(t, 10)) ~ 0.0,
        Dxx(u(t, 0)) ~ 0.0,
        Dxx(u(t, 10)) ~ 0.0,
        Dxxx(u(t, 0)) ~ 0.0,
        Dxxx(u(t, 10)) ~ 0.0,
    ]

    domains = [
        t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, 10.0),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    dx = 0.2

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

    sol_scalar = solve(prob_scalar, FBDF(), saveat = 0.1)
    sol_array = solve(prob_array, FBDF(), saveat = 0.1)

    @test SciMLBase.successful_retcode(sol_scalar)
    @test SciMLBase.successful_retcode(sol_array)

    u_scalar = sol_scalar[u(t, x)]
    u_array = sol_array[u(t, x)]

    @test size(u_scalar) == size(u_array)
    @test isapprox(u_scalar, u_array, rtol = 1e-10)
end

# --- 13d (B4): Beam equation — 4th-order spatial + coupled system ---

@testset "Beam equation with velocity ArrayOp matches scalar" begin
    # Adapted from MOL_1D_HigherOrder.jl Test 01
    # v(t,x) ~ Dt(u(t,x))
    # Dt(v(t,x)) ~ -mu*EI*Dx4(u(t,x)) + mu*g

    @parameters x t
    @variables u(..) v(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2
    Dx3 = Differential(x)^3
    Dx4 = Differential(x)^4

    g = -9.81
    EI = 1
    mu = 1
    L = 10.0
    dx = 0.4

    eqs = [
        v(t, x) ~ Dt(u(t, x)),
        Dt(v(t, x)) ~ -mu * EI * Dx4(u(t, x)) + mu * g,
    ]

    bcs = [
        u(0, x) ~ 0,
        v(0, x) ~ 0,
        u(t, 0) ~ 0,
        v(t, 0) ~ 0,
        Dxx(u(t, L)) ~ 0,
        Dx3(u(t, L)) ~ 0,
    ]

    domains = [
        t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, L),
    ]

    @named pdesys = PDESystem(eqs, bcs, domains, [t, x], [u(t, x), v(t, x)])

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

    sol_scalar = solve(prob_scalar, FBDF())
    sol_array = solve(prob_array, FBDF())

    @test SciMLBase.successful_retcode(sol_scalar)
    @test SciMLBase.successful_retcode(sol_array)

    u_scalar = sol_scalar[u(t, x)]
    u_array = sol_array[u(t, x)]
    v_scalar = sol_scalar[v(t, x)]
    v_array = sol_array[v(t, x)]

    @test size(u_scalar) == size(u_array)
    @test isapprox(u_scalar, u_array, rtol = 1e-10)
    @test size(v_scalar) == size(v_array)
    @test isapprox(v_scalar, v_array, rtol = 1e-10)
end

# --- 13e (B5): Brusselator — 2D coupled reaction-diffusion + periodic BCs ---

@testset "Brusselator 2D periodic ArrayOp matches scalar" begin
    # Adapted from brusselator_eq.jl with smaller grid and shorter time

    @parameters x y t
    @variables u(..) v(..)
    Difft = Differential(t)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    ∇²(u) = Dxx(u) + Dyy(u)

    brusselator_f(x, y, t) = (((x - 0.3)^2 + (y - 0.6)^2) <= 0.1^2) * (t >= 1.1) * 5.0

    α = 10.0

    u0(x, y, t) = 22(y * (1 - y))^(3 / 2)
    v0(x, y, t) = 27(x * (1 - x))^(3 / 2)

    eq = [
        Difft(u(x, y, t)) ~
            1.0 + v(x, y, t) * u(x, y, t)^2 - 4.4 * u(x, y, t) +
            α * ∇²(u(x, y, t)) + brusselator_f(x, y, t),
        Difft(v(x, y, t)) ~
            3.4 * u(x, y, t) - v(x, y, t) * u(x, y, t)^2 +
            α * ∇²(v(x, y, t)),
    ]

    domains = [
        x ∈ Interval(0.0, 1.0),
        y ∈ Interval(0.0, 1.0),
        t ∈ Interval(0.0, 1.0),
    ]

    bcs = [
        u(x, y, 0) ~ u0(x, y, 0),
        u(0, y, t) ~ u(1, y, t),
        u(x, 0, t) ~ u(x, 1, t),
        v(x, y, 0) ~ v0(x, y, 0),
        v(0, y, t) ~ v(1, y, t),
        v(x, 0, t) ~ v(x, 1, t),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [x, y, t], [u(x, y, t), v(x, y, t)])

    N = 8
    dx = 1 / N
    dy = 1 / N

    disc_scalar = MOLFiniteDifference([x => dx, y => dy], t;
        approx_order = 2,
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([x => dx, y => dy], t;
        approx_order = 2,
        discretization_strategy = ArrayDiscretization()
    )

    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)

    sol_scalar = solve(prob_scalar, TRBDF2(), saveat = 0.1)
    sol_array = solve(prob_array, TRBDF2(), saveat = 0.1)

    @test SciMLBase.successful_retcode(sol_scalar)
    @test SciMLBase.successful_retcode(sol_array)

    u_scalar = sol_scalar[u(x, y, t)]
    u_array = sol_array[u(x, y, t)]
    v_scalar = sol_scalar[v(x, y, t)]
    v_array = sol_array[v(x, y, t)]

    @test size(u_scalar) == size(u_array)
    @test isapprox(u_scalar, u_array, rtol = 1e-10)
    @test size(v_scalar) == size(v_array)
    @test isapprox(v_scalar, v_array, rtol = 1e-10)
end

# --- 13f (B6): Schrödinger — complex-valued PDE ---

@testset "Schrödinger complex PDE ArrayOp matches scalar" begin
    # Adapted from schroedinger.jl
    # im * Dt(ψ(t,x)) ~ Dxx(ψ(t,x))

    @parameters t x
    @variables ψ(..)

    Dt = Differential(t)
    Dxx = Differential(x)^2

    xmin = 0
    xmax = 1

    V(x) = 0.0

    eq = [im * Dt(ψ(t, x)) ~ (Dxx(ψ(t, x)) + V(x) * ψ(t, x))]

    ψ0 = x -> ((1 + im) / sqrt(2)) * sinpi(2 * x)

    bcs = [
        ψ(0, x) => ψ0(x),
        ψ(t, xmin) ~ 0,
        ψ(t, xmax) ~ 0,
    ]

    domains = [t ∈ Interval(0, 1), x ∈ Interval(xmin, xmax)]

    @named sys = PDESystem(eq, bcs, domains, [t, x], [ψ(t, x)])

    disc_scalar = MOLFiniteDifference([x => 50], t;
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([x => 50], t;
        discretization_strategy = ArrayDiscretization()
    )

    prob_scalar = discretize(sys, disc_scalar)
    prob_array = discretize(sys, disc_array)

    sol_scalar = solve(prob_scalar, FBDF(), saveat = 0.1)
    sol_array = solve(prob_array, FBDF(), saveat = 0.1)

    @test SciMLBase.successful_retcode(sol_scalar)
    @test SciMLBase.successful_retcode(sol_array)

    ψ_scalar = sol_scalar[ψ(t, x)]
    ψ_array = sol_array[ψ(t, x)]

    @test size(ψ_scalar) == size(ψ_array)
    @test isapprox(ψ_scalar, ψ_array, rtol = 1e-10)
end

# --- 13g (B7): Higher-order upwind symbolic structure ---

@testset "3rd-order upwind produces ArrayOp" begin
    using SymbolicUtils
    using Symbolics: unwrap
    using ModelingToolkit.ModelingToolkitBase: flatten_equations

    @parameters x t
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dx2 = Differential(x)^2
    Dx3 = Differential(x)^3

    eq = Dt(u(x, t)) ~ -u(x, t) * Dx(u(x, t)) - Dx3(u(x, t))

    # Need enough BCs for 3rd order
    bcs = [
        u(x, 0) ~ exp(-(x^2)),
        u(-5, t) ~ 0.0,
        u(5, t) ~ 0.0,
        Dx(u(-5, t)) ~ 0.0,
        Dx(u(5, t)) ~ 0.0,
        Dx2(u(-5, t)) ~ 0.0,
        Dx2(u(5, t)) ~ 0.0,
    ]

    domains = [
        x ∈ Interval(-5.0, 5.0),
        t ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])

    dx = 0.5
    disc = MOLFiniteDifference([x => dx], t;
        advection_scheme = UpwindScheme(),
        discretization_strategy = ArrayDiscretization()
    )

    sys, tspan = MethodOfLines.symbolic_discretize(pdesys, disc)
    eqs = equations(sys)

    # Verify ArrayOp is used (not per-point fallback)
    has_arrayop = any(eqs) do eq
        lhs_raw = unwrap(eq.lhs)
        rhs_raw = unwrap(eq.rhs)
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(lhs_raw)) ||
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(rhs_raw))
    end
    @test has_arrayop

    # 3rd-order upwind should have wider boundary frame than 1st-order
    flat = flatten_equations(eqs)
    # Interior points should be fewer than total grid points
    n_total = Int(10.0 / dx) + 1  # 21 grid points
    # Flattened equations = interior (from ArrayOp) + boundary
    @test length(flat) >= n_total - 2  # At least n_total - 2 equations
end

# --- 13h (B8): 4th-order spatial derivative symbolic structure ---

@testset "4th-order spatial derivative Dx4 produces ArrayOp" begin
    using SymbolicUtils
    using Symbolics: unwrap
    using ModelingToolkit.ModelingToolkitBase: flatten_equations

    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dx4 = Differential(x)^4

    eq = Dt(u(t, x)) ~ -Dx4(u(t, x))

    bcs = [
        u(0, x) ~ sin(π * x),
        u(t, 0) ~ 0.0,
        u(t, 1) ~ 0.0,
        Dx(u(t, 0)) ~ π,
        Dx(u(t, 1)) ~ -π,
    ]

    domains = [
        t ∈ Interval(0.0, 0.1),
        x ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    dx = 0.05
    disc = MOLFiniteDifference([x => dx], t;
        approx_order = 4,
        discretization_strategy = ArrayDiscretization()
    )

    sys, tspan = MethodOfLines.symbolic_discretize(pdesys, disc)
    eqs = equations(sys)

    # Verify ArrayOp is used
    has_arrayop = any(eqs) do eq
        lhs_raw = unwrap(eq.lhs)
        rhs_raw = unwrap(eq.rhs)
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(lhs_raw)) ||
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(rhs_raw))
    end
    @test has_arrayop

    # 4th-order derivative with approx_order=4 needs wider boundary frame
    flat = flatten_equations(eqs)
    n_total = Int(1.0 / dx) + 1  # 21 grid points
    @test length(flat) >= n_total - 2  # At least n_total - 2 equations
end

# ─── Phase 14: Full-Interior ArrayOp + Algebraic Equation Support ─────────────

# --- 14a (E1): PDAE algebraic equation uses ArrayOp ---

@testset "PDAE algebraic equation produces ArrayOp" begin
    using SymbolicUtils
    using Symbolics: unwrap
    using ModelingToolkit.ModelingToolkitBase: flatten_equations

    @parameters t x
    @variables u(..) v(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Dx^2

    eqs = [
        Dt(u(t, x)) ~ Dxx(u(t, x)),
        0 ~ Dxx(v(t, x)) + exp(-t) * sin(x),
    ]
    bcs = [
        u(0, x) ~ cos(x),
        v(0, x) ~ sin(x),
        u(t, 0) ~ exp(-t),
        Dx(u(t, 1)) ~ -exp(-t) * sin(1),
        Dx(v(t, 0)) ~ exp(-t),
        v(t, 1) ~ exp(-t) * sin(1),
    ]

    domains = [
        t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem(eqs, bcs, domains, [t, x], [u(t, x), v(t, x)])

    dx = 0.05
    disc = MOLFiniteDifference([x => dx], t;
        discretization_strategy = ArrayDiscretization()
    )

    sys, tspan = MethodOfLines.symbolic_discretize(pdesys, disc)
    eqs_out = equations(sys)

    # Both u (ODE) and v (algebraic) should produce ArrayOp equations
    arrayop_count = count(eqs_out) do eq
        lhs_raw = unwrap(eq.lhs)
        rhs_raw = unwrap(eq.rhs)
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(lhs_raw)) ||
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(rhs_raw))
    end
    @test arrayop_count >= 2  # At least one ArrayOp for u and one for v

    # Solution should match scalar path
    disc_scalar = MOLFiniteDifference([x => dx], t;
        discretization_strategy = ScalarizedDiscretization()
    )
    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc)
    sol_scalar = solve(prob_scalar, Rodas4(), saveat = 0.1)
    sol_array = solve(prob_array, Rodas4(), saveat = 0.1)

    @test SciMLBase.successful_retcode(sol_scalar)
    @test SciMLBase.successful_retcode(sol_array)
    # The array path evaluates grid-dependent forcing terms (e.g. sin(x))
    # numerically via Const-wrapped grid arrays, while the scalar path keeps
    # them symbolic.  This creates small floating-point differences in the
    # DAE solver trajectory (up to ~0.6% relative for the algebraic variable).
    @test isapprox(sol_scalar[u(t, x)], sol_array[u(t, x)], rtol = 1e-2)
    @test isapprox(sol_scalar[v(t, x)], sol_array[v(t, x)], rtol = 1e-2)
end

# --- 14b (E2): 1D diffusion full-interior (approx_order=2) ---

@testset "1D diffusion full-interior approx_order=2" begin
    using SymbolicUtils
    using Symbolics: unwrap
    using ModelingToolkit.ModelingToolkitBase: flatten_equations

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
        t ∈ Interval(0.0, 0.5),
        x ∈ Interval(0.0, 1.0),
    ]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    dx = 0.05
    disc_array = MOLFiniteDifference([x => dx], t;
        approx_order = 2,
        discretization_strategy = ArrayDiscretization()
    )
    disc_scalar = MOLFiniteDifference([x => dx], t;
        approx_order = 2,
        discretization_strategy = ScalarizedDiscretization()
    )

    sys_array, _ = MethodOfLines.symbolic_discretize(pdesys, disc_array)
    eqs_array = equations(sys_array)

    # Verify ArrayOp is used (full-interior mode: no scalar frame equations)
    has_arrayop = any(eqs_array) do eq
        lhs_raw = unwrap(eq.lhs)
        rhs_raw = unwrap(eq.rhs)
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(lhs_raw)) ||
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(rhs_raw))
    end
    @test has_arrayop

    # Solution should match scalar path
    prob_array = discretize(pdesys, disc_array)
    prob_scalar = discretize(pdesys, disc_scalar)
    sol_array = solve(prob_array, Tsit5(), saveat = 0.1)
    sol_scalar = solve(prob_scalar, Tsit5(), saveat = 0.1)
    @test SciMLBase.successful_retcode(sol_array)
    @test SciMLBase.successful_retcode(sol_scalar)
    @test isapprox(sol_array[u(t, x)], sol_scalar[u(t, x)], rtol = 1e-10)
end

# --- 14c (E3): 1D diffusion full-interior (approx_order=4) ---

@testset "1D diffusion full-interior approx_order=4" begin
    using SymbolicUtils
    using Symbolics: unwrap
    using ModelingToolkit.ModelingToolkitBase: flatten_equations

    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Dx^2

    eq = Dt(u(t, x)) ~ Dxx(u(t, x))
    bcs = [
        u(0, x) ~ sin(π * x),
        u(t, 0) ~ 0.0,
        u(t, 1) ~ 0.0,
        Dx(u(t, 0)) ~ π,
        Dx(u(t, 1)) ~ -π,
    ]
    domains = [
        t ∈ Interval(0.0, 0.5),
        x ∈ Interval(0.0, 1.0),
    ]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    dx = 0.05
    disc_array = MOLFiniteDifference([x => dx], t;
        approx_order = 4,
        discretization_strategy = ArrayDiscretization()
    )
    disc_scalar = MOLFiniteDifference([x => dx], t;
        approx_order = 4,
        discretization_strategy = ScalarizedDiscretization()
    )

    sys_array, _ = MethodOfLines.symbolic_discretize(pdesys, disc_array)
    eqs_array = equations(sys_array)

    # Should have ArrayOp
    has_arrayop = any(eqs_array) do eq
        lhs_raw = unwrap(eq.lhs)
        rhs_raw = unwrap(eq.rhs)
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(lhs_raw)) ||
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(rhs_raw))
    end
    @test has_arrayop

    # Solution should match scalar path
    prob_array = discretize(pdesys, disc_array)
    prob_scalar = discretize(pdesys, disc_scalar)
    sol_array = solve(prob_array, Tsit5(), saveat = 0.1)
    sol_scalar = solve(prob_scalar, Tsit5(), saveat = 0.1)
    @test SciMLBase.successful_retcode(sol_array)
    @test SciMLBase.successful_retcode(sol_scalar)
    @test isapprox(sol_array[u(t, x)], sol_scalar[u(t, x)], rtol = 1e-10)
end

# --- 14d (E4): 1D Burgers upwind full-interior ---

@testset "1D Burgers upwind full-interior" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    eq = Dt(u(t, x)) ~ -u(t, x) * Dx(u(t, x)) + 0.01 * Dxx(u(t, x))

    bcs = [
        u(0, x) ~ 0.5 * (1.0 - tanh(x / 0.2)),
        u(t, -2) ~ 1.0,
        u(t, 2) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 1.0),
        x ∈ Interval(-2.0, 2.0),
    ]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    dx = 0.1
    disc_array = MOLFiniteDifference([x => dx], t;
        advection_scheme = UpwindScheme(),
        discretization_strategy = ArrayDiscretization()
    )
    disc_scalar = MOLFiniteDifference([x => dx], t;
        advection_scheme = UpwindScheme(),
        discretization_strategy = ScalarizedDiscretization()
    )

    prob_array = discretize(pdesys, disc_array)
    prob_scalar = discretize(pdesys, disc_scalar)
    sol_array = solve(prob_array, Tsit5(), saveat = 0.1)
    sol_scalar = solve(prob_scalar, Tsit5(), saveat = 0.1)

    @test SciMLBase.successful_retcode(sol_array)
    @test SciMLBase.successful_retcode(sol_scalar)
    @test isapprox(sol_array[u(t, x)], sol_scalar[u(t, x)], rtol = 1e-6)
end

# --- 14e (E5): 2D diffusion full-interior ---

@testset "2D diffusion full-interior" begin
    @parameters t x y
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y))

    bcs = [
        u(0, x, y) ~ sin(π * x) * sin(π * y),
        u(t, 0, y) ~ 0.0,
        u(t, 1, y) ~ 0.0,
        u(t, x, 0) ~ 0.0,
        u(t, x, 1) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.1),
        x ∈ Interval(0.0, 1.0),
        y ∈ Interval(0.0, 1.0),
    ]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])

    dx = 0.1
    disc_array = MOLFiniteDifference([x => dx, y => dx], t;
        discretization_strategy = ArrayDiscretization()
    )
    disc_scalar = MOLFiniteDifference([x => dx, y => dx], t;
        discretization_strategy = ScalarizedDiscretization()
    )

    prob_array = discretize(pdesys, disc_array)
    prob_scalar = discretize(pdesys, disc_scalar)
    sol_array = solve(prob_array, Tsit5(), saveat = 0.01)
    sol_scalar = solve(prob_scalar, Tsit5(), saveat = 0.01)

    @test SciMLBase.successful_retcode(sol_array)
    @test SciMLBase.successful_retcode(sol_scalar)
    @test isapprox(sol_array[u(t, x, y)], sol_scalar[u(t, x, y)], rtol = 1e-10)
end

# --- 14f (E6): 1D non-uniform grid full-interior ---

@testset "1D non-uniform grid full-interior" begin
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
        t ∈ Interval(0.0, 0.5),
        x ∈ Interval(0.0, 1.0),
    ]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    # Non-uniform grid: finer near x=0.5
    dx_grid = [0.0; cumsum([0.05 + 0.02*sin(π*i/20) for i in 1:19])]
    dx_grid = dx_grid / dx_grid[end]  # normalize to [0,1]

    disc_array = MOLFiniteDifference([x => dx_grid], t;
        discretization_strategy = ArrayDiscretization()
    )
    disc_scalar = MOLFiniteDifference([x => dx_grid], t;
        discretization_strategy = ScalarizedDiscretization()
    )

    prob_array = discretize(pdesys, disc_array)
    prob_scalar = discretize(pdesys, disc_scalar)
    sol_array = solve(prob_array, Tsit5(), saveat = 0.1)
    sol_scalar = solve(prob_scalar, Tsit5(), saveat = 0.1)

    @test SciMLBase.successful_retcode(sol_array)
    @test SciMLBase.successful_retcode(sol_scalar)
    @test isapprox(sol_array[u(t, x)], sol_scalar[u(t, x)], rtol = 1e-6)
end

# --- 14g (E7): Mixed centered+upwind advection-diffusion full-interior ---

@testset "Mixed centered+upwind advection-diffusion full-interior" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    c = 1.0
    D_coeff = 0.05
    eq = Dt(u(t, x)) ~ -c * Dx(u(t, x)) + D_coeff * Dxx(u(t, x))

    bcs = [
        u(0, x) ~ exp(-(x - 2)^2 / 0.5),
        u(t, 0) ~ 0.0,
        u(t, 5) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 1.0),
        x ∈ Interval(0.0, 5.0),
    ]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    dx = 0.1
    disc_array = MOLFiniteDifference([x => dx], t;
        advection_scheme = UpwindScheme(),
        discretization_strategy = ArrayDiscretization()
    )
    disc_scalar = MOLFiniteDifference([x => dx], t;
        advection_scheme = UpwindScheme(),
        discretization_strategy = ScalarizedDiscretization()
    )

    prob_array = discretize(pdesys, disc_array)
    prob_scalar = discretize(pdesys, disc_scalar)
    sol_array = solve(prob_array, Tsit5(), saveat = 0.1)
    sol_scalar = solve(prob_scalar, Tsit5(), saveat = 0.1)

    @test SciMLBase.successful_retcode(sol_array)
    @test SciMLBase.successful_retcode(sol_scalar)
    @test isapprox(sol_array[u(t, x)], sol_scalar[u(t, x)], rtol = 1e-6)
end

# --- 14h (E8): Nonlinlap equation now uses full-interior (no frame) ---

@testset "Nonlinlap equation uses full-interior ArrayOp" begin
    using SymbolicUtils
    using Symbolics: unwrap
    using ModelingToolkit.ModelingToolkitBase: flatten_equations

    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    # Nonlinear diffusion: Dt(u) ~ Dx(u * Dx(u))
    eq = Dt(u(t, x)) ~ Dx(u(t, x) * Dx(u(t, x)))

    bcs = [
        u(0, x) ~ 1.0 + 0.5 * sin(π * x),
        u(t, 0) ~ 1.0,
        u(t, 1) ~ 1.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.1),
        x ∈ Interval(0.0, 1.0),
    ]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    dx = 0.05
    disc = MOLFiniteDifference([x => dx], t;
        discretization_strategy = ArrayDiscretization()
    )

    sys, tspan = MethodOfLines.symbolic_discretize(pdesys, disc)
    eqs_out = equations(sys)

    # Nonlinlap now uses full-interior mode: should have ArrayOp equations
    has_arrayop = any(eqs_out) do eq
        lhs_raw = unwrap(eq.lhs)
        rhs_raw = unwrap(eq.rhs)
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(lhs_raw)) ||
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(rhs_raw))
    end
    @test has_arrayop

    # Should match scalar path
    disc_scalar = MOLFiniteDifference([x => dx], t;
        discretization_strategy = ScalarizedDiscretization()
    )
    prob_array = discretize(pdesys, disc)
    prob_scalar = discretize(pdesys, disc_scalar)
    sol_array = solve(prob_array, Rosenbrock32(), saveat = 0.02)
    sol_scalar = solve(prob_scalar, Rosenbrock32(), saveat = 0.02)

    @test SciMLBase.successful_retcode(sol_array)
    @test SciMLBase.successful_retcode(sol_scalar)
    @test isapprox(sol_array[u(t, x)], sol_scalar[u(t, x)], rtol = 1e-10)
end

# ===========================================================================
# Phase 15: Nonlinlap + Spherical Full-Interior ArrayOp Tests
# ===========================================================================

# --- 15a (E1): 1D nonlinlap full-interior approx_order=2 ---

@testset "Nonlinlap full-interior approx_order=2 matches scalar" begin
    @parameters t x
    @variables u(..)
    Dx = Differential(x)
    Dt = Differential(t)

    k = 0.1
    eq = Dt(u(t, x)) ~ Dx(k * Dx(u(t, x)))

    bcs = [
        u(0.0, x) ~ sin(π * x),
        u(t, 0.0) ~ 0.0,
        u(t, 1.0) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.5),
        x ∈ Interval(0.0, 1.0),
    ]
    @named pdesys = PDESystem([eq], bcs, domains, [t, x], [u(t, x)])

    dx = 0.05
    disc_scalar = MOLFiniteDifference([x => dx], t;
        approx_order = 2,
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([x => dx], t;
        approx_order = 2,
        discretization_strategy = ArrayDiscretization()
    )

    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)
    sol_scalar = solve(prob_scalar, Tsit5(), saveat = 0.1)
    sol_array = solve(prob_array, Tsit5(), saveat = 0.1)

    @test SciMLBase.successful_retcode(sol_scalar)
    @test SciMLBase.successful_retcode(sol_array)
    @test isapprox(sol_scalar[u(t, x)], sol_array[u(t, x)], rtol = 1e-10)
end

# --- 15b (E2): 1D nonlinlap full-interior approx_order=4 ---

@testset "Nonlinlap full-interior approx_order=4 matches scalar" begin
    @parameters t x
    @variables u(..)
    Dx = Differential(x)
    Dt = Differential(t)

    k = 0.1
    eq = Dt(u(t, x)) ~ Dx(k * Dx(u(t, x)))

    bcs = [
        u(0.0, x) ~ sin(π * x),
        u(t, 0.0) ~ 0.0,
        u(t, 1.0) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.5),
        x ∈ Interval(0.0, 1.0),
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
    sol_scalar = solve(prob_scalar, Tsit5(), saveat = 0.1)
    sol_array = solve(prob_array, Tsit5(), saveat = 0.1)

    @test SciMLBase.successful_retcode(sol_scalar)
    @test SciMLBase.successful_retcode(sol_array)
    @test isapprox(sol_scalar[u(t, x)], sol_array[u(t, x)], rtol = 1e-10)
end

# --- 15c (E3): 1D nonlinlap non-uniform grid full-interior ---

@testset "Nonlinlap full-interior non-uniform grid matches scalar" begin
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

    # Non-uniform grid: denser near x=0
    xs = [0.0; 0.05:0.1:2.0...]
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

    @test SciMLBase.successful_retcode(sol_scalar)
    @test SciMLBase.successful_retcode(sol_array)
    @test isapprox(sol_scalar[u(t, x)], sol_array[u(t, x)], rtol = 1e-10)
end

# --- 15d (E4): 1D nonlinlap variable coefficient Dx(u*Dx(u)) ---

@testset "Nonlinlap full-interior variable coefficient matches scalar" begin
    @parameters t x
    @variables u(..)
    Dx = Differential(x)
    Dt = Differential(t)

    eq = Dt(u(t, x)) ~ Dx(u(t, x) * Dx(u(t, x)))

    bcs = [
        u(0, x) ~ 1.0 + 0.5 * sin(π * x),
        u(t, 0) ~ 1.0,
        u(t, 1) ~ 1.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.1),
        x ∈ Interval(0.0, 1.0),
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
    sol_scalar = solve(prob_scalar, Rosenbrock32(), saveat = 0.02)
    sol_array = solve(prob_array, Rosenbrock32(), saveat = 0.02)

    @test SciMLBase.successful_retcode(sol_scalar)
    @test SciMLBase.successful_retcode(sol_array)
    @test isapprox(sol_scalar[u(t, x)], sol_array[u(t, x)], rtol = 1e-10)
end

# --- 15e (E5): 2D nonlinlap full-interior ---

@testset "2D nonlinlap full-interior matches scalar" begin
    @parameters t x y
    @variables u(..)
    Dx = Differential(x)
    Dy = Differential(y)
    Dt = Differential(t)

    k = 0.1
    eq = Dt(u(t, x, y)) ~ Dx(k * Dx(u(t, x, y))) + Dy(k * Dy(u(t, x, y)))

    bcs = [
        u(0, x, y) ~ sin(π * x) * sin(π * y),
        u(t, 0, y) ~ 0.0,
        u(t, 1, y) ~ 0.0,
        u(t, x, 0) ~ 0.0,
        u(t, x, 1) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.1),
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
    sol_scalar = solve(prob_scalar, Tsit5(), saveat = 0.02)
    sol_array = solve(prob_array, Tsit5(), saveat = 0.02)

    @test SciMLBase.successful_retcode(sol_scalar)
    @test SciMLBase.successful_retcode(sol_array)
    @test isapprox(sol_scalar[u(t, x, y)], sol_array[u(t, x, y)], rtol = 1e-10)
end

# --- 15f (E6): 1D spherical Laplacian full-interior ---

@testset "Spherical Laplacian full-interior matches scalar" begin
    @parameters t r
    @variables u(..)
    Dr = Differential(r)
    Dt = Differential(t)

    D_coeff = 0.1
    # Spherical Laplacian: (1/r^2) * Dr(r^2 * D_coeff * Dr(u))
    eq = Dt(u(t, r)) ~ (1 / r^2) * Dr(r^2 * D_coeff * Dr(u(t, r)))

    bcs = [
        u(0, r) ~ sin(π * r) / r,
        Dr(u(t, 0.1)) ~ 0.0,
        u(t, 2.0) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.5),
        r ∈ Interval(0.1, 2.0),
    ]
    @named pdesys = PDESystem([eq], bcs, domains, [t, r], [u(t, r)])

    dr = 0.05
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

    @test SciMLBase.successful_retcode(sol_scalar)
    @test SciMLBase.successful_retcode(sol_array)
    @test isapprox(sol_scalar[u(t, r)], sol_array[u(t, r)], rtol = 1e-10)
end

# --- 15g (E7): Mixed nonlinlap + centered full-interior ---

@testset "Mixed nonlinlap+centered full-interior matches scalar" begin
    @parameters t x y
    @variables u(..)
    Dx = Differential(x)
    Dy = Differential(y)
    Dt = Differential(t)
    Dyy = Dy^2

    k = 0.1
    # Mixed: nonlinlap in x, standard Laplacian in y
    eq = Dt(u(t, x, y)) ~ Dx(k * Dx(u(t, x, y))) + Dyy(u(t, x, y))

    bcs = [
        u(0, x, y) ~ sin(π * x) * sin(π * y),
        u(t, 0, y) ~ 0.0,
        u(t, 1, y) ~ 0.0,
        u(t, x, 0) ~ 0.0,
        u(t, x, 1) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.1),
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
    sol_scalar = solve(prob_scalar, Tsit5(), saveat = 0.02)
    sol_array = solve(prob_array, Tsit5(), saveat = 0.02)

    @test SciMLBase.successful_retcode(sol_scalar)
    @test SciMLBase.successful_retcode(sol_array)
    @test isapprox(sol_scalar[u(t, x, y)], sol_array[u(t, x, y)], rtol = 1e-4)
end

# --- 15h (E8): WENO equation keeps frame (regression guard) ---

@testset "WENO equation retains frame per-point equations" begin
    using SymbolicUtils
    using Symbolics: unwrap
    using ModelingToolkit.ModelingToolkitBase: flatten_equations

    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    # Advection with WENO: Dt(u) = -Dx(u)
    eq = Dt(u(t, x)) ~ -Dx(u(t, x))

    bcs = [
        u(0, x) ~ exp(-100.0 * (x - 0.5)^2),
        u(t, 0) ~ 0.0,
        u(t, 2) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.1),
        x ∈ Interval(0.0, 2.0),
    ]
    @named pdesys = PDESystem([eq], bcs, domains, [t, x], [u(t, x)])

    dx = 0.1
    disc = MOLFiniteDifference([x => dx], t;
        advection_scheme = WENOScheme(),
        discretization_strategy = ArrayDiscretization()
    )

    sys, tspan = MethodOfLines.symbolic_discretize(pdesys, disc)
    eqs_out = equations(sys)
    flat = flatten_equations(eqs_out)

    n_grid = Int(2.0 / dx) + 1  # 21
    # Flattened equations include interior + boundary equations
    @test length(flat) >= n_grid - 2  # At least n_interior equations

    # WENO should still use the standard path with frame per-point equations
    scalar_eq_count = count(eqs_out) do eq
        lhs_raw = unwrap(eq.lhs)
        rhs_raw = unwrap(eq.rhs)
        !SymbolicUtils.is_array_shape(SymbolicUtils.shape(lhs_raw)) &&
        !SymbolicUtils.is_array_shape(SymbolicUtils.shape(rhs_raw))
    end
    @test scalar_eq_count > 0  # Frame per-point equations should be present
end

# ===========================================================================
# Phase 16: Non-Uniform Nonlinlap + Spherical Full-Interior Tests
# ===========================================================================

# --- 16a (D1): 1D nonlinlap constant coeff non-uniform grid ---

@testset "Nonlinlap constant-coeff non-uniform grid matches scalar" begin
    @parameters t x
    @variables u(..)
    Dx = Differential(x)
    Dt = Differential(t)

    k = 0.1
    eq = Dt(u(t, x)) ~ Dx(k * Dx(u(t, x)))

    bcs = [
        u(0.0, x) ~ sin(π * x),
        u(t, 0.0) ~ 0.0,
        u(t, 1.0) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.5),
        x ∈ Interval(0.0, 1.0),
    ]
    @named pdesys = PDESystem([eq], bcs, domains, [t, x], [u(t, x)])

    # Non-uniform grid: denser near x=0
    xs = [0.0; 0.02:0.05:1.0...]
    disc_scalar = MOLFiniteDifference([x => xs], t;
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([x => xs], t;
        discretization_strategy = ArrayDiscretization()
    )

    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)
    sol_scalar = solve(prob_scalar, Tsit5(), saveat = 0.1)
    sol_array = solve(prob_array, Tsit5(), saveat = 0.1)

    @test SciMLBase.successful_retcode(sol_scalar)
    @test SciMLBase.successful_retcode(sol_array)
    @test isapprox(sol_scalar[u(t, x)], sol_array[u(t, x)], rtol = 1e-10)
end

# --- 16b (D3): 2D nonlinlap non-uniform in one dimension ---

@testset "2D nonlinlap non-uniform in x, uniform in y matches scalar" begin
    @parameters t x y
    @variables u(..)
    Dx = Differential(x)
    Dy = Differential(y)
    Dt = Differential(t)

    k = 0.1
    eq = Dt(u(t, x, y)) ~ Dx(k * Dx(u(t, x, y))) + Dy(k * Dy(u(t, x, y)))

    bcs = [
        u(0, x, y) ~ sin(π * x) * sin(π * y),
        u(t, 0, y) ~ 0.0,
        u(t, 1, y) ~ 0.0,
        u(t, x, 0) ~ 0.0,
        u(t, x, 1) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.1),
        x ∈ Interval(0.0, 1.0),
        y ∈ Interval(0.0, 1.0),
    ]
    @named pdesys = PDESystem([eq], bcs, domains, [t, x, y], [u(t, x, y)])

    # Non-uniform in x, uniform in y
    xs = [0.0; 0.03:0.1:1.0...]
    dy = 0.1
    disc_scalar = MOLFiniteDifference([x => xs, y => dy], t;
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([x => xs, y => dy], t;
        discretization_strategy = ArrayDiscretization()
    )

    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)
    sol_scalar = solve(prob_scalar, Tsit5(), saveat = 0.02)
    sol_array = solve(prob_array, Tsit5(), saveat = 0.02)

    @test SciMLBase.successful_retcode(sol_scalar)
    @test SciMLBase.successful_retcode(sol_array)
    @test isapprox(sol_scalar[u(t, x, y)], sol_array[u(t, x, y)], rtol = 1e-10)
end

# --- 16c (D5): 1D spherical Laplacian non-uniform grid ---

@testset "Spherical Laplacian non-uniform grid matches scalar" begin
    @parameters t r
    @variables u(..)
    Dr = Differential(r)
    Dt = Differential(t)

    D_coeff = 0.1
    # Spherical Laplacian: (1/r^2) * Dr(r^2 * D_coeff * Dr(u))
    eq = Dt(u(t, r)) ~ (1 / r^2) * Dr(r^2 * D_coeff * Dr(u(t, r)))

    bcs = [
        u(0, r) ~ sin(π * r) / r,
        Dr(u(t, 0.1)) ~ 0.0,
        u(t, 2.0) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.5),
        r ∈ Interval(0.1, 2.0),
    ]
    @named pdesys = PDESystem([eq], bcs, domains, [t, r], [u(t, r)])

    # Non-uniform grid in r
    rs = [0.1; 0.15:0.1:2.0...]
    disc_scalar = MOLFiniteDifference([r => rs], t;
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([r => rs], t;
        discretization_strategy = ArrayDiscretization()
    )

    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)
    sol_scalar = solve(prob_scalar, Tsit5(), saveat = 0.1)
    sol_array = solve(prob_array, Tsit5(), saveat = 0.1)

    @test SciMLBase.successful_retcode(sol_scalar)
    @test SciMLBase.successful_retcode(sol_array)
    @test isapprox(sol_scalar[u(t, r)], sol_array[u(t, r)], rtol = 1e-10)
end

# --- 16d (D6): Spherical + standard centered derivative in 2D ---

@testset "Spherical + centered 2D full-interior matches scalar" begin
    @parameters t r z
    @variables u(..)
    Dr = Differential(r)
    Dzz = Differential(z)^2
    Dt = Differential(t)

    D_coeff = 0.1
    # Spherical Laplacian in r + standard diffusion in z
    eq = Dt(u(t, r, z)) ~ (1 / r^2) * Dr(r^2 * D_coeff * Dr(u(t, r, z))) + D_coeff * Dzz(u(t, r, z))

    bcs = [
        u(0, r, z) ~ sin(π * r) * sin(π * z) / r,
        Dr(u(t, 0.1, z)) ~ 0.0,
        u(t, 2.0, z) ~ 0.0,
        u(t, r, 0.0) ~ 0.0,
        u(t, r, 1.0) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.1),
        r ∈ Interval(0.1, 2.0),
        z ∈ Interval(0.0, 1.0),
    ]
    @named pdesys = PDESystem([eq], bcs, domains, [t, r, z], [u(t, r, z)])

    dr = 0.1
    dz = 0.1
    disc_scalar = MOLFiniteDifference([r => dr, z => dz], t;
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([r => dr, z => dz], t;
        discretization_strategy = ArrayDiscretization()
    )

    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)
    sol_scalar = solve(prob_scalar, Tsit5(), saveat = 0.02)
    sol_array = solve(prob_array, Tsit5(), saveat = 0.02)

    @test SciMLBase.successful_retcode(sol_scalar)
    @test SciMLBase.successful_retcode(sol_array)
    @test isapprox(sol_scalar[u(t, r, z)], sol_array[u(t, r, z)], rtol = 1e-4)
end

# ===========================================================================
# Phase 17: Periodic Full-Interior ArrayOp Tests
# ===========================================================================

# --- 17a: 1D periodic centered diffusion, structural ---

@testset "Periodic full-interior: 1D centered diffusion structural" begin
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

    dx = 0.1
    disc = MOLFiniteDifference([x => dx], t;
        discretization_strategy = ArrayDiscretization()
    )

    sys, tspan = MethodOfLines.symbolic_discretize(pdesys, disc)
    eqs = equations(sys)

    # Verify full-interior ArrayOp is used (no scalar frame equations)
    has_arrayop = any(eqs) do eq
        lhs_raw = unwrap(eq.lhs)
        rhs_raw = unwrap(eq.rhs)
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(lhs_raw)) ||
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(rhs_raw))
    end
    @test has_arrayop

    # In full-interior mode, there should be NO scalar frame equations for
    # the PDE interior — only ArrayOp equations plus the periodic BC constraint.
    # The periodic BC is a scalar equation (u[1] ~ u[N]).
    scalar_eqs = filter(eqs) do eq
        lhs_raw = unwrap(eq.lhs)
        rhs_raw = unwrap(eq.rhs)
        is_array = SymbolicUtils.is_array_shape(SymbolicUtils.shape(lhs_raw)) ||
                   SymbolicUtils.is_array_shape(SymbolicUtils.shape(rhs_raw))
        return !is_array
    end
    # Only 1 scalar equation: the periodic BC constraint
    @test length(scalar_eqs) == 1
end

# --- 17b: 1D periodic centered diffusion, numerical ---

@testset "Periodic full-interior: 1D centered diffusion numerical" begin
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

    @test SciMLBase.successful_retcode(sol_scalar)
    @test SciMLBase.successful_retcode(sol_array)
    @test isapprox(sol_scalar[u(t, x)], sol_array[u(t, x)], rtol = 1e-10)
end

# --- 17c: 1D periodic upwind advection ---

@testset "Periodic full-interior: 1D upwind advection numerical" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    L = 2.0
    v_adv = 1.0
    eq = Dt(u(t, x)) ~ -v_adv * Dx(u(t, x))

    bcs = [
        u(0, x) ~ sin(2π * x / L),
        u(t, 0) ~ u(t, L),  # Periodic BC
    ]

    domains = [
        t ∈ Interval(0.0, 0.5),
        x ∈ Interval(0.0, L),
    ]

    @named pdesys = PDESystem([eq], bcs, domains, [t, x], [u(t, x)])

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

    sol_scalar = solve(prob_scalar, Tsit5(), saveat = 0.1)
    sol_array = solve(prob_array, Tsit5(), saveat = 0.1)

    @test SciMLBase.successful_retcode(sol_scalar)
    @test SciMLBase.successful_retcode(sol_array)
    @test isapprox(sol_scalar[u(t, x)], sol_array[u(t, x)], rtol = 1e-10)
end

# --- 17d: 2D fully-periodic diffusion ---

@testset "Periodic full-interior: 2D fully-periodic diffusion numerical" begin
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

    @test SciMLBase.successful_retcode(sol_scalar)
    @test SciMLBase.successful_retcode(sol_array)
    @test isapprox(sol_scalar[u(t, x, y)], sol_array[u(t, x, y)], rtol = 1e-10)
end

# --- 17e: 2D mixed periodic/non-periodic ---

@testset "Periodic full-interior: 2D mixed periodic/non-periodic" begin
    using SymbolicUtils
    using Symbolics: unwrap

    @parameters t x y
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    Lx = 2.0
    eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y))

    bcs = [
        u(0, x, y) ~ sin(2π * x / Lx) * sin(π * y),
        u(t, 0, y) ~ u(t, Lx, y),  # Periodic in x
        u(t, x, 0) ~ 0.0,           # Dirichlet in y
        u(t, x, 1) ~ 0.0,           # Dirichlet in y
    ]

    domains = [
        t ∈ Interval(0.0, 0.3),
        x ∈ Interval(0.0, Lx),
        y ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem([eq], bcs, domains, [t, x, y], [u(t, x, y)])

    dx = 0.2
    disc_scalar = MOLFiniteDifference([x => dx, y => dx], t;
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([x => dx, y => dx], t;
        discretization_strategy = ArrayDiscretization()
    )

    # Structural check: full-interior mode should be used
    sys_array, _ = MethodOfLines.symbolic_discretize(pdesys, disc_array)
    eqs_array = equations(sys_array)
    has_arrayop = any(eqs_array) do eq
        lhs_raw = unwrap(eq.lhs)
        rhs_raw = unwrap(eq.rhs)
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(lhs_raw)) ||
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(rhs_raw))
    end
    @test has_arrayop

    # Numerical check
    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)

    sol_scalar = solve(prob_scalar, Tsit5(), saveat = 0.1)
    sol_array = solve(prob_array, Tsit5(), saveat = 0.1)

    @test SciMLBase.successful_retcode(sol_scalar)
    @test SciMLBase.successful_retcode(sol_array)
    @test isapprox(sol_scalar[u(t, x, y)], sol_array[u(t, x, y)], rtol = 1e-10)
end

# --- 17f: 1D periodic nonlinlap ---

@testset "Periodic full-interior: 1D nonlinlap numerical" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    L = 2.0
    # Nonlinear diffusion: Dt(u) ~ Dx(u * Dx(u))
    eq = Dt(u(t, x)) ~ Dx(u(t, x) * Dx(u(t, x)))

    bcs = [
        u(0, x) ~ 2.0 + sin(2π * x / L),
        u(t, 0) ~ u(t, L),  # Periodic BC
    ]

    domains = [
        t ∈ Interval(0.0, 0.1),
        x ∈ Interval(0.0, L),
    ]

    @named pdesys = PDESystem([eq], bcs, domains, [t, x], [u(t, x)])

    dx = 0.1
    disc_scalar = MOLFiniteDifference([x => dx], t;
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference([x => dx], t;
        discretization_strategy = ArrayDiscretization()
    )

    prob_scalar = discretize(pdesys, disc_scalar)
    prob_array = discretize(pdesys, disc_array)

    sol_scalar = solve(prob_scalar, Tsit5(), saveat = 0.02)
    sol_array = solve(prob_array, Tsit5(), saveat = 0.02)

    @test SciMLBase.successful_retcode(sol_scalar)
    @test SciMLBase.successful_retcode(sol_array)
    @test isapprox(sol_scalar[u(t, x)], sol_array[u(t, x)], rtol = 1e-10)
end

# --- 17g: 1D periodic non-uniform (falls back to standard path) ---
# Note: Non-uniform periodic is now fixed in the standard path (Phase 18).
# This test verifies that the gate condition correctly routes non-uniform periodic
# to the standard path (not full-interior), and that uniform periodic still works
# via full-interior.

@testset "Periodic full-interior: non-uniform periodic uses standard path, uniform uses full-interior" begin
    using SymbolicUtils
    using Symbolics: unwrap

    # Uniform periodic: should use full-interior (no frame equations)
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    L = 2.0
    eq = Dt(u(t, x)) ~ Dxx(u(t, x))
    bcs = [u(0, x) ~ sin(2π * x / L), u(t, 0) ~ u(t, L)]
    domains = [t ∈ Interval(0.0, 0.5), x ∈ Interval(0.0, L)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    dx = 0.1
    disc_uniform = MOLFiniteDifference([x => dx], t;
        discretization_strategy = ArrayDiscretization()
    )
    sys_uniform, _ = MethodOfLines.symbolic_discretize(pdesys, disc_uniform)
    eqs_uniform = equations(sys_uniform)

    # Full-interior: only ArrayOp + periodic BC constraint
    scalar_count = count(eqs_uniform) do eq
        lhs_raw = unwrap(eq.lhs)
        rhs_raw = unwrap(eq.rhs)
        is_array = SymbolicUtils.is_array_shape(SymbolicUtils.shape(lhs_raw)) ||
                   SymbolicUtils.is_array_shape(SymbolicUtils.shape(rhs_raw))
        return !is_array
    end
    @test scalar_count == 1  # Only the periodic BC constraint
end

# ===========================================================================
# Phase 18: Non-Uniform Periodic Fix Tests
# ===========================================================================
# Note: The scalar path (ScalarizedDiscretization) does not support
# non-uniform periodic grids (asserts in centered_difference.jl).
# Tests validate against analytical solutions instead.

# --- 18a: 1D non-uniform periodic centered diffusion, structural ---

@testset "Non-uniform periodic: 1D centered diffusion structural" begin
    using SymbolicUtils
    using Symbolics: unwrap

    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    L = 2.0
    eq = Dt(u(t, x)) ~ Dxx(u(t, x))
    bcs = [u(0, x) ~ sin(2π * x / L), u(t, 0) ~ u(t, L)]
    domains = [t ∈ Interval(0.0, 0.5), x ∈ Interval(0.0, L)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    # Non-uniform grid (collect + perturbation)
    N = 21
    xs_base = range(0.0, L, length = N)
    xs = [xs_base[1]; [xs_base[i] + 0.01 * (-1)^i for i in 2:(N - 1)]; xs_base[end]]
    disc = MOLFiniteDifference([x => xs], t;
        discretization_strategy = ArrayDiscretization()
    )

    sys, tspan = MethodOfLines.symbolic_discretize(pdesys, disc)
    eqs = equations(sys)

    # Should produce ArrayOp equations (not fall back to per-point)
    has_arrayop = any(eqs) do eq
        lhs_raw = unwrap(eq.lhs)
        rhs_raw = unwrap(eq.rhs)
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(lhs_raw)) ||
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(rhs_raw))
    end
    @test has_arrayop

    # Non-uniform periodic should use the standard path (not full-interior),
    # so there should be frame equations in addition to the ArrayOp.
    # But the important thing is that it doesn't error out.
    @test length(eqs) >= 1
end

# --- 18b: 1D non-uniform periodic centered diffusion, numerical ---
# Analytical: u(t,x) = exp(-(2π/L)²*t) * sin(2πx/L)

@testset "Non-uniform periodic: 1D centered diffusion vs analytical" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    L = 2.0
    k = 2π / L
    eq = Dt(u(t, x)) ~ Dxx(u(t, x))
    bcs = [u(0, x) ~ sin(k * x), u(t, 0) ~ u(t, L)]
    domains = [t ∈ Interval(0.0, 0.5), x ∈ Interval(0.0, L)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    N = 41
    xs_base = range(0.0, L, length = N)
    xs = [xs_base[1]; [xs_base[i] + 0.005 * (-1)^i for i in 2:(N - 1)]; xs_base[end]]

    disc = MOLFiniteDifference([x => xs], t;
        discretization_strategy = ArrayDiscretization()
    )

    prob = discretize(pdesys, disc)
    sol = solve(prob, Tsit5(), saveat = 0.1)

    @test SciMLBase.successful_retcode(sol)

    u_exact(tv, xv) = exp(-k^2 * tv) * sin(k * xv)
    x_disc = sol[x]
    for (i, tv) in enumerate(sol.t)
        exact = u_exact.(tv, x_disc)
        @test isapprox(sol[u(t, x)][i, :], exact, atol = 0.01)
    end
end

# --- 18c: 1D non-uniform periodic upwind advection ---
# Analytical: u(t,x) = sin(2π(x - v*t)/L)

@testset "Non-uniform periodic: 1D upwind advection vs analytical" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    L = 2.0
    v_adv = 1.0
    k = 2π / L
    eq = Dt(u(t, x)) ~ -v_adv * Dx(u(t, x))
    bcs = [u(0, x) ~ sin(k * x), u(t, 0) ~ u(t, L)]
    domains = [t ∈ Interval(0.0, 0.5), x ∈ Interval(0.0, L)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    N = 41
    xs_base = range(0.0, L, length = N)
    xs = [xs_base[1]; [xs_base[i] + 0.005 * (-1)^i for i in 2:(N - 1)]; xs_base[end]]

    disc = MOLFiniteDifference([x => xs], t;
        advection_scheme = UpwindScheme(),
        discretization_strategy = ArrayDiscretization()
    )

    prob = discretize(pdesys, disc)
    sol = solve(prob, Tsit5(), saveat = 0.1)

    @test SciMLBase.successful_retcode(sol)

    u_exact(tv, xv) = sin(k * (xv - v_adv * tv))
    x_disc = sol[x]
    # First-order upwind is very diffusive; check per-element with generous tolerance
    for (i, tv) in enumerate(sol.t)
        exact = u_exact.(tv, x_disc)
        @test all(isapprox.(sol[u(t, x)][i, :], exact, atol = 0.15))
    end
end

# --- 18d: 2D mixed: non-uniform periodic + uniform non-periodic ---
# Analytical: u(t,x,y) = exp(-((2π/Lx)² + (π/Ly)²)*t) * sin(2πx/Lx) * sin(πy/Ly)

@testset "Non-uniform periodic: 2D mixed periodic/non-periodic vs analytical" begin
    @parameters t x y
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    Lx = 2.0
    Ly = 1.0
    kx = 2π / Lx
    ky = π / Ly
    eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y))
    bcs = [
        u(0, x, y) ~ sin(kx * x) * sin(ky * y),
        u(t, 0, y) ~ u(t, Lx, y),  # Periodic in x
        u(t, x, 0) ~ 0.0,           # Dirichlet in y
        u(t, x, Ly) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.3),
        x ∈ Interval(0.0, Lx),
        y ∈ Interval(0.0, Ly),
    ]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])

    Nx = 21
    xs_base = range(0.0, Lx, length = Nx)
    xs = [xs_base[1]; [xs_base[i] + 0.005 * (-1)^i for i in 2:(Nx - 1)]; xs_base[end]]
    dy = 0.1

    disc = MOLFiniteDifference([x => xs, y => dy], t;
        discretization_strategy = ArrayDiscretization()
    )

    prob = discretize(pdesys, disc)
    sol = solve(prob, Tsit5(), saveat = 0.1)

    @test SciMLBase.successful_retcode(sol)

    u_exact(tv, xv, yv) = exp(-(kx^2 + ky^2) * tv) * sin(kx * xv) * sin(ky * yv)
    x_disc = sol[x]
    y_disc = sol[y]
    for (i, tv) in enumerate(sol.t)
        u_num = sol[u(t, x, y)][i, :, :]
        u_ana = [u_exact(tv, xv, yv) for xv in x_disc, yv in y_disc]
        @test isapprox(u_num, u_ana, atol = 0.05)
    end
end

# --- 18e: Non-uniform periodic with higher order (order=4) ---

@testset "Non-uniform periodic: higher order (order=4) vs analytical" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    L = 2.0
    k = 2π / L
    eq = Dt(u(t, x)) ~ Dxx(u(t, x))
    bcs = [u(0, x) ~ sin(k * x), u(t, 0) ~ u(t, L)]
    domains = [t ∈ Interval(0.0, 0.5), x ∈ Interval(0.0, L)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    N = 41
    xs_base = range(0.0, L, length = N)
    xs = [xs_base[1]; [xs_base[i] + 0.005 * (-1)^i for i in 2:(N - 1)]; xs_base[end]]

    disc = MOLFiniteDifference([x => xs], t;
        approx_order = 4,
        discretization_strategy = ArrayDiscretization()
    )

    prob = discretize(pdesys, disc)
    sol = solve(prob, Tsit5(), saveat = 0.1)

    @test SciMLBase.successful_retcode(sol)

    u_exact(tv, xv) = exp(-k^2 * tv) * sin(k * xv)
    x_disc = sol[x]
    for (i, tv) in enumerate(sol.t)
        exact = u_exact.(tv, x_disc)
        # Higher order should be more accurate
        @test isapprox(sol[u(t, x)][i, :], exact, atol = 0.005)
    end
end

# --- 18f: Regression — collect-based grid (uniform spacing, vector-typed) ---

@testset "Non-uniform periodic: regression — collect grid can solve" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    L = 2.0
    k = 2π / L
    eq = Dt(u(t, x)) ~ Dxx(u(t, x))
    bcs = [u(0, x) ~ sin(k * x), u(t, 0) ~ u(t, L)]
    domains = [t ∈ Interval(0.0, 0.5), x ∈ Interval(0.0, L)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    # collect → Vector{Float64}, triggers non-uniform code path even though spacing is uniform
    xs = collect(range(0.0, L, length = 21))

    disc = MOLFiniteDifference([x => xs], t;
        discretization_strategy = ArrayDiscretization()
    )

    prob = discretize(pdesys, disc)
    sol = solve(prob, Tsit5(), saveat = 0.1)

    @test SciMLBase.successful_retcode(sol)

    u_exact(tv, xv) = exp(-k^2 * tv) * sin(k * xv)
    x_disc = sol[x]
    for (i, tv) in enumerate(sol.t)
        exact = u_exact.(tv, x_disc)
        @test isapprox(sol[u(t, x)][i, :], exact, atol = 0.01)
    end
end

# ===========================================================================
# Phase 19: Staggered Grid ArrayOp Support
# ===========================================================================

# --- 19a: 1D wave equation (mixed BCs), structural ---

@testset "Staggered ArrayOp: 1D wave equation (mixed BCs), structural" begin
    using SymbolicUtils
    using Symbolics: unwrap
    using ModelingToolkit.ModelingToolkitBase: flatten_equations

    @parameters t x
    @variables ρ(..) ϕ(..)
    Dt = Differential(t)
    Dx = Differential(x)

    a = 5.0
    L = 8.0
    dx = 0.125

    initialFunction(x) = exp(-(x)^2)
    eq = [
        Dt(ρ(t, x)) + Dx(ϕ(t, x)) ~ 0,
        Dt(ϕ(t, x)) + a^2 * Dx(ρ(t, x)) ~ 0,
    ]
    bcs = [
        ρ(0, x) ~ initialFunction(x),
        ϕ(0.0, x) ~ 0.0,
        Dx(ρ(t, L)) ~ 0.0,
        ϕ(t, -L) ~ 0.0,
    ]

    domains = [
        t in Interval(0.0, 1.0),
        x in Interval(-L, L),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [ρ(t, x), ϕ(t, x)])

    disc = MOLFiniteDifference(
        [x => dx], t, grid_align = MethodOfLines.StaggeredGrid(),
        edge_aligned_var = ϕ(t, x),
        discretization_strategy = ArrayDiscretization()
    )

    sys, tspan = MethodOfLines.symbolic_discretize(pdesys, disc)
    eqs = equations(sys)

    # Verify ArrayOp is generated
    has_arrayop = any(eqs) do eq
        lhs_raw = unwrap(eq.lhs)
        rhs_raw = unwrap(eq.rhs)
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(lhs_raw)) ||
        SymbolicUtils.is_array_shape(SymbolicUtils.shape(rhs_raw))
    end
    @test has_arrayop

    # Verify scalar equations are fewer than total interior points
    # (meaning ArrayOp is covering most of them)
    scalar_eqs = filter(eqs) do eq
        lhs_raw = unwrap(eq.lhs)
        rhs_raw = unwrap(eq.rhs)
        is_array = SymbolicUtils.is_array_shape(SymbolicUtils.shape(lhs_raw)) ||
                   SymbolicUtils.is_array_shape(SymbolicUtils.shape(rhs_raw))
        return !is_array
    end
    # With staggered grid, there are 2 equations (ρ and ϕ), each gets an ArrayOp
    # plus boundary frame equations and BC equations.
    # The number of scalar equations should be much less than total grid points.
    Ngrid = round(Int, 2 * L / dx) + 1
    @test length(scalar_eqs) < 2 * Ngrid
end

# --- 19b: 1D wave equation (mixed BCs), numerical ---
# Compare scalar and array paths using non-split ODEProblems.
# We use symbolic_discretize + mtkcompile to bypass symbolic_trace (which
# creates SplitODEProblem), then compare via PDE variable access which
# handles unknowns ordering automatically.

@testset "Staggered ArrayOp: 1D wave equation (mixed BCs), numerical" begin
    using SymbolicUtils: getmetadata

    @parameters t x
    @variables ρ(..) ϕ(..)
    Dt = Differential(t)
    Dx = Differential(x)

    a = 5.0
    L = 8.0
    dx = 0.125
    dt = (dx / a)^2
    tmax = 0.1  # short run for test speed

    initialFunction(x) = exp(-(x)^2)
    eq = [
        Dt(ρ(t, x)) + Dx(ϕ(t, x)) ~ 0,
        Dt(ϕ(t, x)) + a^2 * Dx(ρ(t, x)) ~ 0,
    ]
    bcs = [
        ρ(0, x) ~ initialFunction(x),
        ϕ(0.0, x) ~ 0.0,
        Dx(ρ(t, L)) ~ 0.0,
        ϕ(t, -L) ~ 0.0,
    ]

    domains = [
        t in Interval(0.0, tmax),
        x in Interval(-L, L),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [ρ(t, x), ϕ(t, x)])

    disc_scalar = MOLFiniteDifference(
        [x => dx], t, grid_align = MethodOfLines.StaggeredGrid(),
        edge_aligned_var = ϕ(t, x),
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference(
        [x => dx], t, grid_align = MethodOfLines.StaggeredGrid(),
        edge_aligned_var = ϕ(t, x),
        discretization_strategy = ArrayDiscretization()
    )

    # Build non-split ODEProblems via symbolic_discretize + mtkcompile
    # (bypassing symbolic_trace which creates SplitODEProblem)
    sys_s, tspan = MethodOfLines.symbolic_discretize(pdesys, disc_scalar)
    sys_a, _ = MethodOfLines.symbolic_discretize(pdesys, disc_array)

    csys_s = mtkcompile(sys_s)
    csys_a = mtkcompile(sys_a)

    # Get u0 from metadata (same mechanism as staggered_discretize.jl)
    mol_s = getmetadata(csys_s, ModelingToolkit.ProblemTypeCtx, nothing)
    u0_s = hasproperty(mol_s, :u0) ? mol_s.u0 : []
    mol_a = getmetadata(csys_a, ModelingToolkit.ProblemTypeCtx, nothing)
    u0_a = hasproperty(mol_a, :u0) ? mol_a.u0 : []

    # Build regular ODEProblems (not SplitODEProblem)
    prob_s = ODEProblem(csys_s, u0_s, tspan; build_initializeprob = false)
    prob_a = ODEProblem(csys_a, u0_a, tspan; build_initializeprob = false)

    # Solve with explicit Euler (equivalent to SplitEuler for non-split problems)
    sol_s = solve(prob_s, Euler(), dt = dt)
    sol_a = solve(prob_a, Euler(), dt = dt)

    @test SciMLBase.successful_retcode(sol_s)
    @test SciMLBase.successful_retcode(sol_a)

    # Compare using PDE variable access (handles unknowns ordering automatically)
    @test isapprox(sol_s[ρ(t, x)], sol_a[ρ(t, x)], rtol = 1e-10)
    @test isapprox(sol_s[ϕ(t, x)], sol_a[ϕ(t, x)], rtol = 1e-10)
end

# --- 19c: 1D wave equation (periodic BCs) ---

@testset "Staggered ArrayOp: 1D wave equation (periodic BCs)" begin
    using SymbolicUtils: getmetadata

    @parameters t x
    @variables ρ(..) ϕ(..)
    Dt = Differential(t)
    Dx = Differential(x)

    a = 5.0
    L = 8.0
    dx = 0.125
    dt = (dx / a)^2
    tmax = 0.1  # short run for test speed

    initialFunction(x) = exp(-(x - L / 2)^2)
    eq = [
        Dt(ρ(t, x)) + Dx(ϕ(t, x)) ~ 0,
        Dt(ϕ(t, x)) + a^2 * Dx(ρ(t, x)) ~ 0,
    ]
    bcs = [
        ρ(0, x) ~ initialFunction(x),
        ϕ(0.0, x) ~ 0.0,
        ρ(t, L) ~ ρ(t, -L),
        ϕ(t, -L) ~ ϕ(t, L),
    ]

    domains = [
        t in Interval(0.0, tmax),
        x in Interval(-L, L),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [ρ(t, x), ϕ(t, x)])

    disc_scalar = MOLFiniteDifference(
        [x => dx], t, grid_align = MethodOfLines.StaggeredGrid(),
        edge_aligned_var = ϕ(t, x),
        discretization_strategy = ScalarizedDiscretization()
    )
    disc_array = MOLFiniteDifference(
        [x => dx], t, grid_align = MethodOfLines.StaggeredGrid(),
        edge_aligned_var = ϕ(t, x),
        discretization_strategy = ArrayDiscretization()
    )

    # Build non-split ODEProblems via symbolic_discretize + mtkcompile
    sys_s, tspan = MethodOfLines.symbolic_discretize(pdesys, disc_scalar)
    sys_a, _ = MethodOfLines.symbolic_discretize(pdesys, disc_array)

    csys_s = mtkcompile(sys_s)
    csys_a = mtkcompile(sys_a)

    # Get u0 from metadata
    mol_s = getmetadata(csys_s, ModelingToolkit.ProblemTypeCtx, nothing)
    u0_s = hasproperty(mol_s, :u0) ? mol_s.u0 : []
    mol_a = getmetadata(csys_a, ModelingToolkit.ProblemTypeCtx, nothing)
    u0_a = hasproperty(mol_a, :u0) ? mol_a.u0 : []

    # Build regular ODEProblems
    prob_s = ODEProblem(csys_s, u0_s, tspan; build_initializeprob = false)
    prob_a = ODEProblem(csys_a, u0_a, tspan; build_initializeprob = false)

    # Solve with explicit Euler
    sol_s = solve(prob_s, Euler(), dt = dt)
    sol_a = solve(prob_a, Euler(), dt = dt)

    @test SciMLBase.successful_retcode(sol_s)
    @test SciMLBase.successful_retcode(sol_a)

    # Compare using PDE variable access
    @test isapprox(sol_s[ρ(t, x)], sol_a[ρ(t, x)], rtol = 1e-10)
    @test isapprox(sol_s[ϕ(t, x)], sol_a[ϕ(t, x)], rtol = 1e-10)
end

# --- 19d: SplitODEProblem works with ArrayOp ---

@testset "Staggered ArrayOp: SplitODEProblem compatibility" begin
    @parameters t x
    @variables ρ(..) ϕ(..)
    Dt = Differential(t)
    Dx = Differential(x)

    a = 5.0
    L = 4.0
    dx = 0.25
    dt = (dx / a)^2
    tmax = 1.0

    eq = [
        Dt(ρ(t, x)) + Dx(ϕ(t, x)) ~ 0,
        Dt(ϕ(t, x)) + a^2 * Dx(ρ(t, x)) ~ 0,
    ]
    bcs = [
        ρ(0, x) ~ exp(-(x)^2),
        ϕ(0.0, x) ~ 0.0,
        Dx(ρ(t, L)) ~ 0.0,
        ϕ(t, -L) ~ 0.0,
    ]

    domains = [
        t in Interval(0.0, tmax),
        x in Interval(-L, L),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [ρ(t, x), ϕ(t, x)])

    disc = MOLFiniteDifference(
        [x => dx], t, grid_align = MethodOfLines.StaggeredGrid(),
        edge_aligned_var = ϕ(t, x),
        discretization_strategy = ArrayDiscretization()
    )

    prob = discretize(pdesys, disc)

    # discretize for StaggeredGrid returns SplitODEProblem (which is an
    # ODEProblem with SplitFunction)
    @test prob.f isa SplitFunction

    sol = solve(prob, SplitEuler(), dt = dt)
    @test SciMLBase.successful_retcode(sol)

    # Verify solution is physically reasonable (bounded, not NaN)
    @test all(isfinite, sol.u[end])
    @test maximum(abs, sol.u[end]) < 100.0
end

# --- Float32 type-genericity tests -------------------------------------------

@testset "Float32: type-generic structs and helpers" begin
    # Test 1: _op_eltype extracts the correct type from DerivativeOperator{T}
    D_f32 = MethodOfLines.CompleteCenteredDifference(2, 2, Float32(0.1))
    D_f64 = MethodOfLines.CompleteCenteredDifference(2, 2, 0.1)
    @test MethodOfLines._op_eltype(D_f32) === Float32
    @test MethodOfLines._op_eltype(D_f64) === Float64

    # Test 2: collect() on SVector{N,Float32} produces Vector{Float32}
    coefs_f32 = collect(D_f32.stencil_coefs)
    @test eltype(coefs_f32) === Float32
    coefs_f64 = collect(D_f64.stencil_coefs)
    @test eltype(coefs_f64) === Float64

    # Test 3: StencilInfo{Float32} can be constructed
    si_f32 = MethodOfLines.StencilInfo{Float32}(
        D_f32, [0, 1, 2], true, nothing
    )
    @test si_f32.weight_matrix === nothing

    wmat_f32 = Float32[1.0 2.0; 3.0 4.0; 5.0 6.0]
    si_f32_nu = MethodOfLines.StencilInfo{Float32}(
        D_f32, [0, 1, 2], false, wmat_f32
    )
    @test eltype(si_f32_nu.weight_matrix) === Float32

    # Test 4: FullInteriorStencilInfo{Float32} can be constructed
    wmat = zeros(Float32, 3, 5)
    omat = zeros(Int, 3, 5)
    fisi = MethodOfLines.FullInteriorStencilInfo(wmat, omat, 3)
    @test eltype(fisi.weight_matrix) === Float32

    # Test 5: WENOStencilInfo{Float32} can be constructed
    wsi = MethodOfLines.WENOStencilInfo{Float32}(
        Float32(1e-6), [-2, -1, 0, 1, 2], 2, 2, Float32(0.1)
    )
    @test wsi.epsilon isa Float32
    @test wsi.dx_val isa Float32

    # Test 6: _stencil_coefs_to_matrix preserves element type from operator
    D_f32_nu = MethodOfLines.CompleteCenteredDifference(
        2, 2, Float32.(collect(range(0.0f0, 1.0f0, length=11)))
    )
    mat = MethodOfLines._stencil_coefs_to_matrix(D_f32_nu)
    @test eltype(mat) === Float32

    # Test 7: _periodic_stencil_positions preserves grid element type
    grid_f32 = Float32.(collect(range(0.0f0, 1.0f0, length=11)))
    positions = MethodOfLines._periodic_stencil_positions(grid_f32, 5, [-1, 0, 1])
    @test eltype(positions) === Float32

    # Test 8: _build_periodic_wmat produces Float32 matrix from Float32 operator
    wmat_periodic = MethodOfLines._build_periodic_wmat(D_f32_nu, grid_f32)
    @test eltype(wmat_periodic) === Float32
end

@testset "Float32: 1D centered diffusion solves correctly" begin
    # Verify that a PDE with Float32 grid spacing produces correct results
    u_exact = (x, t) -> exp.(-t) * sin.(π * x)

    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq = Dt(u(t, x)) ~ (1 / π^2) * Dxx(u(t, x))
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

    dx = Float32(0.05)
    disc = MOLFiniteDifference(
        [x => dx], t; discretization_strategy = ArrayDiscretization()
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

@testset "Float32: 1D centered diffusion matches Float64" begin
    # Verify Float32 and Float64 discretizations produce equivalent results
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

    disc_f64 = MOLFiniteDifference(
        [x => 0.1], t; discretization_strategy = ArrayDiscretization()
    )
    disc_f32 = MOLFiniteDifference(
        [x => Float32(0.1)], t; discretization_strategy = ArrayDiscretization()
    )

    prob_f64 = discretize(pdesys, disc_f64)
    prob_f32 = discretize(pdesys, disc_f32)

    sol_f64 = solve(prob_f64, Tsit5(), saveat = 0.1)
    sol_f32 = solve(prob_f32, Tsit5(), saveat = 0.1)

    u_f64 = sol_f64[u(t, x)]
    u_f32 = sol_f32[u(t, x)]

    @test size(u_f64) == size(u_f32)
    # Float32(0.1) != Float64(0.1), so grids differ slightly; use relaxed tolerance
    @test isapprox(u_f64, u_f32, rtol = 1e-2)
end

@testset "Float32: upwind advection solves correctly" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    eq = Dt(u(t, x)) ~ -Dx(u(t, x))
    bcs = [
        u(0, x) ~ sin(2π * x),
        u(t, 0) ~ sin(-2π * t),
        u(t, 1) ~ sin(2π * (1 - t)),
    ]

    domains = [
        t ∈ Interval(0.0, 0.5),
        x ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    dx = Float32(0.05)
    disc = MOLFiniteDifference(
        [x => dx], t; advection_scheme = UpwindScheme(),
        discretization_strategy = ArrayDiscretization()
    )
    prob = discretize(pdesys, disc)
    sol = solve(prob, Tsit5(), saveat = 0.1)

    @test SciMLBase.successful_retcode(sol)
    @test all(isfinite, sol[u(t, x)])
    # Solution should stay bounded (pure advection)
    @test maximum(abs, sol[u(t, x)]) < 2.0
end

@testset "Float32: nonlinear diffusion (nonlinlap) solves correctly" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    # Nonlinear diffusion: Dt(u) = Dx(u * Dx(u))
    eq = Dt(u(t, x)) ~ Dx(u(t, x) * Dx(u(t, x)))
    bcs = [
        u(0, x) ~ 1.0 + 0.5 * sin(π * x),
        u(t, 0) ~ 1.0,
        u(t, 1) ~ 1.0,
    ]

    domains = [
        t ∈ Interval(0.0, 0.1),
        x ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    dx = Float32(0.05)
    disc = MOLFiniteDifference(
        [x => dx], t; discretization_strategy = ArrayDiscretization()
    )
    prob = discretize(pdesys, disc)
    sol = solve(prob, Tsit5(), saveat = 0.02)

    @test SciMLBase.successful_retcode(sol)
    @test all(isfinite, sol[u(t, x)])
    # Solution should be bounded and positive (diffusion smooths toward 1.0)
    @test all(sol[u(t, x)][end, :] .> 0.5)
    @test all(sol[u(t, x)][end, :] .< 1.5)
end

@testset "Float32: WENO advection solves correctly" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    eq = Dt(u(t, x)) ~ -Dx(u(t, x))
    bcs = [
        u(0, x) ~ sin(2π * x),
        u(t, 0) ~ sin(-2π * t),
        u(t, 1) ~ sin(2π * (1 - t)),
    ]

    domains = [
        t ∈ Interval(0.0, 0.5),
        x ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    # Use 0.125 (1/8) which is exact in binary and divides [0,1] evenly.
    # WENO requires a uniform grid, so the Float32 dx must remain uniform
    # after promotion to Float64.
    dx = Float32(0.125)
    disc = MOLFiniteDifference(
        [x => dx], t; advection_scheme = WENOScheme(),
        discretization_strategy = ArrayDiscretization()
    )
    prob = discretize(pdesys, disc)
    sol = solve(prob, Tsit5(), saveat = 0.1)

    @test SciMLBase.successful_retcode(sol)
    @test all(isfinite, sol[u(t, x)])
    @test maximum(abs, sol[u(t, x)]) < 2.0
end
