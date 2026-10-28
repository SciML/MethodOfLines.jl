# 1D diffusion problem

# Packages and inclusions
using ModelingToolkit, MethodOfLines, LinearAlgebra, Test
using ModelingToolkit: Differential, get_unknowns
using NonlinearSolve, SciMLBase
using DomainSets

# Handrolled discrete laplace problem
@testset "Hand rolled 1D Laplace" begin
    @variables a b c d
    eqs = [
        0 ~ a - 1,
        0 ~ a + c - 2 * b,
        0 ~ b + d - 2 * c,
        0 ~ d - 1,
    ]
    @named ns = System(eqs, [a, b, c, d], [])
    sns = mtkcompile(ns)
    prob = NonlinearProblem(sns, Dict(get_unknowns(sns) .=> 0.0))

    sol = NonlinearSolve.solve(prob, NewtonRaphson())

    @test map(v -> sol[v], [a, b, c, d]) ≈ ones(4)
end

# Laplace's Equation, same as above but with MOL discretization
@testset "1D Laplace - constant solution" begin
    @parameters x
    @variables u(..)
    Dxx = Differential(x)^2

    eq = Dxx(u(x)) ~ 0
    dx = 1 / 3

    bcs = [
        u(0) ~ 1,
        u(1) ~ 1,
    ]

    # Space and time domains
    domains = [x ∈ Interval(0.0, 1.0)]

    @named pdesys = PDESystem([eq], bcs, domains, [x], [u(x)])
    discretization = MOLFiniteDifference([x => dx], nothing, approx_order = 2)
    prob = discretize(pdesys, discretization)

    sol = NonlinearSolve.solve(prob, NewtonRaphson())
    sol′ = sol[u(x)]
    @test sol′ ≈ ones(4)
end

# Laplace's Equation, linear solution
@testset "1D Laplace - linear solution" begin
    @parameters x
    @variables u(..)
    Dxx = Differential(x)^2

    eq = Dxx(u(x)) ~ 0
    dx = 0.1

    bcs = [
        u(0) ~ 1,
        u(1) ~ 2,
    ]

    # Space and time domains
    domains = [x ∈ Interval(0.0, 1.0)]

    @named pdesys = PDESystem([eq], bcs, domains, [x], [u(x)])
    discretization = MOLFiniteDifference([x => dx], nothing, approx_order = 2)

    prob = discretize(pdesys, discretization)
    sol = NonlinearSolve.solve(prob, NewtonRaphson())
    sol′ = sol[u(x)]

    @test sol′ ≈ 1.0:0.1:2.0
end

# 2D heat
@testset "2D heat equation" begin
    @parameters x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ 0
    dx = 0.1
    dy = 0.1

    bcs = [
        u(0, y) ~ x * y,
        u(1, y) ~ x * y,
        u(x, 0) ~ x * y,
        u(x, 1) ~ x * y,
    ]

    # Space and time domains
    domains = [
        x ∈ Interval(0.0, 1.0),
        y ∈ Interval(0.0, 1.0),
    ]

    @named pdesys = PDESystem([eq], bcs, domains, [x, y], [u(x, y)])

    # Note that we pass in `nothing` for the time variable `t` here since we
    # are creating a stationary problem without a dependence on time, only space.
    discretization = MOLFiniteDifference([x => dx, y => dy], nothing, approx_order = 2)

    prob = discretize(pdesys, discretization)
    sol = NonlinearSolve.solve(prob, NewtonRaphson())

    xs = sol[x]
    ys = sol[y]
    u_sol = sol[u(x, y)]

    # test boundary
    @test all(abs.(u_sol[:, 1]) .< eps(Float32))
    @test all(abs.(u_sol[1, :]) .< eps(Float32))
    @test u_sol[2:(end - 1), end] ≈ (0.0:dy:1.0)[2:(end - 1)]
    @test u_sol[end, 2:(end - 1)] ≈ (0.0:dx:1.0)[2:(end - 1)]
    # `end` would ask the solution for the field's axes.
    @test_throws "cannot be resolved" sol[u(x, y), end, :]

    # test interior with finite differences
    interior = CartesianIndices((axes(xs)[1], axes(ys)[1]))[2:(end - 1), 2:(end - 1)]
    fd = map(interior) do I
        abs(
            u_sol[(I - CartesianIndex(1, 0))] + u_sol[(I + CartesianIndex(1, 0))] -
                2 * u_sol[I]
        ) < 0.01
    end #
    @test all(fd)
end

# Structural simplification used to eliminate most of these unknowns into observed
# expressions (8 of 81 left on 9x9) and Newton then stalled on 11x11.
@testset "2D Laplace - exact solution u = x*y, $bc_style boundary" for bc_style in
    (:Dirichlet, :Neumann)
    @parameters x y
    @variables u(..)
    Dx = Differential(x)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ 0
    bcs = [
        u(0, y) ~ 0,
        bc_style === :Dirichlet ? u(1, y) ~ y : Dx(u(1, y)) ~ y,
        u(x, 0) ~ 0,
        u(x, 1) ~ x,
    ]
    domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]

    for n in (9, 11)
        h = 1 / (n - 1)
        @named pdesys = PDESystem([eq], bcs, domains, [x, y], [u(x, y)])
        prob = discretize(pdesys, MOLFiniteDifference([x => h, y => h], nothing, approx_order = 2))
        # (n - 2)^2 interior unknowns plus the boundary nodes the boundary conditions fix
        @test length(prob.u0) == n^2

        sol = NonlinearSolve.solve(prob, NewtonRaphson())
        @test sol.retcode == ReturnCode.Success
        grid = range(0, 1, length = n)
        interior = 2:(n - 1)
        @test sol[u(x, y)][interior, interior] ≈ [xi * yi for xi in grid[interior], yi in grid[interior]] atol = 1.0e-10
    end
end

@testset "Stationary problems use a sparse Jacobian unless told otherwise" begin
    @parameters x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]
    h = 0.1

    @named pdesys1 = PDESystem(
        [Dxx(u(x)) ~ 0], [u(0) ~ 1, u(1) ~ 2], [x ∈ Interval(0.0, 1.0)], [x], [u(x)]
    )
    prob1 = discretize(pdesys1, MOLFiniteDifference([x => h], nothing))
    @test nameof(typeof(prob1.f.jac_prototype)) === :SparseMatrixCSC
    @test NonlinearSolve.solve(prob1, NewtonRaphson()).retcode == ReturnCode.Success

    eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ 0
    bcs = [u(0, y) ~ 0, u(1, y) ~ y, u(x, 0) ~ 0, u(x, 1) ~ x]
    @named pdesys2 = PDESystem([eq], bcs, domains, [x, y], [u(x, y)])
    disc = MOLFiniteDifference([x => h, y => h], nothing)
    prob2 = discretize(pdesys2, disc)
    @test nameof(typeof(prob2.f.jac_prototype)) === :SparseMatrixCSC
    @test NonlinearSolve.solve(prob2, NewtonRaphson()).retcode == ReturnCode.Success

    dense = discretize(pdesys2, disc; sparse = false)
    @test nameof(typeof(dense.f.jac_prototype)) !== :SparseMatrixCSC
    @test NonlinearSolve.solve(dense, NewtonRaphson()).retcode == ReturnCode.Success
end

@testset "2D Laplace - default tolerance succeeds on fine grids" begin
    @parameters x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ 0
    bcs = [u(0, y) ~ 0, u(1, y) ~ y, u(x, 0) ~ 0, u(x, 1) ~ x]
    domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]
    n = 31
    h = 1 / (n - 1)
    @named pdesys = PDESystem([eq], bcs, domains, [x, y], [u(x, y)])
    disc = MOLFiniteDifference([x => h, y => h], nothing)

    # The roundoff floor of the 1/h^2 stencil residual exceeds NonlinearSolve's default abstol.
    prob = discretize(pdesys, disc)
    sol = NonlinearSolve.solve(prob, NewtonRaphson())
    @test SciMLBase.successful_retcode(sol)
    grid = range(0, 1, length = n)
    interior = 2:(n - 1)
    @test sol[u(x, y)][interior, interior] ≈ [xi * yi for xi in grid[interior], yi in grid[interior]] atol = 1.0e-10

    # The tolerance also scales without a symbolic Jacobian (the path used on Julia 1.10).
    nojac = discretize(pdesys, disc; jac = false)
    @test SciMLBase.successful_retcode(NonlinearSolve.solve(nojac, NewtonRaphson()))

    # An explicit abstol is respected.
    @test discretize(pdesys, disc; abstol = 1.0e-9).kwargs[:abstol] == 1.0e-9
end

# Systems with unequal equation and unknown counts never reach the stationary path:
# PDEBase rejects them while discretizing.
@testset "Non-square stationary systems are rejected" begin
    @parameters x
    @variables u(..)
    Dxx = Differential(x)^2
    @named pdesys = PDESystem(
        [Dxx(u(x)) ~ 0, Dxx(u(x)) ~ 1], [u(0) ~ 1, u(1) ~ 2],
        [x ∈ Interval(0.0, 1.0)], [x], [u(x)]
    )
    @test_throws "same number of equations and unknowns" discretize(
        pdesys, MOLFiniteDifference([x => 0.25], nothing)
    )
end

@testset "Stationary default abstol is bounded and optional" begin
    @parameters x
    @variables u(..)
    Dxx = Differential(x)^2
    domains = [x ∈ Interval(0.0, 1.0)]
    disc = MOLFiniteDifference([x => 0.25], nothing)

    # J(u0) is ~e^100 here; an unbounded Jacobian-scaled abstol would accept any iterate.
    @named stiff = PDESystem(
        [Dxx(u(x)) - (exp(100 * u(x)) - 1) ~ 0], [u(0) ~ 0, u(1) ~ 0], domains, [x], [u(x)]
    )
    prob = discretize(stiff, disc)
    @test prob.kwargs[:abstol] <= 1.0e-8
    sol = NonlinearSolve.solve(prob, NewtonRaphson())
    @test !SciMLBase.successful_retcode(sol) || maximum(abs, sol[u(x)]) < 1.0e-6

    # Without a symbolic Jacobian the scale comes from a finite-difference probe instead.
    @named linear = PDESystem([Dxx(u(x)) ~ 0], [u(0) ~ 1, u(1) ~ 2], domains, [x], [u(x)])
    nojac = discretize(linear, disc; jac = false)
    @test eps()^(4 // 5) <= nojac.kwargs[:abstol] <= 1.0e-8
    @test NonlinearSolve.solve(nojac, NewtonRaphson()).retcode == ReturnCode.Success

    # J(u0 = 1) throws a DomainError for sqrt(u - 2); building the problem must not.
    @named domain = PDESystem(
        [Dxx(u(x)) + sqrt(u(x) - 2) ~ 0], [u(0) ~ 3, u(1) ~ 3], domains, [x], [u(x)]
    )
    guarded = discretize(domain, disc)
    @test !haskey(guarded.kwargs, :abstol)
    sol = NonlinearSolve.solve(remake(guarded; u0 = fill(3.0, length(guarded.u0))), NewtonRaphson())
    @test SciMLBase.successful_retcode(sol)
end
