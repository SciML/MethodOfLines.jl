# Pseudospectral discretization tests, modeled on the SciMLBenchmarks
# SimpleHandwrittenPDE spectral benchmarks:
#   allen_cahn_spectral_wpd, burgers_spectral_wpd (Chebyshev collocation)
#   kdv_spectral_wpd, ks_spectral_wpd (Fourier collocation)

using ModelingToolkit, MethodOfLines, LinearAlgebra, Test, OrdinaryDiffEq, DomainSets
using SciMLBase: successful_retcode
using ModelingToolkit: Differential, get_eqs, mtkcompile, get_unknowns, Symbolics

const MOL = MethodOfLines

# Independent reference: Chebyshev-Lobatto differentiation matrix, Trefethen
# "Spectral Methods in MATLAB" construction on the descending grid x_j = cos(pi*j/N).
function cheb_matrix_ref(N)
    x = cospi.((0:N) / N)
    c = [2; ones(N - 1); 2] .* (-1) .^ (0:N)
    X = repeat(x, 1, N + 1)
    dX = X - permutedims(X)
    D = (c * permutedims(1 ./ c)) ./ (dX + I)
    return D - Diagonal(vec(sum(D, dims = 2)))
end

# Map compiled unknowns `u(t)[k]` back to their grid indices.
function unknown_perm(simpsys)
    return map(get_unknowns(simpsys)) do uk
        Int(Symbolics.value(arguments(Symbolics.unwrap(uk))[2]))
    end
end

function ode_rhs(pdesys, disc)
    sys, tspan = symbolic_discretize(pdesys, disc)
    simpsys = mtkcompile(sys)
    return ODEProblem(simpsys, nothing, tspan), unknown_perm(simpsys)
end

@testset "Collocation grids and differentiation matrices" begin
    n = 8
    grid = MOL.spectral_grid(ChebyshevCollocation(n), -1.0, 1.0)
    @test grid ≈ -cospi.((0:(n - 1)) / (n - 1))
    D2 = MOL.spectral_diff_matrix(ChebyshevCollocation(n), grid, 2)
    Dref = (cheb_matrix_ref(n - 1)^2)[end:-1:1, end:-1:1]
    @test D2 ≈ Dref atol = 1.0e-8

    N = 16
    a, b = -10.0, 10.0
    gridf = MOL.spectral_grid(FourierCollocation(N), a, b)
    @test length(gridf) == N + 1
    @test gridf[end] - gridf[1] ≈ b - a
    ξ = gridf[1:(end - 1)]
    u = sin.(2π .* ξ ./ (b - a))
    for (d, exact) in (
            1 => (2π / (b - a)) .* cos.(2π .* ξ ./ (b - a)),
            3 => -(2π / (b - a))^3 .* cos.(2π .* ξ ./ (b - a)),
        )
        Dd = MOL.spectral_diff_matrix(FourierCollocation(N), gridf, d)
        @test norm(Dd * u - exact, Inf) < 1.0e-10
    end
end

@testset "Allen-Cahn, Chebyshev collocation" begin
    # u_t = 3(u - u^3) + eps u_xx, u(0,x) = cos(2pi x), u(t,+-1) = 1
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eps_ = 1.0e-3
    eq = Dt(u(t, x)) ~ 3 * (u(t, x) - u(t, x)^3) + eps_ * Dxx(u(t, x))
    bcs = [u(0, x) ~ cospi(2x), u(t, -1) ~ 1.0, u(t, 1) ~ 1.0]
    domains = [t ∈ Interval(0.0, 0.5), x ∈ Interval(-1.0, 1.0)]
    @named pdesys = PDESystem([eq], bcs, domains, [t, x], [u(t, x)])

    n = 32
    disc = PseudospectralDiscretization([x => n], t)
    discmap = get_discrete(pdesys, disc)
    @test length(discmap[x]) == n

    sys, tspan = symbolic_discretize(pdesys, disc)
    # interior collocation equations plus two boundary conditions
    @test length(get_eqs(sys)) == n

    prob, perm = ode_rhs(pdesys, disc)
    grid = discmap[x]
    D2 = MOL.spectral_diff_matrix(ChebyshevCollocation(n), grid, 2)
    ufull = ones(n)  # Dirichlet u(+-1) = 1
    ufull[perm] = prob.u0
    manual = (3 .* (ufull .- ufull .^ 3) .+ eps_ .* (D2 * ufull))[perm]
    @test norm(prob.f(prob.u0, prob.p, 0.0) - manual, Inf) < 1.0e-10

    prob_dae = discretize(pdesys, disc)
    sol = solve(prob_dae; abstol = 1.0e-8, reltol = 1.0e-8)
    @test successful_retcode(sol)
    usol = sol[u(t, x)]
    @test size(usol, 2) == n
    @test all(isfinite, usol)
    @test all(<(1.5), abs.(usol))
end

@testset "Burgers, Chebyshev collocation" begin
    # u_t = -Dx(u^2) + nu u_xx, u(0,x) = exp(-x^2/(2*0.1^2)), u(t,+-1) = 0
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Dx^2

    nu = 1.0e-2
    u0func(x) = exp(-x^2 / (2 * 0.1^2))
    bcs = [u(0, x) ~ u0func(x), u(t, -1) ~ 0.0, u(t, 1) ~ 0.0]
    domains = [t ∈ Interval(0.0, 0.3), x ∈ Interval(-1.0, 1.0)]

    n = 32
    disc = PseudospectralDiscretization([x => n], t)

    # Conservative form Dx(u^2)
    eq = Dt(u(t, x)) ~ -Dx(u(t, x)^2) + nu * Dxx(u(t, x))
    @named psys_c = PDESystem([eq], bcs, domains, [t, x], [u(t, x)])
    prob_c, perm_c = ode_rhs(psys_c, disc)
    grid = MOL.spectral_grid(ChebyshevCollocation(n), -1.0, 1.0)
    D1 = MOL.spectral_diff_matrix(ChebyshevCollocation(n), grid, 1)
    D2 = MOL.spectral_diff_matrix(ChebyshevCollocation(n), grid, 2)
    ufull = zeros(n)
    ufull[perm_c] = prob_c.u0
    manual = (-D1 * (ufull .^ 2) .+ nu .* (D2 * ufull))[perm_c]
    @test norm(prob_c.f(prob_c.u0, prob_c.p, 0.0) - manual, Inf) < 1.0e-10

    # Convective form -u * Dx(u): equal up to collocation aliasing
    eq = Dt(u(t, x)) ~ -u(t, x) * Dx(u(t, x)) + nu * Dxx(u(t, x))
    @named psys_nc = PDESystem([eq], bcs, domains, [t, x], [u(t, x)])
    prob_nc, perm_nc = ode_rhs(psys_nc, disc)
    ufull_nc = zeros(n)
    ufull_nc[perm_nc] = prob_nc.u0
    manual_nc = (-ufull_nc .* (D1 * ufull_nc) .+ nu .* (D2 * ufull_nc))[perm_nc]
    @test norm(prob_nc.f(prob_nc.u0, prob_nc.p, 0.0) - manual_nc, Inf) < 1.0e-10

    sol = solve(discretize(psys_nc, disc); abstol = 1.0e-8, reltol = 1.0e-8)
    @test successful_retcode(sol)
    @test all(isfinite, sol[u(t, x)])
end

@testset "KdV, Fourier collocation" begin
    # u_t = -6 u u_x - u_xxx, u(0,x) = cos(pi x/L), periodic on [-L, L]
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxxx = Dx^3

    L = 16.0
    eq = Dt(u(t, x)) ~ -6 * u(t, x) * Dx(u(t, x)) - Dxxx(u(t, x))
    bcs = [
        u(0, x) ~ cospi(x / L),
        u(t, -L) ~ u(t, L),
        Dx(u(t, -L)) ~ Dx(u(t, L)),
        (Dx^2)(u(t, -L)) ~ (Dx^2)(u(t, L)),
    ]
    domains = [t ∈ Interval(0.0, 0.5), x ∈ Interval(-L, L)]
    @named pdesys = PDESystem([eq], bcs, domains, [t, x], [u(t, x)])

    N = 32
    disc = PseudospectralDiscretization([x => FourierCollocation(N)], t)
    discmap = get_discrete(pdesys, disc)
    @test length(discmap[x]) == N + 1

    sys, tspan = symbolic_discretize(pdesys, disc)
    # N interior equations + the periodic alias condition; the derivative
    # matching BCs are satisfied identically and skipped
    @test length(get_eqs(sys)) == N + 1

    prob, perm = ode_rhs(pdesys, disc)
    grid = discmap[x]
    D1 = MOL.spectral_diff_matrix(FourierCollocation(N), grid, 1)
    D3 = MOL.spectral_diff_matrix(FourierCollocation(N), grid, 3)
    ufull = zeros(N + 1)
    ufull[perm] = prob.u0
    ufull[1] = ufull[end]
    udistinct = ufull[2:end]
    manual = (-6 .* udistinct .* (D1 * udistinct) .- D3 * udistinct)[perm .- 1]
    @test norm(prob.f(prob.u0, prob.p, 0.0) - manual, Inf) < 1.0e-9

    # The Fourier differentiation matrix has zero column sums and is
    # skew-symmetric, so the semi-discrete system conserves sum(du) = 0
    # identically.
    du0 = prob.f(prob.u0, prob.p, 0.0)
    @test abs(sum(du0)) < 1.0e-10

    sol = solve(discretize(pdesys, disc); abstol = 1.0e-8, reltol = 1.0e-8)
    @test successful_retcode(sol)
    usol = sol[u(t, x)]
    @test all(isfinite, usol)
end

@testset "Kuramoto-Sivashinsky, Fourier collocation" begin
    # u_t = -u u_x - 1/2 u_xx - 1/16 u_xxxx, u(0,x) = cos(2pi x/L), periodic
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    L = 16.0
    eq = Dt(u(t, x)) ~ -u(t, x) * Dx(u(t, x)) - (Dx^2)(u(t, x)) / 2 -
        (Dx^4)(u(t, x)) / 16
    bcs = [u(0, x) ~ cospi(2x / L), u(t, -L) ~ u(t, L)]
    domains = [t ∈ Interval(0.0, 0.5), x ∈ Interval(-L, L)]
    @named pdesys = PDESystem([eq], bcs, domains, [t, x], [u(t, x)])

    N = 32
    disc = PseudospectralDiscretization([x => FourierCollocation(N)], t)
    prob, perm = ode_rhs(pdesys, disc)
    grid = get_discrete(pdesys, disc)[x]
    D1 = MOL.spectral_diff_matrix(FourierCollocation(N), grid, 1)
    D2 = MOL.spectral_diff_matrix(FourierCollocation(N), grid, 2)
    D4 = MOL.spectral_diff_matrix(FourierCollocation(N), grid, 4)
    ufull = zeros(N + 1)
    ufull[perm] = prob.u0
    ufull[1] = ufull[end]
    udistinct = ufull[2:end]
    manual = (
        -udistinct .* (D1 * udistinct) .- (D2 * udistinct) / 2 .-
            (D4 * udistinct) / 16
    )[perm .- 1]
    @test norm(prob.f(prob.u0, prob.p, 0.0) - manual, Inf) < 1.0e-9

    sol = solve(discretize(pdesys, disc); abstol = 1.0e-8, reltol = 1.0e-8)
    @test successful_retcode(sol)
    @test all(isfinite, sol[u(t, x)])
end

@testset "Chebyshev manufactured solution" begin
    # u = e^{-pi^2 t/4} cos(pi x/2), Dirichlet u(t,+-1) = 0
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    eq = Dt(u(t, x)) ~ Dxx(u(t, x))
    bcs = [u(0, x) ~ cospi(x / 2), u(t, -1) ~ 0.0, u(t, 1) ~ 0.0]
    domains = [t ∈ Interval(0.0, 0.5), x ∈ Interval(-1.0, 1.0)]
    @named pdesys = PDESystem([eq], bcs, domains, [t, x], [u(t, x)])

    disc = PseudospectralDiscretization([x => 16], t)
    sol = solve(discretize(pdesys, disc); abstol = 1.0e-10, reltol = 1.0e-10)
    exact = [exp(-π^2 * ti / 4) * cospi(xi / 2) for ti in sol.t, xi in sol[x]]
    @test norm(sol[u(t, x)] - exact, Inf) < 1.0e-6
end

@testset "Fourier manufactured solution" begin
    # u = e^{-t} cos(x), periodic on [-pi, pi]
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2

    L = Float64(π)
    eq = Dt(u(t, x)) ~ Dxx(u(t, x))
    bcs = [u(0, x) ~ cos(x), u(t, -L) ~ u(t, L)]
    domains = [t ∈ Interval(0.0, 0.5), x ∈ Interval(-L, L)]
    @named pdesys = PDESystem([eq], bcs, domains, [t, x], [u(t, x)])

    disc = PseudospectralDiscretization([x => FourierCollocation(16)], t)
    sol = solve(discretize(pdesys, disc); abstol = 1.0e-10, reltol = 1.0e-10)
    exact = [exp(-ti) * cos(xi) for ti in sol.t, xi in sol[x]]
    @test norm(sol[u(t, x)] - exact, Inf) < 1.0e-6
end

@testset "PseudospectralDiscretization errors" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    eq = Dt(u(t, x)) ~ (Dx^2)(u(t, x))
    domains = [t ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0)]

    # FourierCollocation without a periodic boundary condition
    bcs = [u(0, x) ~ 0.0, u(t, 0) ~ 0.0, u(t, 1) ~ 0.0]
    @named psys1 = PDESystem([eq], bcs, domains, [t, x], [u(t, x)])
    @test_throws ArgumentError symbolic_discretize(
        psys1, PseudospectralDiscretization([x => FourierCollocation(8)], t)
    )

    # Periodic boundary condition on a non-Fourier direction
    bcs = [u(0, x) ~ 0.0, u(t, 0) ~ u(t, 1)]
    @named psys2 = PDESystem([eq], bcs, domains, [t, x], [u(t, x)])
    @test_throws ArgumentError symbolic_discretize(
        psys2, PseudospectralDiscretization([x => 8], t)
    )

    # Missing collocation specification
    @parameters y
    domains2 = [
        t ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0),
    ]
    eq2 = Dt(u(t, x, y)) ~ (Dx^2)(u(t, x, y)) + (Differential(y)^2)(u(t, x, y))
    bcs2 = [
        u(0, x, y) ~ 0.0, u(t, 0, y) ~ 0.0, u(t, 1, y) ~ 0.0,
        u(t, x, 0) ~ u(t, x, 1),
    ]
    @named psys3 = PDESystem([eq2], bcs2, domains2, [t, x, y], [u(t, x, y)])
    @test_throws AssertionError symbolic_discretize(
        psys3, PseudospectralDiscretization([x => 8], t)
    )
end
