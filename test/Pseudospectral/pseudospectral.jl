# Pseudospectral discretization tests, modeled on the SciMLBenchmarks
# SimpleHandwrittenPDE spectral benchmarks:
#   allen_cahn_spectral_wpd, burgers_spectral_wpd (Chebyshev collocation)
#   kdv_spectral_wpd, ks_spectral_wpd (Fourier collocation)

using ModelingToolkit, MethodOfLines, LinearAlgebra, Test, OrdinaryDiffEq, DomainSets
using NonlinearSolve
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

# Residual of the `DAEProblem` `discretize` builds, at `u0` with `du` given.
function dae_residual(prob, du)
    res = similar(prob.u0)
    prob.f(res, du, prob.u0, prob.p, 0.0)
    return res
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

@testset "Array form: equation count is independent of resolution" begin
    @parameters t x y
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Dx^2
    Dyy = Differential(y)^2

    eq = Dt(u(t, x)) ~ Dxx(u(t, x)) - u(t, x)^3
    bcs = [u(0, x) ~ cospi(x / 2), u(t, -1) ~ 0.0, Dx(u(t, 1)) ~ 0.0]
    domains = [t ∈ Interval(0.0, 0.5), x ∈ Interval(-1.0, 1.0)]
    @named pdesys = PDESystem([eq], bcs, domains, [t, x], [u(t, x)])
    # one interior array equation plus one scalar equation per boundary
    for n in (8, 40)
        sys, _ = symbolic_discretize(pdesys, PseudospectralDiscretization([x => n], t))
        @test length(get_eqs(sys)) == 3
        @test length(get_unknowns(sys)) == n
    end

    eq2 = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y))
    bcs2 = [
        u(0, x, y) ~ cospi(x / 2) * cos(y), u(t, -1, y) ~ 0.0, Dx(u(t, 1, y)) ~ 0.0,
        u(t, x, -Float64(π)) ~ u(t, x, Float64(π)),
    ]
    domains2 = [
        t ∈ Interval(0.0, 0.2), x ∈ Interval(-1.0, 1.0),
        y ∈ Interval(-Float64(π), Float64(π)),
    ]
    @named pdesys2 = PDESystem([eq2], bcs2, domains2, [t, x, y], [u(t, x, y)])
    # interior, two x faces, the periodic y seam, and the two corners of the seam
    for (n, m) in ((8, 8), (24, 20))
        disc = PseudospectralDiscretization([x => n, y => FourierCollocation(m)], t)
        sys, _ = symbolic_discretize(pdesys2, disc)
        @test length(get_eqs(sys)) == 6
        @test length(get_unknowns(sys)) == n * (m + 1)
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

    prob, perm = ode_rhs(pdesys, disc)
    grid = discmap[x]
    D2 = MOL.spectral_diff_matrix(ChebyshevCollocation(n), grid, 2)
    ufull = ones(n)  # Dirichlet u(+-1) = 1
    ufull[perm] = prob.u0
    manual = (3 .* (ufull .- ufull .^ 3) .+ eps_ .* (D2 * ufull))[perm]
    @test norm(prob.f(prob.u0, prob.p, 0.0) - manual, Inf) < 1.0e-10

    prob_dae = discretize(pdesys, disc)
    @test prob_dae isa DAEProblem
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
    grid = MOL.spectral_grid(ChebyshevCollocation(n), -1.0, 1.0)
    D1 = MOL.spectral_diff_matrix(ChebyshevCollocation(n), grid, 1)
    D2 = MOL.spectral_diff_matrix(ChebyshevCollocation(n), grid, 2)

    # Conservative form Dx(u^2), through the DAE residual of the array form
    eq = Dt(u(t, x)) ~ -Dx(u(t, x)^2) + nu * Dxx(u(t, x))
    @named psys_c = PDESystem([eq], bcs, domains, [t, x], [u(t, x)])
    prob_c = discretize(psys_c, disc)
    u0 = prob_c.u0
    du = zeros(n)
    du[2:(n - 1)] = (-D1 * (u0 .^ 2) .+ nu .* (D2 * u0))[2:(n - 1)]
    @test norm(dae_residual(prob_c, du), Inf) < 1.0e-10

    # Convective form -u * Dx(u), through the compiled ODE path
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
    # the interior array equation plus the periodic alias condition; the
    # derivative matching BCs are satisfied identically and skipped
    @test length(get_eqs(sys)) == 2
    @test length(get_unknowns(sys)) == N + 1

    prob = discretize(pdesys, disc)
    grid = discmap[x]
    D1 = MOL.spectral_diff_matrix(FourierCollocation(N), grid, 1)
    D3 = MOL.spectral_diff_matrix(FourierCollocation(N), grid, 3)
    ud = prob.u0[2:end]
    f = -6 .* ud .* (D1 * ud) .- D3 * ud
    du = vcat(f[end], f)
    @test norm(dae_residual(prob, du), Inf) < 1.0e-9

    # The Fourier differentiation matrix has zero column sums and is
    # skew-symmetric, so the semi-discrete system conserves sum(du) = 0
    # identically.
    @test abs(sum(f)) < 1.0e-10

    sol = solve(prob; abstol = 1.0e-8, reltol = 1.0e-8)
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

@testset "Chebyshev manufactured solutions" begin
    @parameters t x
    @variables u(..) v(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Dx^2
    domains = [t ∈ Interval(0.0, 0.5), x ∈ Interval(-1.0, 1.0)]

    function solve_and_compare(pdesys, disc, exact; tol = 1.0e-6, kwargs...)
        sol = solve(discretize(pdesys, disc); abstol = 1.0e-10, reltol = 1.0e-10)
        @test successful_retcode(sol)
        ex = [exact(ti, xi) for ti in sol.t, xi in sol[x]]
        @test norm(sol[u(t, x)] - ex, Inf) < tol
        return sol
    end

    @testset "Dirichlet" begin
        # u = e^{-pi^2 t/4} cos(pi x/2)
        eq = Dt(u(t, x)) ~ Dxx(u(t, x))
        bcs = [u(0, x) ~ cospi(x / 2), u(t, -1) ~ 0.0, u(t, 1) ~ 0.0]
        @named pdesys = PDESystem([eq], bcs, domains, [t, x], [u(t, x)])
        solve_and_compare(
            pdesys, PseudospectralDiscretization([x => 16], t),
            (ti, xi) -> exp(-π^2 * ti / 4) * cospi(xi / 2)
        )
        # the same problem on a custom (here Chebyshev-Lobatto) node vector
        nodes = -cospi.((0:15) ./ 15)
        solve_and_compare(
            pdesys, PseudospectralDiscretization([x => nodes], t),
            (ti, xi) -> exp(-π^2 * ti / 4) * cospi(xi / 2)
        )
    end

    @testset "Neumann" begin
        # u = e^{-pi^2 t/4} sin(pi x/2), u_x(t, +-1) = 0
        eq = Dt(u(t, x)) ~ Dxx(u(t, x))
        bcs = [u(0, x) ~ sinpi(x / 2), Dx(u(t, -1)) ~ 0.0, Dx(u(t, 1)) ~ 0.0]
        @named pdesys = PDESystem([eq], bcs, domains, [t, x], [u(t, x)])
        solve_and_compare(
            pdesys, PseudospectralDiscretization([x => 20], t),
            (ti, xi) -> exp(-π^2 * ti / 4) * sinpi(xi / 2)
        )
    end

    @testset "Robin" begin
        # u = e^{-t} e^{x}: u_t = u_xx - 2u, u_x - u = 0 at both ends
        eq = Dt(u(t, x)) ~ Dxx(u(t, x)) - 2u(t, x)
        bcs = [u(0, x) ~ exp(x), Dx(u(t, -1)) - u(t, -1) ~ 0.0, Dx(u(t, 1)) ~ u(t, 1)]
        @named pdesys = PDESystem([eq], bcs, domains, [t, x], [u(t, x)])
        solve_and_compare(
            pdesys, PseudospectralDiscretization([x => 20], t),
            (ti, xi) -> exp(-ti + xi)
        )
    end

    @testset "Time-dependent Dirichlet" begin
        # u = e^{t + x}
        eq = Dt(u(t, x)) ~ Dxx(u(t, x))
        bcs = [u(0, x) ~ exp(x), u(t, -1) ~ exp(t - 1), u(t, 1) ~ exp(t + 1)]
        @named pdesys = PDESystem([eq], bcs, domains, [t, x], [u(t, x)])
        solve_and_compare(
            pdesys, PseudospectralDiscretization([x => 20], t),
            (ti, xi) -> exp(ti + xi)
        )
    end

    @testset "Second-order boundary condition" begin
        # u = e^{-pi^2 t} sin(pi x), u_xx(t, +-1) = 0
        eq = Dt(u(t, x)) ~ Dxx(u(t, x))
        bcs = [u(0, x) ~ sinpi(x), Dxx(u(t, -1)) ~ 0.0, Dxx(u(t, 1)) ~ 0.0]
        @named pdesys = PDESystem([eq], bcs, domains, [t, x], [u(t, x)])
        solve_and_compare(
            pdesys, PseudospectralDiscretization([x => 16], t),
            (ti, xi) -> exp(-π^2 * ti) * sinpi(xi); tol = 1.0e-5
        )
    end

    @testset "Nested nonlinear laplacian" begin
        # u = e^{-t}(1 - x^2), forced so that u_t = Dx((1 + x^2) Dx(u)) + g - u
        eq = Dt(u(t, x)) ~ Dx((1 + x^2) * Dx(u(t, x))) + exp(-t) * (2 + 6x^2) - u(t, x)
        bcs = [u(0, x) ~ 1 - x^2, u(t, -1) ~ 0.0, u(t, 1) ~ 0.0]
        @named pdesys = PDESystem([eq], bcs, domains, [t, x], [u(t, x)])
        solve_and_compare(
            pdesys, PseudospectralDiscretization([x => 12], t),
            (ti, xi) -> exp(-ti) * (1 - xi^2)
        )
    end

    @testset "Coupled system" begin
        # u = e^{-pi^2 t/4} cos(pi x/2) cos t, v = -e^{-pi^2 t/4} cos(pi x/2) sin t
        eqs = [
            Dt(u(t, x)) ~ Dxx(u(t, x)) + v(t, x),
            Dt(v(t, x)) ~ Dxx(v(t, x)) - u(t, x),
        ]
        bcs = [
            u(0, x) ~ cospi(x / 2), v(0, x) ~ 0.0,
            u(t, -1) ~ 0.0, u(t, 1) ~ 0.0, v(t, -1) ~ 0.0, v(t, 1) ~ 0.0,
        ]
        @named pdesys = PDESystem(eqs, bcs, domains, [t, x], [u(t, x), v(t, x)])
        disc = PseudospectralDiscretization([x => 16], t)
        sys, _ = symbolic_discretize(pdesys, disc)
        @test length(get_eqs(sys)) == 6
        sol = solve(discretize(pdesys, disc); abstol = 1.0e-10, reltol = 1.0e-10)
        @test successful_retcode(sol)
        exu = [exp(-π^2 * ti / 4) * cospi(xi / 2) * cos(ti) for ti in sol.t, xi in sol[x]]
        exv = [-exp(-π^2 * ti / 4) * cospi(xi / 2) * sin(ti) for ti in sol.t, xi in sol[x]]
        @test norm(sol[u(t, x)] - exu, Inf) < 1.0e-6
        @test norm(sol[v(t, x)] - exv, Inf) < 1.0e-6
    end
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

    for N in (16, 15)
        disc = PseudospectralDiscretization([x => FourierCollocation(N)], t)
        sol = solve(discretize(pdesys, disc); abstol = 1.0e-10, reltol = 1.0e-10)
        exact = [exp(-ti) * cos(xi) for ti in sol.t, xi in sol[x]]
        @test norm(sol[u(t, x)] - exact, Inf) < 1.0e-6
    end
end

@testset "Two spatial dimensions" begin
    @parameters t x y
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Dx^2
    Dyy = Differential(y)^2
    eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y))

    @testset "Chebyshev x Chebyshev, Dirichlet" begin
        # u = e^{-pi^2 t/2} cos(pi x/2) cos(pi y/2)
        bcs = [
            u(0, x, y) ~ cospi(x / 2) * cospi(y / 2),
            u(t, -1, y) ~ 0.0, u(t, 1, y) ~ 0.0, u(t, x, -1) ~ 0.0, u(t, x, 1) ~ 0.0,
        ]
        domains = [
            t ∈ Interval(0.0, 0.2), x ∈ Interval(-1.0, 1.0), y ∈ Interval(-1.0, 1.0),
        ]
        @named pdesys = PDESystem([eq], bcs, domains, [t, x, y], [u(t, x, y)])
        disc = PseudospectralDiscretization([x => 12, y => 12], t)
        sys, _ = symbolic_discretize(pdesys, disc)
        # interior, four faces, four corners
        @test length(get_eqs(sys)) == 9
        sol = solve(discretize(pdesys, disc); abstol = 1.0e-9, reltol = 1.0e-9)
        @test successful_retcode(sol)
        usol = sol[u(t, x, y)]
        exact = [
            exp(-π^2 * ti / 2) * cospi(xi / 2) * cospi(yi / 2)
                for ti in sol.t, xi in sol[x], yi in sol[y]
        ]
        @test norm(usol - exact, Inf) < 1.0e-6
    end

    @testset "Chebyshev x Fourier, Neumann face" begin
        # u = e^{-(pi^2/16 + 1) t} sin(pi (x + 1)/4) cos(y): u(t, -1, y) = 0, u_x(t, 1, y) = 0
        Lp = Float64(π)
        bcs = [
            u(0, x, y) ~ sinpi((x + 1) / 4) * cos(y),
            u(t, -1, y) ~ 0.0, Dx(u(t, 1, y)) ~ 0.0, u(t, x, -Lp) ~ u(t, x, Lp),
        ]
        domains = [
            t ∈ Interval(0.0, 0.2), x ∈ Interval(-1.0, 1.0), y ∈ Interval(-Lp, Lp),
        ]
        @named pdesys = PDESystem([eq], bcs, domains, [t, x, y], [u(t, x, y)])
        disc = PseudospectralDiscretization([x => 14, y => FourierCollocation(12)], t)
        sol = solve(discretize(pdesys, disc); abstol = 1.0e-9, reltol = 1.0e-9)
        @test successful_retcode(sol)
        usol = sol[u(t, x, y)]
        exact = [
            exp(-(π^2 / 16 + 1) * ti) * sinpi((xi + 1) / 4) * cos(yi)
                for ti in sol.t, xi in sol[x], yi in sol[y]
        ]
        # The corner points at the periodic alias index (y = -pi) are pinned to
        # zero by the corner equations, as on the finite difference path, so
        # compare on the distinct periodic points only.
        @test norm(usol[:, :, 2:end] - exact[:, :, 2:end], Inf) < 1.0e-6
    end
end

@testset "Stationary problem" begin
    # u'' = -pi^2/4 cos(pi x/2), u(+-1) = 0  =>  u = cos(pi x/2)
    @parameters x
    @variables u(..)
    Dxx = Differential(x)^2
    eq = Dxx(u(x)) ~ -π^2 / 4 * cospi(x / 2)
    bcs = [u(-1) ~ 0.0, u(1) ~ 0.0]
    @named pdesys = PDESystem([eq], bcs, [x ∈ Interval(-1.0, 1.0)], [x], [u(x)])
    prob = discretize(pdesys, PseudospectralDiscretization([x => 16]))
    @test prob isa NonlinearProblem
    sol = solve(prob, NewtonRaphson())
    @test successful_retcode(sol)
    @test norm(sol[u(x)] - cospi.(sol[x] ./ 2), Inf) < 1.0e-10
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

    # A jump in a derivative matching condition cannot be represented
    bcs = [u(0, x) ~ 0.0, u(t, 0) ~ u(t, 1), Dx(u(t, 0)) ~ Dx(u(t, 1)) + 1.0]
    @named psys4 = PDESystem([eq], bcs, domains, [t, x], [u(t, x)])
    @test_throws ArgumentError symbolic_discretize(
        psys4, PseudospectralDiscretization([x => FourierCollocation(8)], t)
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

    @test_throws ArgumentError ChebyshevCollocation(1)
    @test_throws ArgumentError FourierCollocation(3)
    @test_throws ArgumentError PseudospectralDiscretization([x => "eight"], t)
end
