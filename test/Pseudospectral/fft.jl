# FFT-based Fourier derivatives through the AbstractFFTs extension (FFTW backend),
# and the array form in three spatial dimensions.

using ModelingToolkit, MethodOfLines, LinearAlgebra, Test, OrdinaryDiffEq, DomainSets
using FFTW
using SciMLBase: successful_retcode
using ModelingToolkit: get_eqs

const MOL = MethodOfLines

@testset "FFT operators match the dense matrices" begin
    for spec in (
                FourierCollocation(16), FourierCollocation(15), ChebyshevCollocation(17),
                ChebyshevCollocation(16),
            ), d in 1:4
        N = spec.n
        grid = MOL.spectral_grid(spec, -3.0, 5.0)
        D = MOL.spectral_diff_matrix(spec, grid, d)
        op = MOL.fast_spectral_operator(spec, grid, d, D)
        @test op !== nothing
        tol = 1.0e-9 * norm(D, Inf)
        v = randn(N)
        @test norm(MOL.apply_along(op, v, Val(1)) - D * v, Inf) < tol
        X = randn(N, 5)
        @test norm(MOL.apply_along(op, X, Val(1)) - D * X, Inf) < tol
        Y = randn(5, N)
        @test norm(MOL.apply_along(op, Y, Val(2)) - Y * transpose(D), Inf) < tol
        Z = randn(3, N, 4)
        ref = MOL.apply_along(D, Z, Val(2))
        @test norm(MOL.apply_along(op, Z, Val(2)) - ref, Inf) < tol
        for k in 1:4
            @test ref[:, :, k] ≈ Z[:, :, k] * transpose(D)
        end
        # element types without a plan (dual numbers) go through the matrix
        @test norm(MOL.apply_along(op, big.(v), Val(1)) - D * v, Inf) < tol
    end
    # exact for a polynomial of the interpolant's degree
    spec = ChebyshevCollocation(12)
    grid = MOL.spectral_grid(spec, -1.0, 1.0)
    D = MOL.spectral_diff_matrix(spec, grid, 2)
    op = MOL.fast_spectral_operator(spec, grid, 2, D)
    @test MOL.apply_along(op, grid .^ 11, Val(1)) ≈ 110 .* grid .^ 9 atol = 1.0e-9
end

@testset "Chebyshev heat equation with FFT derivatives" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    eq = Dt(u(t, x)) ~ Dxx(u(t, x))
    bcs = [u(0, x) ~ cospi(x / 2), u(t, -1) ~ 0.0, u(t, 1) ~ 0.0]
    domains = [t ∈ Interval(0.0, 0.5), x ∈ Interval(-1.0, 1.0)]
    @named pdesys = PDESystem([eq], bcs, domains, [t, x], [u(t, x)])
    n = 20
    sys, _ = symbolic_discretize(pdesys, PseudospectralDiscretization([x => n], t))
    @test occursin("ChebyshevFFTDerivative", string(get_eqs(sys)[1]))
    sys_dense, _ = symbolic_discretize(pdesys, PseudospectralDiscretization([x => ChebyshevCollocation(n; fft = false)], t))
    @test !occursin("ChebyshevFFTDerivative", string(get_eqs(sys_dense)[1]))
    prob = discretize(pdesys, PseudospectralDiscretization([x => n], t))
    sol = solve(prob; abstol = 1.0e-10, reltol = 1.0e-10)
    @test successful_retcode(sol)
    exact = [exp(-π^2 * ti / 4) * cospi(xi / 2) for ti in sol.t, xi in sol[x]]
    @test norm(sol[u(t, x)] - exact, Inf) < 1.0e-6
end

@testset "Kuramoto-Sivashinsky with FFT derivatives" begin
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

    N = 64
    dense = discretize(pdesys, PseudospectralDiscretization([x => FourierCollocation(N; fft = false)], t))
    fast = discretize(pdesys, PseudospectralDiscretization([x => FourierCollocation(N)], t))
    du = zeros(N + 1)
    rd = similar(du)
    rf = similar(du)
    dense.f(rd, du, dense.u0, dense.p, 0.0)
    fast.f(rf, du, fast.u0, fast.p, 0.0)
    @test norm(rd - rf, Inf) < 1.0e-9
    sys, _ = symbolic_discretize(pdesys, PseudospectralDiscretization([x => FourierCollocation(N)], t))
    @test occursin("FourierFFTDerivative", string(get_eqs(sys)[1]))

    sol_dense = solve(dense; abstol = 1.0e-8, reltol = 1.0e-8, saveat = 0.1)
    sol_fft = solve(fast; abstol = 1.0e-8, reltol = 1.0e-8, saveat = 0.1)
    @test successful_retcode(sol_dense) && successful_retcode(sol_fft)
    @test norm(sol_dense[u(t, x)] - sol_fft[u(t, x)], Inf) < 1.0e-5
end

@testset "Three spatial dimensions" begin
    # u = e^{-(pi^2/2 + 1) t} cos(pi x/2) cos(pi y/2) cos(z)
    @parameters t x y z
    @variables u(..)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    Dzz = Differential(z)^2
    Lp = Float64(π)
    eq = Dt(u(t, x, y, z)) ~ Dxx(u(t, x, y, z)) + Dyy(u(t, x, y, z)) + Dzz(u(t, x, y, z))
    bcs = [
        u(0, x, y, z) ~ cospi(x / 2) * cospi(y / 2) * cos(z),
        u(t, -1, y, z) ~ 0.0, u(t, 1, y, z) ~ 0.0,
        u(t, x, -1, z) ~ 0.0, u(t, x, 1, z) ~ 0.0,
        u(t, x, y, -Lp) ~ u(t, x, y, Lp),
    ]
    domains = [
        t ∈ Interval(0.0, 0.1), x ∈ Interval(-1.0, 1.0), y ∈ Interval(-1.0, 1.0),
        z ∈ Interval(-Lp, Lp),
    ]
    @named pdesys = PDESystem([eq], bcs, domains, [t, x, y, z], [u(t, x, y, z)])
    neqs = map(((8, 8), (12, 10))) do (n, m)
        disc = PseudospectralDiscretization([x => n, y => n, z => FourierCollocation(m)], t)
        sys, _ = symbolic_discretize(pdesys, disc)
        @test length(ModelingToolkit.get_unknowns(sys)) == n * n * (m + 1)
        length(get_eqs(sys))
    end
    # the equation count does not grow with the resolution
    @test neqs[1] == neqs[2] < 40
    disc = PseudospectralDiscretization([x => 10, y => 10, z => FourierCollocation(8)], t)
    sol = solve(discretize(pdesys, disc); abstol = 1.0e-9, reltol = 1.0e-9)
    @test successful_retcode(sol)
    usol = sol[u(t, x, y, z)]
    exact = [
        exp(-(π^2 / 2 + 1) * ti) * cospi(xi / 2) * cospi(yi / 2) * cos(zi)
            for ti in sol.t, xi in sol[x], yi in sol[y], zi in sol[z]
    ]
    # alias index of the periodic direction excluded, as in the 2D test
    @test norm(usol[:, :, :, 2:end] - exact[:, :, :, 2:end], Inf) < 1.0e-6
end
