# FFT-based Fourier derivatives through the AbstractFFTs extension (FFTW backend),
# and the array form in three spatial dimensions.

using ModelingToolkit, MethodOfLines, LinearAlgebra, Test, OrdinaryDiffEq, DomainSets
using FFTW
using SciMLBase: successful_retcode
using ModelingToolkit: get_eqs

const MOL = MethodOfLines

@testset "FFT operator matches the dense matrix" begin
    for N in (16, 15), d in 1:4
        spec = FourierCollocation(N)
        grid = MOL.spectral_grid(spec, -3.0, 5.0)
        D = MOL.spectral_diff_matrix(spec, grid, d)
        op = MOL.fast_fourier_operator(spec, grid, d, D)
        @test op !== nothing
        v = randn(N)
        @test MOL.apply_along(op, v, Val(1)) ≈ D * v atol = 1.0e-10
        X = randn(N, 5)
        @test MOL.apply_along(op, X, Val(1)) ≈ D * X atol = 1.0e-10
        Y = randn(5, N)
        @test MOL.apply_along(op, Y, Val(2)) ≈ Y * transpose(D) atol = 1.0e-10
        Z = randn(3, N, 4)
        ref = MOL.apply_along(D, Z, Val(2))
        @test MOL.apply_along(op, Z, Val(2)) ≈ ref atol = 1.0e-10
        for k in 1:4
            @test ref[:, :, k] ≈ Z[:, :, k] * transpose(D)
        end
        # element types without a plan (dual numbers) go through the matrix
        @test MOL.apply_along(op, big.(v), Val(1)) ≈ D * v atol = 1.0e-10
    end
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
