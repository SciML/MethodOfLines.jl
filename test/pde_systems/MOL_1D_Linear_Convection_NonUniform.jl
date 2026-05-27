using ModelingToolkit, MethodOfLines, LinearAlgebra, Test, OrdinaryDiffEq, DomainSets
using ModelingToolkit: Differential

@parameters t x
@variables u(..)

const L2_RTOL = 0.2
const MASS_RTOL = 5e-2

cell_widths(x::AbstractVector) = [diff(x)...; diff(x)[end]]

function l2_norm(u::AbstractVector, x::AbstractVector)
    w = cell_widths(x)
    return sqrt(sum(w .* abs2.(u)))
end

function rel_l2(u::AbstractVector, uref::AbstractVector, x::AbstractVector)
    err = l2_norm(u .- uref, x)
    ref = max(l2_norm(uref, x), eps(eltype(u)))
    return err / ref
end

function trapz_mass(u::AbstractVector, x::AbstractVector)
    dx = diff(x)
    return sum((u[1:(end - 1)] .+ u[2:end]) .* dx ./ 2)
end

function chebyshev_nodes(a, b, n::Integer)
    k = 1:n
    nodes = sort((a + b) / 2 .+ (b - a) / 2 .* cos.(π * (2k .- 1) ./ (2n)))
    nodes[1] = a
    nodes[end] = b
    return collect(nodes)
end

function symmetric_cluster_grid(a, b, n::Integer; stretch = 6.5)
    ξ = range(-1, 1, length = n)
    map = (sinh.(stretch .* ξ) ./ sinh(stretch) .+ 1) ./ 2
    x = a .+ (b - a) .* map
    x[1] = a
    x[end] = b
    return collect(x)
end

function one_sided_cluster_grid(a, b, n::Integer; ratio = 1000.0)
    m = n - 1
    r = ratio^(1 / (m - 1))
    dx = collect(r .^ (0:(m - 1)))
    dx .*= (b - a) / sum(dx)
    return collect(a .+ [0.0; cumsum(dx)])
end

function right_cluster_grid(a, b, n::Integer; ratio = 1000.0)
    m = n - 1
    r = ratio^(1 / (m - 1))
    dx = reverse(collect(r .^ (0:(m - 1))))
    dx .*= (b - a) / sum(dx)
    return collect(a .+ [0.0; cumsum(dx)])
end

stretching_ratio(x::AbstractVector) = maximum(diff(x)) / minimum(diff(x))

function advection_timestep(x::AbstractVector, v::Real)
    return 0.25 * minimum(abs, diff(x)) / abs(v)
end

function translating_sine_exact(x, t, v, L)
    return sin.(2π .* (x .- v .* t) ./ L)
end

function solve_mms_advection(;
        xgrid,
        v,
        tspan = (0.0, 0.4),
        u0,
        u_exact,
        saveat = nothing,
        advection_scheme = UpwindScheme(),
        approx_order = 2,
    )
    xgrid = collect(xgrid)
    L = xgrid[end] - xgrid[1]
    x0, xL = xgrid[1], xgrid[end]
    t0, tf = tspan
    saveat = isnothing(saveat) ? [tf] : saveat

    Dt = Differential(t)
    Dx = Differential(x)

    eq = Dt(u(t, x)) ~ -v * Dx(u(t, x))
    bcs = [
        u(t0, x) ~ u0(x),
        u(t, x0) ~ u_exact(x0, t),
        u(t, xL) ~ u_exact(xL, t),
    ]
    domains = [t ∈ Interval(t0, tf), x ∈ Interval(x0, xL)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    disc = MOLFiniteDifference(
        [x => xgrid], t; advection_scheme, approx_order,
    )
    prob = discretize(pdesys, disc)
    dt = advection_timestep(xgrid, v)
    sol = solve(prob, SSPRK33(); dt = dt, saveat = saveat, adaptive = false)
    return sol, disc, prob, L
end

function inflow_exact(x, t, v, L, uL)
    return v >= 0 ? uL(t - x / v) : uL(t - (L - x) / abs(v))
end

function solve_inflow_advection(;
        xgrid,
        v,
        tspan = (0.0, 0.35),
        uL,
        saveat = nothing,
        advection_scheme = UpwindScheme(),
        approx_order = 2,
    )
    xgrid = collect(xgrid)
    L = xgrid[end] - xgrid[1]
    t0, tf = tspan
    saveat = isnothing(saveat) ? [tf] : saveat

    Dt = Differential(t)
    Dx = Differential(x)

    eq = Dt(u(t, x)) ~ -v * Dx(u(t, x))
    inflow = v >= 0 ? xgrid[1] : xgrid[end]
    outflow = v >= 0 ? xgrid[end] : xgrid[1]
    bcs = [
        u(t0, x) ~ inflow_exact(x, t0, v, L, uL),
        u(t, inflow) ~ uL(t),
        Dx(u(t, outflow)) ~ 0.0,
    ]
    domains = [t ∈ Interval(t0, tf), x ∈ Interval(xgrid[1], xgrid[end])]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    disc = MOLFiniteDifference(
        [x => xgrid], t; advection_scheme, approx_order,
    )
    prob = discretize(pdesys, disc)
    dt = advection_timestep(xgrid, v)
    sol = solve(prob, SSPRK33(); dt = dt, saveat = saveat, adaptive = false)
    return sol, disc, prob, L
end

@testset "Extreme stretching resilience" begin
    L = 1.0
    v = 1.0
    tspan = (0.0, 0.25)
    u0 = x -> sin(2π * x / L)
    u_exact = (x, t) -> translating_sine_exact(x, t, v, L)

    grids = [
        ("symmetric", symmetric_cluster_grid(0.0, L, 101; stretch = 7.0), 50.0),
        ("left", one_sided_cluster_grid(0.0, L, 101; ratio = 1000.0), 100.0),
        ("right", right_cluster_grid(0.0, L, 101; ratio = 1000.0), 100.0),
        ("chebyshev", chebyshev_nodes(0.0, L, 101), 25.0),
    ]

    for (label, xgrid, min_ratio) in grids
        @testset "$label grid" begin
            @test stretching_ratio(xgrid) >= min_ratio
            sol, = solve_mms_advection(;
                xgrid, v, tspan, u0, u_exact,
            )
            xs = sol[x]
            u_num = sol[u(t, x)][end, :]
            @test all(isfinite, u_num)
            @test maximum(abs, u_num) < 5
            @test !any(isnan, u_num)
        end
    end
end

@testset "Directional switching awareness" begin
    L = 1.0
    tspan = (0.0, 0.3)
    xgrid = symmetric_cluster_grid(0.0, L, 111; stretch = 5.5)
    u0 = x -> sin(2π * x / L)

    for v in (1.0, -1.0)
        @testset "v = $v" begin
            u_exact = (x, t) -> translating_sine_exact(x, t, v, L)
            sol, _, _, = solve_mms_advection(;
                xgrid, v, tspan, u0, u_exact,
            )
            xs = sol[x]
            tf = tspan[2]
            u_num = sol[u(t, x)][end, :]
            u_ref = u_exact(xs, tf)
            @test rel_l2(u_num, u_ref, xs) < L2_RTOL
        end
    end

    @testset "spatially varying velocity" begin
        xgrid = symmetric_cluster_grid(0.0, L, 95; stretch = 5.0)
        t0, tf = tspan
        Dt = Differential(t)
        Dx = Differential(x)

        vel(x) = 0.6 * sin(2π * x / L)
        eq = Dt(u(t, x)) ~ -vel(x) * Dx(u(t, x))
        bcs = [
            u(t0, x) ~ sin(2π * x / L),
            u(t, xgrid[1]) ~ sin(2π * (-vel(xgrid[1]) * t) / L),
            u(t, xgrid[end]) ~ sin(2π * (xgrid[end] - vel(xgrid[end]) * t) / L),
        ]
        domains = [t ∈ Interval(t0, tf), x ∈ Interval(xgrid[1], xgrid[end])]
        @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

        disc = MOLFiniteDifference([x => xgrid], t; advection_scheme = UpwindScheme())
        prob = discretize(pdesys, disc)
        dt = 0.2 * minimum(diff(xgrid)) / 0.6
        sol = solve(prob, SSPRK33(); dt = dt, saveat = [tf], adaptive = false)
        u_num = sol[u(t, x)][end, :]
        @test all(isfinite, u_num)
        @test maximum(abs, u_num) < 5
    end

    @testset "inflow boundaries" begin
        xgrid = one_sided_cluster_grid(0.0, L, 97; ratio = 500.0)
        uL(t) = sin(2π * t / L)
        for v in (0.8, -0.8)
            sol, _, _, = solve_inflow_advection(;
                xgrid, v, tspan = (0.0, 0.2), uL,
            )
            xs = sol[x]
            tf = 0.2
            u_num = sol[u(t, x)][end, :]
            u_ref = inflow_exact.(xs, tf, v, L, uL)
            @test rel_l2(u_num, u_ref, xs) < 0.35
        end
    end
end

@testset "Accuracy and leakage" begin
    L = 1.0
    v = 1.0
    tf = 0.4
    u0 = x -> sin(2π * x / L)
    u_exact = (x, t) -> translating_sine_exact(x, t, v, L)

    x_uniform = collect(range(0.0, L; length = 121))
    x_nonuniform = symmetric_cluster_grid(0.0, L, 121; stretch = 5.0)

    sol_u, = solve_mms_advection(;
        xgrid = x_uniform, v, tspan = (0.0, tf), u0, u_exact,
    )
    sol_n, = solve_mms_advection(;
        xgrid = x_nonuniform, v, tspan = (0.0, tf), u0, u_exact,
    )

    xs_u = sol_u[x]
    xs_n = sol_n[x]
    err_u = rel_l2(
        sol_u[u(t, x)][end, :],
        u_exact(xs_u, tf),
        xs_u,
    )
    err_n = rel_l2(
        sol_n[u(t, x)][end, :],
        u_exact(xs_n, tf),
        xs_n,
    )

    @test err_u < L2_RTOL
    @test err_n < L2_RTOL
    @test err_n < 5err_u

    @testset "Gaussian mass conservation" begin
        xgrid = symmetric_cluster_grid(0.0, L, 141; stretch = 6.0)
        σ = 0.04
        μ = 0.5
        v = 0.6
        tshort = 0.08
        u0 = x -> exp(-((x - μ)^2) / (2σ^2))
        gaussian_exact = (x, t) -> exp.(-((x .- μ .- v .* t) .^ 2) ./ (2σ^2))
        sol, _, _, = solve_mms_advection(;
            xgrid, v, tspan = (0.0, tshort), u0, u_exact = gaussian_exact,
        )
        xs = sol[x]
        m0 = trapz_mass(u0.(xs), xs)
        mT = trapz_mass(sol[u(t, x)][end, :], xs)
        @test isapprox(mT, m0; rtol = MASS_RTOL)
    end
end

@testset "Interface integrity" begin
    L = 1.0
    tspan = (0.0, 0.2)
    u0 = x -> sin(2π * x / L)

    @testset "AbstractVector routing" begin
        xgrid = symmetric_cluster_grid(0.0, L, 61; stretch = 4.5)
        Dt = Differential(t)
        Dx = Differential(x)

        eq = Dt(u(t, x)) ~ -Dx(u(t, x))
        bcs = [
            u(0.0, x) ~ u0(x),
            u(t, xgrid[1]) ~ sin(2π * (-t) / L),
            u(t, xgrid[end]) ~ sin(2π * (xgrid[end] - t) / L),
        ]
        domains = [t ∈ Interval(tspan...), x ∈ Interval(xgrid[1], xgrid[end])]
        @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

        disc = MOLFiniteDifference(
            [x => xgrid], t; advection_scheme = UpwindScheme(),
        )
        vmap = MethodOfLines.VariableMap(pdesys, disc)
        s = MethodOfLines.construct_discrete_space(vmap, disc)

        @test s.grid[x] === xgrid
        @test s.grid[x] isa AbstractVector
        @test !(s.grid[x] isa StepRangeLen)
        @test s.dxs[x] isa Vector
        @test length(s.dxs[x]) == length(xgrid) - 1
        @test s.dxs[x] ≈ diff(xgrid)

        prob = discretize(pdesys, disc)
        @test prob isa SciMLBase.ODEProblem
    end

    @testset "chebyspace constructor" begin
        @parameters xch
        xgrid = last(chebyspace(51, xch ∈ Interval(0.0, L)))
        sol, = solve_mms_advection(;
            xgrid, v = 1.0, tspan, u0,
            u_exact = (x, t) -> translating_sine_exact(x, t, 1.0, L),
        )
        @test all(isfinite, sol[u(t, x)][end, :])
    end
    @testset "type promotion" begin
        xgrid32 = Float32.(symmetric_cluster_grid(0.0, L, 51; stretch = 4.0))
        sol, disc, = solve_mms_advection(;
            xgrid = xgrid32, v = 1.0f0, tspan, u0,
            u_exact = (x, t) -> translating_sine_exact(x, t, 1.0f0, Float32(L)),
        )
        @test eltype(sol[x]) <: AbstractFloat
        @test all(isfinite, sol[u(t, x)][end, :])
        @test disc.dxs[x] == xgrid32
    end

    @testset "uniform fallback guard" begin
        dx = L / 80
        xgrid = symmetric_cluster_grid(0.0, L, 81; stretch = 4.0)
        disc_vec = MOLFiniteDifference([x => xgrid], t; advection_scheme = UpwindScheme())
        disc_step = MOLFiniteDifference([x => dx], t; advection_scheme = UpwindScheme())

        Dt = Differential(t)
        Dx = Differential(x)
        eq = Dt(u(t, x)) ~ -Dx(u(t, x))
        bcs = [
            u(0.0, x) ~ u0(x),
            u(t, 0.0) ~ sin(-2π * t / L),
            u(t, L) ~ sin(2π * (L - t) / L),
        ]
        domains = [t ∈ Interval(tspan...), x ∈ Interval(0.0, L)]
        @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

        vmap = MethodOfLines.VariableMap(pdesys, disc_vec)
        s_vec = MethodOfLines.construct_discrete_space(vmap, disc_vec)
        vmap = MethodOfLines.VariableMap(pdesys, disc_step)
        s_step = MethodOfLines.construct_discrete_space(vmap, disc_step)

        @test s_vec.grid[x] isa AbstractVector
        @test !(s_vec.grid[x] isa StepRangeLen)
        @test s_step.grid[x] isa StepRangeLen
        @test s_vec.dxs[x] isa Vector
        @test s_step.dxs[x] isa Number
    end
end
