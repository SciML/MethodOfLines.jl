using ModelingToolkit, MethodOfLines, LinearAlgebra, Test, OrdinaryDiffEq, DomainSets
using OrdinaryDiffEqSSPRK: SSPRK33
using SciMLBase
using ModelingToolkit: Differential

@parameters t x x1 x2 x3 x4
@variables u(..) u1(..) u2(..) u3(..) u4(..)

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
    x = collect(a .+ [0.0; cumsum(dx)])
    x[end] = b
    return x
end

function right_cluster_grid(a, b, n::Integer; ratio = 1000.0)
    m = n - 1
    r = ratio^(1 / (m - 1))
    dx = reverse(collect(r .^ (0:(m - 1))))
    dx .*= (b - a) / sum(dx)
    x = collect(a .+ [0.0; cumsum(dx)])
    x[end] = b
    return x
end

stretching_ratio(x::AbstractVector) = maximum(diff(x)) / minimum(diff(x))

function advection_timestep(x::AbstractVector, v::Real)
    return 0.25 * minimum(abs, diff(x)) / abs(v)
end

function translating_sine_exact(x, t, v, L)
    return sin.(2π .* (x .- v .* t) ./ L)
end

function solve_periodic_advection(;
        xgrid,
        v,
        tspan = (0.0, 0.4),
        u0,
        u_exact,
        saveat = nothing,
        advection_scheme = UpwindScheme(),
    )
    xgrid = collect(xgrid)
    L = xgrid[end] - xgrid[1]
    t0, tf = tspan
    saveat = isnothing(saveat) ? [tf] : saveat
    x0, xL = xgrid[1], xgrid[end]

    Dt = Differential(t)
    Dx = Differential(x)

    eq = Dt(u(t, x)) ~ -v * Dx(u(t, x))
    bcs = [
        u(t0, x) ~ u0(x),
        u(t, x0) ~ u(t, xL),
    ]
    domains = [t ∈ Interval(t0, tf), x ∈ Interval(x0, xL)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    disc = MOLFiniteDifference([x => xgrid], t; advection_scheme)
    prob = discretize(pdesys, disc)
    dt = advection_timestep(xgrid, v)
    sol = solve(prob, SSPRK33(); dt = dt, saveat = saveat, adaptive = false)
    return sol, disc, prob, L
end

function interface_aligned_grids(;
        n1 = 51,
        n2 = 51,
        L1 = 0.5,
        L2 = 0.5,
        x1_builder = right_cluster_grid,
        x2_builder = one_sided_cluster_grid,
        ratio = 500.0,
    )
    x1grid = x1_builder(0.0, L1, n1; ratio = ratio)
    x2grid = x2_builder(L1, L1 + L2, n2; ratio = ratio)
    @assert isapprox(x1grid[end], x2grid[1]; rtol = 0, atol = eps(eltype(x1grid)))
    return x1grid, x2grid
end

function solve_multi_domain_interface_advection(;
        x1grid,
        x2grid,
        v,
        tspan = (0.0, 0.3),
        u0,
        u_exact,
        saveat = nothing,
        v2 = nothing,
        advection_scheme = UpwindScheme(),
    )
    x1grid = collect(x1grid)
    x2grid = collect(x2grid)
    v2 = isnothing(v2) ? v : v2
    L = x1grid[end] - x1grid[1] + x2grid[end] - x2grid[1]
    t0, tf = tspan
    saveat = isnothing(saveat) ? [tf] : saveat

    Dt = Differential(t)
    Dx1 = Differential(x1)
    Dx2 = Differential(x2)

    eqs = [
        Dt(u1(t, x1)) ~ -v * Dx1(u1(t, x1)),
        Dt(u2(t, x2)) ~ -v2 * Dx2(u2(t, x2)),
    ]
    bcs = [
        u1(t0, x1) ~ u0(x1),
        u2(t0, x2) ~ u0(x2),
        u1(t, x1grid[end]) ~ u2(t, x2grid[1]),
    ]
    if v >= 0
        push!(bcs, u1(t, x1grid[1]) ~ u_exact(x1grid[1], t))
        push!(bcs, Dx2(u2(t, x2grid[end])) ~ 0.0)
    else
        push!(bcs, u2(t, x2grid[end]) ~ u_exact(x2grid[end], t))
        push!(bcs, Dx1(u1(t, x1grid[1])) ~ 0.0)
    end
    domains = [
        t ∈ Interval(t0, tf),
        x1 ∈ Interval(x1grid[1], x1grid[end]),
        x2 ∈ Interval(x2grid[1], x2grid[end]),
    ]
    @named pdesys = PDESystem(
        eqs, bcs, domains, [t, x1, x2], [u1(t, x1), u2(t, x2)],
    )

    disc = MOLFiniteDifference(
        [x1 => x1grid, x2 => x2grid], t; advection_scheme,
    )
    prob = discretize(pdesys, disc)
    dt = min(advection_timestep(x1grid, max(abs(v), abs(v2))), advection_timestep(x2grid, max(abs(v), abs(v2))))
    sol = solve(prob, SSPRK33(); dt = dt, saveat = saveat, adaptive = false)
    return sol, disc, prob, L
end

function solve_chained_interface_advection(;
        grids,
        v,
        tspan = (0.0, 0.1),
        u0,
        u_exact,
        saveat = nothing,
    )
    @assert v > 0 "chained inflow is set up for positive wind"
    grids = collect.(grids)
    xivs = (x1, x2, x3, x4)
    uops = (u1, u2, u3, u4)
    t0, tf = tspan
    saveat = isnothing(saveat) ? [tf] : saveat

    Dt = Differential(t)
    Dxs = (Differential(x1), Differential(x2), Differential(x3), Differential(x4))

    eqs = [Dt(uops[k](t, xivs[k])) ~ -v * Dxs[k](uops[k](t, xivs[k])) for k in 1:4]
    bcs = reduce(vcat, [[uops[k](t0, xivs[k]) ~ u0(xivs[k])] for k in 1:4])
    for k in 1:3
        push!(bcs, uops[k](t, grids[k][end]) ~ uops[k + 1](t, grids[k + 1][1]))
    end
    push!(bcs, uops[1](t, grids[1][1]) ~ u_exact(grids[1][1], t))
    push!(bcs, Dxs[4](uops[4](t, grids[4][end])) ~ 0.0)

    domains = vcat(
        t ∈ Interval(t0, tf),
        [xivs[k] ∈ Interval(grids[k][1], grids[k][end]) for k in 1:4],
    )
    @named pdesys = PDESystem(
        eqs, bcs, domains, [t, xivs...], [uops[k](t, xivs[k]) for k in 1:4],
    )

    disc = MOLFiniteDifference(
        [xivs[k] => grids[k] for k in 1:4], t; advection_scheme = UpwindScheme(),
    )
    prob = discretize(pdesys, disc)
    dt = minimum(advection_timestep(g, v) for g in grids)
    sol = solve(prob, SSPRK33(); dt = dt, saveat = saveat, adaptive = false)
    return sol, disc, prob
end

function build_mismatch_interface_system(;
        tspan = (0.0, 0.2),
        u0 = x -> sin(2π * x),
    )
    t0, = tspan
    Dt = Differential(t)
    Dx1 = Differential(x1)
    Dx2 = Differential(x2)

    eqs = [
        Dt(u1(t, x1)) ~ -Dx1(u1(t, x1)),
        Dt(u2(t, x2)) ~ -Dx2(u2(t, x2)),
    ]
    bcs = [
        u1(t0, x1) ~ u0(x1),
        u2(t0, x2) ~ u0(x2),
        u1(t, 0.5) ~ u2(t, 0.5),
        u1(t, 0.0) ~ 0.0,
        u2(t, 1.0) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(tspan...),
        x1 ∈ Interval(0.0, 0.5),
        x2 ∈ Interval(0.5, 1.0),
    ]
    @named pdesys = PDESystem(
        eqs, bcs, domains, [t, x1, x2], [u1(t, x1), u2(t, x2)],
    )
    return pdesys
end

@testset "Mismatched nonuniform interface coordinates reject discretization" begin
    L = 1.0
    u0 = x -> sin(2π * x / L)

    x1grid = one_sided_cluster_grid(0.0, 0.5, 51; ratio = 200.0)
    x2grid = one_sided_cluster_grid(0.55, 1.0, 51; ratio = 200.0)

    @test x1grid[end] ≈ 0.5
    @test x2grid[1] ≈ 0.55
    @test !isapprox(x1grid[end], x2grid[1])

    pdesys = build_mismatch_interface_system(; u0 = u0)
    disc = MOLFiniteDifference(
        [x1 => x1grid, x2 => x2grid], t; advection_scheme = UpwindScheme(),
    )

    @test_throws ArgumentError get_discrete(pdesys, disc)
    # The validation must fire on the real user entry point, not just get_discrete.
    @test_throws ArgumentError discretize(pdesys, disc)
end

@testset "Mismatched scalar step size rejects vector interface pairing" begin
    Dt = Differential(t)
    Dx1 = Differential(x1)
    Dx2 = Differential(x2)

    eqs = [
        Dt(u1(t, x1)) ~ -Dx1(u1(t, x1)),
        Dt(u2(t, x2)) ~ -Dx2(u2(t, x2)),
    ]
    bcs = [
        u1(0.0, x1) ~ 0.0,
        u2(0.0, x2) ~ 0.0,
        u1(t, 0.5) ~ u2(t, 0.5),
        u1(t, 0.0) ~ 0.0,
        u2(t, 1.0) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.2),
        x1 ∈ Interval(0.0, 0.5),
        x2 ∈ Interval(0.5, 1.0),
    ]
    @named pdesys = PDESystem(
        eqs, bcs, domains, [t, x1, x2], [u1(t, x1), u2(t, x2)],
    )

    x2grid = symmetric_cluster_grid(0.5, 1.0, 41; stretch = 5.0)
    disc = MOLFiniteDifference(
        [x1 => 0.01, x2 => x2grid], t; advection_scheme = UpwindScheme(),
    )

    @test_throws ArgumentError get_discrete(pdesys, disc)
    @test_throws ArgumentError discretize(pdesys, disc)
end

@testset "Cross-domain periodic ring topology rejects discretization" begin
    x1grid = one_sided_cluster_grid(0.0, 0.5, 31; ratio = 50.0)
    x2grid = one_sided_cluster_grid(0.5, 1.0, 31; ratio = 50.0)

    Dt = Differential(t)
    Dx1 = Differential(x1)
    Dx2 = Differential(x2)

    eqs = [
        Dt(u1(t, x1)) ~ -Dx1(u1(t, x1)),
        Dt(u2(t, x2)) ~ -Dx2(u2(t, x2)),
    ]
    bcs = [
        u1(0.0, x1) ~ sin(2π * x1),
        u2(0.0, x2) ~ sin(2π * x2),
        u1(t, 0.5) ~ u2(t, 0.5),
        u2(t, 1.0) ~ u1(t, 0.0),
    ]
    domains = [
        t ∈ Interval(0.0, 0.1),
        x1 ∈ Interval(0.0, 0.5),
        x2 ∈ Interval(0.5, 1.0),
    ]
    @named pdesys = PDESystem(
        eqs, bcs, domains, [t, x1, x2], [u1(t, x1), u2(t, x2)],
    )
    disc = MOLFiniteDifference(
        [x1 => x1grid, x2 => x2grid], t; advection_scheme = UpwindScheme(),
    )

    @test_throws ArgumentError discretize(pdesys, disc)
end

@testset "UpwindScheme order > 1 on nonuniform interface fails gracefully" begin
    pdesys = build_mismatch_interface_system(; u0 = x -> sin(2π * x))
    x1grid = one_sided_cluster_grid(0.0, 0.5, 21; ratio = 50.0)
    x2grid = one_sided_cluster_grid(0.5, 1.0, 21; ratio = 50.0)
    disc = MOLFiniteDifference(
        [x1 => x1grid, x2 => x2grid], t; advection_scheme = UpwindScheme(2),
    )

    @test_throws ArgumentError discretize(pdesys, disc)
end

@testset "Nonuniform interface grids route to AbstractVector topology" begin
    tspan = (0.0, 0.2)
    u0 = x -> sin(2π * x)
    x1grid = right_cluster_grid(0.0, 0.5, 61; ratio = 50.0)
    x2grid = one_sided_cluster_grid(0.5, 1.0, 51; ratio = 50.0)
    @test isapprox(x1grid[end], x2grid[1])

    Dt = Differential(t)
    Dx1 = Differential(x1)
    Dx2 = Differential(x2)

    eqs = [
        Dt(u1(t, x1)) ~ -Dx1(u1(t, x1)),
        Dt(u2(t, x2)) ~ -Dx2(u2(t, x2)),
    ]
    bcs = [
        u1(0.0, x1) ~ u0(x1),
        u2(0.0, x2) ~ u0(x2),
        u1(t, x1grid[end]) ~ u2(t, x2grid[1]),
        u1(t, x1grid[1]) ~ u0(x1grid[1]),
        Dx2(u2(t, x2grid[end])) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(tspan...),
        x1 ∈ Interval(x1grid[1], x1grid[end]),
        x2 ∈ Interval(x2grid[1], x2grid[end]),
    ]
    @named pdesys = PDESystem(
        eqs, bcs, domains, [t, x1, x2], [u1(t, x1), u2(t, x2)],
    )

    disc = MOLFiniteDifference(
        [x1 => x1grid, x2 => x2grid], t; advection_scheme = UpwindScheme(),
    )
    vmap = MethodOfLines.VariableMap(pdesys, disc)
    s = MethodOfLines.construct_discrete_space(vmap, disc)

    for (xi, g) in ((x1, x1grid), (x2, x2grid))
        @test s.grid[xi] === g
        @test s.grid[xi] isa AbstractVector
        @test !(s.grid[xi] isa StepRangeLen)
        @test s.dxs[xi] isa Vector
        @test length(s.dxs[xi]) == length(g) - 1
        @test s.dxs[xi] ≈ diff(g)
    end

    prob = discretize(pdesys, disc)
    @test prob isa SciMLBase.ODEProblem
end

@testset "Periodic advection on nonuniform grid" begin
    L = 1.0
    v = 1.0
    tspan = (0.0, 0.4)
    u0 = x -> sin(2π * x / L)
    u_exact = (x, t) -> translating_sine_exact(x, t, v, L)
    xgrid = symmetric_cluster_grid(0.0, L, 121; stretch = 5.0)

    sol, = solve_periodic_advection(;
        xgrid, v, tspan, u0, u_exact,
    )
    xs = sol[x]
    u_num = sol[u(t, x)][end, :]
    u_ref = u_exact(xs, tspan[2])

    @test all(isfinite, u_num)
    @test !any(isnan, u_num)
    @test rel_l2(u_num, u_ref, xs) < L2_RTOL
end

@testset "Periodic advection with negative wind on chebyshev grid" begin
    L = 1.0
    v = -2.0
    tspan = (0.0, 0.4)
    xgrid = chebyshev_nodes(0.0, L, 121)
    u0 = x -> sin(2π * x / L)
    u_exact = (x, t) -> translating_sine_exact(x, t, v, L)

    @test stretching_ratio(xgrid) >= 20.0

    sol, = solve_periodic_advection(;
        xgrid, v, tspan, u0, u_exact,
    )
    xs = sol[x]
    u_num = sol[u(t, x)][end, :]
    u_ref = u_exact(xs, tspan[2])

    @test all(isfinite, u_num)
    @test !any(isnan, u_num)
    @test rel_l2(u_num, u_ref, xs) < L2_RTOL
end

@testset "Periodic mass conservation on clustered grid" begin
    L = 1.0
    xgrid = symmetric_cluster_grid(0.0, L, 141; stretch = 6.0)
    σ = 0.04
    μ = 0.5
    v = 0.6
    tshort = 0.08
    u0 = x -> exp(-((x - μ)^2) / (2σ^2))
    gaussian_exact = (x, t) -> exp.(-((x .- μ .- v .* t) .^ 2) ./ (2σ^2))

    sol, = solve_periodic_advection(;
        xgrid, v, tspan = (0.0, tshort), u0,
        u_exact = gaussian_exact,
    )
    xs = sol[x]
    m0 = trapz_mass(u0.(xs), xs)
    mT = trapz_mass(sol[u(t, x)][end, :], xs)
    @test isapprox(mT, m0; rtol = MASS_RTOL)
end

@testset "Cross-domain upwind routing with opposed interface clustering" begin
    L = 1.0
    v = 1.0
    tspan = (0.0, 0.25)
    u0 = x -> sin(2π * x / L)
    u_exact = (x, t) -> translating_sine_exact(x, t, v, L)

    x1grid, x2grid = interface_aligned_grids(; ratio = 400.0)
    @test stretching_ratio(x1grid) >= 50.0
    @test stretching_ratio(x2grid) >= 50.0
    @test isapprox(x1grid[end], x2grid[1])

    sol, = solve_multi_domain_interface_advection(;
        x1grid, x2grid, v, tspan, u0, u_exact,
    )

    sol_u1 = sol[u1(t, x1)][end, :]
    sol_u2 = sol[u2(t, x2)][end, :]
    xs1 = sol[x1]
    xs2 = sol[x2]

    @test all(isfinite, sol_u1)
    @test all(isfinite, sol_u2)
    @test !any(isnan, vcat(sol_u1, sol_u2))

    @test isapprox(sol_u1[end], sol_u2[1]; rtol = 0.05, atol = 0.05)
    @test rel_l2(sol_u1, u_exact(xs1, tspan[2]), xs1) < L2_RTOL
    @test rel_l2(sol_u2, u_exact(xs2, tspan[2]), xs2) < L2_RTOL
end

@testset "Cross-domain advection with negative wind" begin
    L = 1.0
    v = -1.2
    tspan = (0.0, 0.3)
    u0 = x -> sin(2π * x / L)
    u_exact = (x, t) -> translating_sine_exact(x, t, v, L)

    x1grid = symmetric_cluster_grid(0.0, 0.5, 71; stretch = 6.0)
    x2grid = symmetric_cluster_grid(0.5, 1.0, 71; stretch = 6.0)
    @test isapprox(x1grid[end], x2grid[1])

    sol, = solve_multi_domain_interface_advection(;
        x1grid, x2grid, v, tspan, u0, u_exact,
    )

    sol_u1 = sol[u1(t, x1)][end, :]
    sol_u2 = sol[u2(t, x2)][end, :]

    @test isapprox(sol_u1[end], sol_u2[1]; rtol = 0.05, atol = 0.05)
    @test rel_l2(sol_u1, u_exact(sol[x1], tspan[2]), sol[x1]) < L2_RTOL
    @test rel_l2(sol_u2, u_exact(sol[x2], tspan[2]), sol[x2]) < L2_RTOL
end

@testset "Interface-adjacent stencil resilience under extreme stretching" begin
    L = 1.0
    v = 1.0
    tspan = (0.0, 0.15)
    u0 = x -> sin(2π * x / L)

    configs = [
        ("interface-fine", 1000.0, right_cluster_grid, one_sided_cluster_grid),
        ("interior-fine", 1000.0, one_sided_cluster_grid, right_cluster_grid),
        ("chebyshev-x1", 1.0, (a, b, n; ratio) -> chebyshev_nodes(a, b, n), one_sided_cluster_grid),
    ]

    for (label, ratio, x1b, x2b) in configs
        @testset "$label" begin
            x1grid, x2grid = interface_aligned_grids(;
                ratio = ratio, x1_builder = x1b, x2_builder = x2b,
            )
            u_exact = (x, t) -> translating_sine_exact(x, t, v, L)
            sol, = solve_multi_domain_interface_advection(;
                x1grid, x2grid, v, tspan, u0, u_exact,
            )
            sol_u1 = sol[u1(t, x1)][end, :]
            sol_u2 = sol[u2(t, x2)][end, :]
            @test all(isfinite, sol_u1)
            @test all(isfinite, sol_u2)
            @test maximum(abs, vcat(sol_u1, sol_u2)) < 5
        end
    end
end

@testset "Multi-domain decoupling with distinct advection speeds" begin
    L = 1.0
    v1 = 1.0
    v2 = 0.4
    tspan = (0.0, 0.2)
    u0 = x -> sin(2π * x / L)

    x1grid, x2grid = interface_aligned_grids(; n1 = 45, n2 = 45, ratio = 300.0)

    t0, tf = tspan
    sol, disc, = solve_multi_domain_interface_advection(;
        x1grid, x2grid, v = v1, v2 = v2, tspan, u0,
        u_exact = (x, t) -> translating_sine_exact(x, t, v1, L),
        saveat = [t0, tf],
    )

    @test length(sol[x1]) == length(x1grid)
    @test length(sol[x2]) == length(x2grid)
    @test disc.dxs[x1] isa Vector
    @test disc.dxs[x2] isa Vector

    sol_u1 = sol[u1(t, x1)]
    sol_u2 = sol[u2(t, x2)]

    @test size(sol_u1, 1) == 2
    @test size(sol_u2, 1) == 2
    @test size(sol_u1, 2) == length(x1grid)
    @test size(sol_u2, 2) == length(x2grid)
    @test sol_u1[1, 2:(end - 1)] ≈ u0.(x1grid[2:(end - 1)]) atol = 1e-10
    @test sol_u2[1, 1:(end - 1)] ≈ u0.(x2grid[1:(end - 1)]) atol = 1e-10
    @test isapprox(sol_u1[end, end], sol_u2[end, 1]; rtol = 0.1, atol = 0.1)

    interior_u1 = sol_u1[end, 2:(end - 2)]
    interior_u2 = sol_u2[end, 3:end]
    @test all(isfinite, interior_u1)
    @test all(isfinite, interior_u2)
    @test !any(isnan, vcat(interior_u1, interior_u2))
    @test length(sol_u1[end, :]) + length(sol_u2[end, :]) == length(x1grid) + length(x2grid)
end

@testset "Four chained nonuniform interfaces" begin
    L = 1.0
    v = 1.0
    tspan = (0.0, 0.1)
    u0 = x -> sin(2π * x / L)
    u_exact = (x, t) -> translating_sine_exact(x, t, v, L)

    g1 = right_cluster_grid(0.0, 0.25, 27; ratio = 60.0)
    g2 = one_sided_cluster_grid(0.25, 0.5, 31; ratio = 40.0)
    g3 = right_cluster_grid(0.5, 0.75, 25; ratio = 80.0)
    g4 = one_sided_cluster_grid(0.75, 1.0, 29; ratio = 30.0)
    grids = (g1, g2, g3, g4)

    for k in 1:3
        @test isapprox(grids[k][end], grids[k + 1][1])
    end

    sol, disc = solve_chained_interface_advection(;
        grids, v, tspan, u0, u_exact,
    )

    sols = [sol[u1(t, x1)][end, :], sol[u2(t, x2)][end, :],
        sol[u3(t, x3)][end, :], sol[u4(t, x4)][end, :]]

    @test all(g -> disc.dxs[g] isa Vector, (x1, x2, x3, x4))
    for (k, sk) in enumerate(sols)
        @test length(sk) == length(grids[k])
        @test all(isfinite, sk)
        @test !any(isnan, sk)
    end

    for k in 1:3
        @test isapprox(sols[k][end], sols[k + 1][1]; rtol = 0.05, atol = 0.05)
    end

    for k in 1:3
        xs = sol[(x1, x2, x3, x4)[k]]
        @test rel_l2(sols[k], u_exact(xs, tspan[2]), xs) < L2_RTOL
    end
end

@testset "Float32 interface grids resolve without promotion" begin
    Lf = 1.0f0
    vf = 1.0f0
    t0, tf = 0.0f0, 0.1f0

    x1grid = Float32.(right_cluster_grid(0.0, 0.5, 41; ratio = 100.0))
    x2grid = Float32.(one_sided_cluster_grid(0.5, 1.0, 41; ratio = 100.0))
    @test eltype(x1grid) == Float32
    @test eltype(x2grid) == Float32
    @test isapprox(x1grid[end], x2grid[1])

    twoπf = 2.0f0 * Float32(π)
    u0 = x -> sin(twoπf * x / Lf)

    Dt = Differential(t)
    Dx1 = Differential(x1)
    Dx2 = Differential(x2)

    eqs = [
        Dt(u1(t, x1)) ~ -vf * Dx1(u1(t, x1)),
        Dt(u2(t, x2)) ~ -vf * Dx2(u2(t, x2)),
    ]
    bcs = [
        u1(t0, x1) ~ u0(x1),
        u2(t0, x2) ~ u0(x2),
        u1(t, x1grid[end]) ~ u2(t, x2grid[1]),
        u1(t, x1grid[1]) ~ sin(twoπf * (-vf * t) / Lf),
        Dx2(u2(t, x2grid[end])) ~ 0.0f0,
    ]
    domains = [
        t ∈ Interval(t0, tf),
        x1 ∈ Interval(x1grid[1], x1grid[end]),
        x2 ∈ Interval(x2grid[1], x2grid[end]),
    ]
    @named pdesys = PDESystem(
        eqs, bcs, domains, [t, x1, x2], [u1(t, x1), u2(t, x2)],
    )

    disc = MOLFiniteDifference(
        [x1 => x1grid, x2 => x2grid], t; advection_scheme = UpwindScheme(),
    )

    # The nonuniform interface routing owns the grid/step-size topology; it must
    # carry the Float32 grids through without widening them. (The ODE state
    # vector itself is promoted to Float64 by the MTK/MOL codegen for any
    # nonuniform problem, interface or not, so that is deliberately out of scope
    # here and not asserted.)
    @test eltype(disc.dxs[x1]) == Float32
    @test eltype(disc.dxs[x2]) == Float32
    @test eltype(x1grid) == Float32

    prob = discretize(pdesys, disc)
    dt = Float32(min(advection_timestep(x1grid, vf), advection_timestep(x2grid, vf)))
    sol = solve(prob, SSPRK33(); dt = dt, saveat = [tf], adaptive = false)

    sol_u1 = sol[u1(t, x1)][end, :]
    sol_u2 = sol[u2(t, x2)][end, :]

    @test eltype(sol[x1]) == Float32
    @test eltype(sol[x2]) == Float32
    @test all(isfinite, sol_u1)
    @test all(isfinite, sol_u2)
    @test !any(isnan, vcat(sol_u1, sol_u2))
    @test isapprox(sol_u1[end], sol_u2[1]; rtol = 0.05, atol = 0.05)
end

@testset "Neumann outflow coexists with nonuniform upwind interface" begin
    L = 1.0
    v = 1.0
    tspan = (0.0, 0.25)
    u0 = x -> sin(2π * x / L)
    u_exact = (x, t) -> translating_sine_exact(x, t, v, L)

    x1grid, x2grid = interface_aligned_grids(; ratio = 200.0)
    @test isapprox(x1grid[end], x2grid[1])

    sol, = solve_multi_domain_interface_advection(;
        x1grid, x2grid, v, tspan, u0, u_exact,
    )

    xs2 = sol[x2]
    sol_u1 = sol[u1(t, x1)][end, :]
    sol_u2 = sol[u2(t, x2)][end, :]

    @test all(isfinite, vcat(sol_u1, sol_u2))
    @test !any(isnan, vcat(sol_u1, sol_u2))

    @test isapprox(sol_u1[end], sol_u2[1]; rtol = 0.05, atol = 0.05)
    @test rel_l2(sol_u1, u_exact(sol[x1], tspan[2]), sol[x1]) < L2_RTOL

    outflow_slope = (sol_u2[end] - sol_u2[end - 1]) / (xs2[end] - xs2[end - 1])
    @test abs(outflow_slope) < 5.0
end
