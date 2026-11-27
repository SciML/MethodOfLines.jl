using ModelingToolkit, MethodOfLines, LinearAlgebra, Test, OrdinaryDiffEq, DomainSets
using OrdinaryDiffEqSSPRK: SSPRK33
using SciMLBase
using ModelingToolkit: Differential
include(joinpath(@__DIR__, "..", "shared", "ode_discretize.jl"))

@parameters t x x1 x2 x3 x4
@variables u(..) u1(..) u2(..) u3(..) u4(..)

const L2_RTOL = 0.2
const MASS_RTOL = 5.0e-2

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
    prob = ode_discretize(pdesys, disc)
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
    prob = ode_discretize(pdesys, disc)
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
        [xivs[k] => grids[k] for k in 1:4], t;
        advection_scheme = UpwindScheme(),
    )
    prob = ode_discretize(pdesys, disc)
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
        [x1 => x1grid, x2 => x2grid], t;
        advection_scheme = UpwindScheme(),
    )

    @test_throws ArgumentError get_discrete(pdesys, disc)
    # The validation must fire on the real user entry point, not just get_discrete.
    @test_throws ArgumentError ode_discretize(pdesys, disc)
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
    @test_throws ArgumentError ode_discretize(pdesys, disc)
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
        [x1 => x1grid, x2 => x2grid], t;
        advection_scheme = UpwindScheme(),
    )

    @test_throws ArgumentError ode_discretize(pdesys, disc)
end

@testset "UpwindScheme order > 1 on nonuniform interface fails gracefully" begin
    pdesys = build_mismatch_interface_system(; u0 = x -> sin(2π * x))
    x1grid = one_sided_cluster_grid(0.0, 0.5, 21; ratio = 50.0)
    x2grid = one_sided_cluster_grid(0.5, 1.0, 21; ratio = 50.0)
    disc = MOLFiniteDifference(
        [x1 => x1grid, x2 => x2grid], t; advection_scheme = UpwindScheme(2),
    )

    @test_throws ArgumentError ode_discretize(pdesys, disc)
end

@testset "UpwindScheme order > 1 on nonuniform periodic wrap fails gracefully" begin
    # Same-variable periodic wraps have b.x == b.x2, so they bypass the
    # mismatched-dx gate; the order guard must still fire for them.
    xgrid = symmetric_cluster_grid(0.0, 1.0, 41; stretch = 5.0)
    Dt = Differential(t)
    Dx = Differential(x)
    eq = Dt(u(t, x)) ~ -Dx(u(t, x))
    bcs = [
        u(0.0, x) ~ sin(2π * x),
        u(t, 0.0) ~ u(t, 1.0),
    ]
    domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
    disc = MOLFiniteDifference(
        [x => xgrid], t; advection_scheme = UpwindScheme(2),
    )

    @test_throws ArgumentError ode_discretize(pdesys, disc)
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
        [x1 => x1grid, x2 => x2grid], t;
        advection_scheme = UpwindScheme(),
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

    prob = ode_discretize(pdesys, disc)
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
    @test sol_u1[1, 2:(end - 1)] ≈ u0.(x1grid[2:(end - 1)]) atol = 1.0e-10
    @test sol_u2[1, 1:(end - 1)] ≈ u0.(x2grid[1:(end - 1)]) atol = 1.0e-10
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

    sols = [
        sol[u1(t, x1)][end, :], sol[u2(t, x2)][end, :],
        sol[u3(t, x3)][end, :], sol[u4(t, x4)][end, :],
    ]

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
        [x1 => x1grid, x2 => x2grid], t;
        advection_scheme = UpwindScheme(),
    )

    # The nonuniform interface routing owns the grid/step-size topology; it must
    # carry the Float32 grids through without widening them. (The ODE state
    # vector itself is promoted to Float64 by the MTK/MOL codegen for any
    # nonuniform problem, interface or not, so that is deliberately out of scope
    # here and not asserted.)
    @test eltype(disc.dxs[x1]) == Float32
    @test eltype(disc.dxs[x2]) == Float32
    @test eltype(x1grid) == Float32

    prob = ode_discretize(pdesys, disc)
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

# Two connected domains [lo, mid] and [mid, hi] with the same equation on both. `kind` is
# :advection (u_t = -u_x, exact solution f(x - t)) or :heat (u_t = u_xx, steady for affine f).
function build_seam_system(kind, lo, mid, hi; f = x -> x - lo)
    rhs(u, X, D) = kind == :advection ? -D(u(t, X)) : D(D(u(t, X)))
    Dt = Differential(t)
    eqs = [
        Dt(u1(t, x1)) ~ rhs(u1, x1, Differential(x1)),
        Dt(u2(t, x2)) ~ rhs(u2, x2, Differential(x2)),
    ]
    inflow = kind == :advection ? f(lo - t) : f(lo)
    outflow = kind == :advection ? f(hi - t) : f(hi)
    bcs = [
        u1(0.0, x1) ~ f(x1),
        u2(0.0, x2) ~ f(x2),
        u1(t, mid) ~ u2(t, mid),
        u1(t, lo) ~ inflow,
        u2(t, hi) ~ outflow,
    ]
    domains = [t ∈ Interval(0.0, 0.2), x1 ∈ Interval(lo, mid), x2 ∈ Interval(mid, hi)]
    @named pdesys = PDESystem(
        eqs, bcs, domains, [t, x1, x2], [u1(t, x1), u2(t, x2)],
    )
    return pdesys
end

function seam_rhs(pdesys, disc)
    prob = ode_discretize(pdesys, disc)
    du = similar(prob.u0)
    prob.f(du, prob.u0, prob.p, 0.0)
    return du, prob.u0
end

# The error a public entry point raises, or nothing.
function thrown(f)
    try
        f()
    catch err
        return err
    end
    return nothing
end

seam_message(f) = sprint(showerror, thrown(f))

# Both public entry points reject, and the message names the interface and derivative.
function test_seam_rejected(pdesys, disc, what)
    for entry in (get_discrete, ode_discretize)
        err = thrown(() -> entry(pdesys, disc))
        @test err isa ArgumentError
        err isa ArgumentError && @test occursin(what, err.msg) && occursin("grid gaps", err.msg)
    end
    return
end

@testset "First-order upwind across unequal seam gaps matches the backward difference" begin
    # u = x^2 at t = 0, so u_t = -(x_i + x_{i-1}) with the true node coordinates, seam state included.
    for n in (9, 17)
        pdesys = build_seam_system(:advection, 0.0, 0.5, 1.5; f = x -> x^2)
        disc = MOLFiniteDifference([x1 => n, x2 => n], t; advection_scheme = UpwindScheme())
        grids = get_discrete(pdesys, disc)
        nodes = sort(unique(vcat(grids[x1], grids[x2])))
        @test (length(grids[x1]), length(grids[x2])) == (n, n)
        @test grids[x1][end] - grids[x1][end - 1] == 0.5 / (n - 1)
        @test grids[x2][2] - grids[x2][1] == 1.0 / (n - 1)

        du, u0 = seam_rhs(pdesys, disc)
        @test length(du) == 2n - 3
        for (k, uk) in enumerate(u0)
            i = findfirst(==(sqrt(uk)), nodes)
            @test i !== nothing
            @test du[k] ≈ -(nodes[i] + nodes[i - 1]) atol = 1.0e-12
        end
        @test any(==(0.5), sqrt.(u0))
    end
end

@testset "Affine first-order advection is exact across unequal seam gaps" begin
    pdesys = build_seam_system(:advection, 0.0, 0.5, 1.5)
    disc = MOLFiniteDifference([x1 => 9, x2 => 9], t; advection_scheme = UpwindScheme())
    du, = seam_rhs(pdesys, disc)
    @test all(≈(-1.0; atol = 1.0e-12), du)
end

@testset "Equal-count and scalar-step first-order advection stay accepted" begin
    for (specs, mid, hi) in (((12, 12), 0.1, 0.2), ((0.06, 0.06), 0.5, 1.0))
        pdesys = build_seam_system(:advection, 0.0, mid, hi)
        disc = MOLFiniteDifference(
            [x1 => specs[1], x2 => specs[2]], t; advection_scheme = UpwindScheme(),
        )
        @test get_discrete(pdesys, disc) isa Dict
        du, = seam_rhs(pdesys, disc)
        @test maximum(abs.(du .+ 1.0)) < 1.0e-12
    end
end

@testset "Heat across unequal seam gaps is rejected from both entry points" begin
    pdesys = build_seam_system(:heat, 0.0, 0.5, 1.5)
    disc = MOLFiniteDifference([x1 => 9, x2 => 9], t)
    test_seam_rejected(pdesys, disc, "derivative order 2")

    # Gaps 1.0 and 1.125 exactly, at an offset where the coordinate spacing is 0.0625.
    a = 2.0^48
    pdesys = build_seam_system(:heat, a, a + 4, a + 8.5; f = x -> x - a)
    disc = MOLFiniteDifference([x1 => 5, x2 => 5], t)
    test_seam_rejected(pdesys, disc, "derivative order 2")
    err = thrown(() -> get_discrete(pdesys, disc))
    @test occursin("[1.0] in x1", err.msg)
    @test occursin("[1.125] in x2", err.msg)

    # Equal gaps at the same offset: accepted, and the steady affine profile has a zero RHS.
    pdesys = build_seam_system(:heat, a, a + 4, a + 8; f = x -> x - a)
    du, = seam_rhs(pdesys, disc)
    @test all(iszero, du)
end

@testset "Schemes that are not coordinate-aware reject unequal seam gaps" begin
    pdesys = build_seam_system(:advection, 0.0, 0.5, 1.5)
    for (label, scheme) in (("UpwindScheme(2)", UpwindScheme(2)), ("WENO advection", WENOScheme()))
        disc = MOLFiniteDifference([x1 => 9, x2 => 9], t; advection_scheme = scheme)
        test_seam_rejected(pdesys, disc, label)
    end

    # The same schemes are fine when the gaps match.
    pdesys = build_seam_system(:advection, 0.0, 0.5, 1.0)
    for scheme in (UpwindScheme(2), WENOScheme())
        disc = MOLFiniteDifference([x1 => 11, x2 => 11], t; advection_scheme = scheme)
        du, = seam_rhs(pdesys, disc)
        @test maximum(abs.(du .+ 1.0)) < 1.0e-8
    end
end

@testset "Nested and mixed derivatives reject unequal seam gaps" begin
    Dt = Differential(t)
    Dx1, Dx2 = Differential(x1), Differential(x2)
    eqs = [
        Dt(u1(t, x1)) ~ Dx1((1 + u1(t, x1)) * Dx1(u1(t, x1))),
        Dt(u2(t, x2)) ~ Dx2((1 + u2(t, x2)) * Dx2(u2(t, x2))),
    ]
    bcs = [
        u1(0.0, x1) ~ x1, u2(0.0, x2) ~ x2, u1(t, 0.5) ~ u2(t, 0.5),
        u1(t, 0.0) ~ 0.0, u2(t, 1.5) ~ 1.5,
    ]
    domains = [t ∈ Interval(0.0, 0.2), x1 ∈ Interval(0.0, 0.5), x2 ∈ Interval(0.5, 1.5)]
    @named nonlinear = PDESystem(eqs, bcs, domains, [t, x1, x2], [u1(t, x1), u2(t, x2)])
    disc = MOLFiniteDifference([x1 => 9, x2 => 9], t)
    test_seam_rejected(nonlinear, disc, "derivative order 2")

    @parameters y
    @variables w1(..) w2(..)
    Dy = Differential(y)
    eqs = [
        Dt(w1(t, x1, y)) ~ Dx1(Dy(w1(t, x1, y))),
        Dt(w2(t, x2, y)) ~ Dx2(Dy(w2(t, x2, y))),
    ]
    bcs = [
        w1(0.0, x1, y) ~ x1 * y, w2(0.0, x2, y) ~ x2 * y, w1(t, 0.5, y) ~ w2(t, 0.5, y),
        w1(t, 0.0, y) ~ 0.0, w2(t, 1.5, y) ~ 1.5 * y,
        w1(t, x1, 0.0) ~ 0.0, w1(t, x1, 1.0) ~ x1,
        w2(t, x2, 0.0) ~ 0.0, w2(t, x2, 1.0) ~ x2,
    ]
    domains = [
        t ∈ Interval(0.0, 0.2), x1 ∈ Interval(0.0, 0.5), x2 ∈ Interval(0.5, 1.5),
        y ∈ Interval(0.0, 1.0),
    ]
    @named mixed = PDESystem(
        eqs, bcs, domains, [t, x1, x2, y], [w1(t, x1, y), w2(t, x2, y)],
    )
    disc = MOLFiniteDifference([x1 => 9, x2 => 9, y => 9], t)
    test_seam_rejected(mixed, disc, "a mixed derivative")
    # The same unequal intervals as explicit node vectors.
    disc = MOLFiniteDifference(
        [
            x1 => collect(range(0.0, 0.5; length = 9)), x2 => collect(range(0.5, 1.5; length = 9)),
            y => 9,
        ], t,
    )
    test_seam_rejected(mixed, disc, "a mixed derivative")
    disc = MOLFiniteDifference([x1 => 11, x2 => 11, y => 9], t)
    domains[3] = x2 ∈ Interval(0.5, 1.0)
    bcs[5] = w2(t, 1.0, y) ~ y
    @named mixed_equal = PDESystem(
        eqs, bcs, domains, [t, x1, x2, y], [w1(t, x1, y), w2(t, x2, y)],
    )
    @test get_discrete(mixed_equal, disc) isa Dict
end

@testset "Float32 equal-width interface grids stay accepted" begin
    for kind in (:advection, :heat)
        pdesys = build_seam_system(kind, 0.1f0, 0.2f0, 0.3f0; f = x -> x)
        disc = MOLFiniteDifference([x1 => 9, x2 => 9], t)
        @test get_discrete(pdesys, disc) isa Dict
        @test symbolic_discretize(pdesys, disc) isa Tuple
    end
end

@testset "A scalar step with a short final cell rejects heat at the seam" begin
    # Steps of 0.06 on [0, 0.5] end in a 0.02 cell next to the interface.
    pdesys = build_seam_system(:heat, 0.0, 0.5, 1.0)
    disc = MOLFiniteDifference([x1 => 0.06, x2 => 0.06], t)
    test_seam_rejected(pdesys, disc, "derivative order 2")
end

@testset "Explicit node vectors with different gaps reject heat from both entry points" begin
    pdesys = build_seam_system(:heat, 0.0, 0.5, 1.5)
    disc = MOLFiniteDifference(
        [x1 => collect(range(0.0, 0.5; length = 9)), x2 => collect(range(0.5, 1.5; length = 9))], t,
    )
    for entry in (get_discrete, ode_discretize)
        @test_throws ArgumentError entry(pdesys, disc)
    end
end

@testset "Gap differences of one rounding unit pass and larger ones are rejected" begin
    # Steps 2^-11 and 2^-11 + 2^-23 in Float32 differ by 2^-12 of a gap, far above the rounding of the nodes.
    mid = 0.25f0
    hi = mid + 512 * (2.0f0^-11 + 2.0f0^-23)
    pdesys = build_seam_system(:heat, 0.0f0, mid, hi; f = x -> x)
    disc = MOLFiniteDifference([x1 => 513, x2 => 513], t)
    test_seam_rejected(pdesys, disc, "derivative order 2")

    # Exactly representable gaps 1 and 2 at a coordinate scale where a gap is a unit in the last place.
    a = 2.0^52
    pdesys = build_seam_system(:heat, a, a + 8, a + 24; f = x -> x - a)
    disc = MOLFiniteDifference([x1 => 9, x2 => 9], t)
    test_seam_rejected(pdesys, disc, "derivative order 2")

    # Every sampled node gap rounds to 1.0, but the right-hand stencil uses the step 1.125.
    pdesys = build_seam_system(:heat, a, a + 8, a + 17; f = x -> x - a)
    test_seam_rejected(pdesys, disc, "derivative order 2")

    # The same scalar step on both sides is accepted although rounded node gaps at this offset differ.
    a40 = 2.0^40
    pdesys = build_seam_system(:heat, a40, a40 + 9, a40 + 18; f = x -> x - a40)
    disc = MOLFiniteDifference([x1 => 0.9, x2 => 0.9], t)
    @test get_discrete(pdesys, disc) isa Dict

    # Equal nominal widths typed as decimals pick up rounding differences between the two gaps.
    for (lo, mid, hi) in ((0.1, 0.2, 0.3), (0.7, 1.4, 2.1), (3.3, 4.4, 5.5))
        pdesys = build_seam_system(:heat, lo, mid, hi; f = x -> x - lo)
        disc = MOLFiniteDifference([x1 => 101, x2 => 101], t)
        @test get_discrete(pdesys, disc) isa Dict
    end
end

@testset "Only the connected fields are checked for derivatives across the seam" begin
    Dt = Differential(t)
    Dx1, Dx2 = Differential(x1), Differential(x2)
    @variables w(..)
    eqs = [
        Dt(u1(t, x1)) ~ -Dx1(u1(t, x1)),
        Dt(u2(t, x2)) ~ -Dx2(u2(t, x2)),
        Dt(w(t, x2)) ~ Dx2(Dx2(w(t, x2))),
    ]
    bcs = [
        u1(0.0, x1) ~ x1, u2(0.0, x2) ~ x2, w(0.0, x2) ~ x2,
        u1(t, 0.5) ~ u2(t, 0.5), u1(t, 0.0) ~ -t, u2(t, 1.5) ~ 1.5 - t,
        w(t, 0.5) ~ 0.5, w(t, 1.5) ~ 1.5,
    ]
    domains = [t ∈ Interval(0.0, 0.2), x1 ∈ Interval(0.0, 0.5), x2 ∈ Interval(0.5, 1.5)]
    @named pdesys = PDESystem(
        eqs, bcs, domains, [t, x1, x2], [u1(t, x1), u2(t, x2), w(t, x2)],
    )
    disc = MOLFiniteDifference([x1 => 9, x2 => 9], t)
    @test get_discrete(pdesys, disc) isa Dict
    @test symbolic_discretize(pdesys, disc) isa Tuple

    # A pointwise source from a connected field does not put a stencil on the interface.
    eqs[3] = Dt(w(t, x2)) ~ Dx2(Dx2(w(t, x2))) + u2(t, x2)
    @named sourced = PDESystem(
        eqs, bcs, domains, [t, x1, x2], [u1(t, x1), u2(t, x2), w(t, x2)],
    )
    @test get_discrete(sourced, disc) isa Dict
    @test symbolic_discretize(sourced, disc) isa Tuple

    # The same fields on explicit node vectors with different gaps.
    vectors = MOLFiniteDifference(
        [x1 => collect(range(0.0, 0.5; length = 9)), x2 => collect(range(0.5, 1.5; length = 9))], t,
    )
    @test get_discrete(pdesys, vectors) isa Dict
end

@testset "Coordinate-aware first-order schemes on explicit vectors are not rejected" begin
    # Steps of 0.06 on [0, 0.5] end in a short cell, so the axes are vectors with unequal seam gaps.
    pdesys = build_seam_system(:advection, 0.0, 0.5, 1.0)
    disc = MOLFiniteDifference([x1 => 0.06, x2 => 0.06], t; advection_scheme = WENOScheme())
    @test get_discrete(pdesys, disc) isa Dict

    # One vector and one uniform axis: the uniform side runs scalar-step WENO across the short gap.
    pdesys = build_seam_system(:advection, 0.0, 0.53125, 1.03125)
    disc = MOLFiniteDifference([x1 => 0.0625, x2 => 0.0625], t; advection_scheme = WENOScheme())
    test_seam_rejected(pdesys, disc, "WENO advection")
end

# A user scheme that reads only the upwind taps of its three-point stencil.
upwind_taps(u, p, t, x, dx) = (u[2] - u[1]) / dx

@testset "User functional schemes are not classified and behave as without the check" begin
    scheme = FunctionalScheme{3, 1}(upwind_taps, [nothing], [nothing], false, []; name = "upwind_taps")
    a = 2.0^48
    for (lo, mid, hi, n) in ((0.0, 0.5, 1.5, 9), (0.0, 0.5, 1.5, 17), (a, a + 4, a + 8.5, 5))
        pdesys = build_seam_system(:advection, lo, mid, hi; f = x -> (x - lo)^2)
        disc = MOLFiniteDifference([x1 => n, x2 => n], t; advection_scheme = scheme)
        grids = get_discrete(pdesys, disc)
        nodes = sort(unique(vcat(grids[x1], grids[x2])))
        du, u0 = seam_rhs(pdesys, disc)
        for (k, uk) in enumerate(u0)
            i = argmin(abs.((nodes .- lo) .^ 2 .- uk))
            @test du[k] ≈ -((nodes[i] - lo) + (nodes[i - 1] - lo)) atol = 1.0e-9
        end
    end
end

@testset "Integer-count axes that the colon range builds cleanly are unchanged" begin
    for (lo, hi, n) in ((0.0, 0.5, 11), (0.0, 1.0, 11), (0.1, 0.2, 11), (0.0, 0.5, 9), (0.5, 1.5, 9))
        dx = (hi - lo) / (n - 1)
        colon = lo:dx:hi
        @test length(colon) == n && colon[end] == hi
        pdesys = build_seam_system(:heat, lo, hi, 2hi - lo)
        disc = MOLFiniteDifference([x1 => n, x2 => n], t)
        @test get_discrete(pdesys, disc)[x1] == collect(colon)
        @test get_discrete(pdesys, disc)[x1] isa StepRangeLen
    end
end

@testset "A short final cell from the integer count rejects heat at the seam" begin
    # 12 points on [0, 0.09] gain a 1.4e-17 final cell (#752); first-order advection on them still works.
    pdesys = build_seam_system(:heat, 0.0, 0.09, 0.18)
    disc = MOLFiniteDifference([x1 => 12, x2 => 12], t)
    test_seam_rejected(pdesys, disc, "derivative order 2")
    pdesys = build_seam_system(:advection, 0.0, 0.09, 0.18)
    du, = seam_rhs(pdesys, disc)
    @test maximum(abs.(du .+ 1.0)) < 1.0e-9
end
