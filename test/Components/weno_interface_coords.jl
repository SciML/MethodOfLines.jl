# bcoord chart transitions for wrapped functional-scheme stencils.

using Test
using ModelingToolkit, MethodOfLines, DomainSets
using ModelingToolkit: operation

const M = MethodOfLines

function build_discrete_system(pdesys, disc)
    v = M.VariableMap(pdesys, disc)
    bcorders = Dict(
        map(xx -> xx => M.d_orders(xx, M.get_bcs(pdesys)), M.PDEBase.all_ivs(v))
    )
    bmap = M.PDEBase.parse_bcs(M.get_bcs(pdesys), v, bcorders)
    s = M.construct_discrete_space(v, disc)
    eqs = M.get_eqs(pdesys)
    eqs = eqs isa AbstractVector ? Vector{Equation}(eqs) : Equation[eqs]
    im = M.PDEBase.construct_var_equation_mapping(eqs, bmap, s, disc)
    return im, s, bmap
end

function perturbed_grid(a, b, n; amp = 0.004, seedmul = 1.0)
    g = collect(range(a, b, length = n))
    g[2:(end - 1)] .+= amp .* sin.(seedmul .* (1:(n - 2)))
    @assert all(diff(g) .> 0)
    return g
end

@testset "Periodic chart transition (bcoord)" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    g = perturbed_grid(0.0, 2.0, 21)
    N = length(g)
    L = g[end] - g[1]

    eq = Dt(u(t, x)) ~ -Dx(u(t, x))
    bcs = [u(0, x) ~ sinpi(x), u(t, 0.0) ~ u(t, 2.0)]
    domains = [t ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 2.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    disc = MOLFiniteDifference([x => g], t; advection_scheme = WENOScheme())
    im, s, bmap = build_discrete_system(pdesys, disc)

    pde = first(keys(im.I))
    u_field = im.var[pde]
    j = M.x2i(s, u_field, x)
    bs = M.PDEBase.filter_interfaces(bmap[only(keys(bmap))][x])
    jx = (j, x)

    haslower, hasupper = M.PDEBase.haslowerupper(bs, x)
    @test haslower && hasupper

    for i in 2:(N - 1)
        @test M.bcoord(CartesianIndex(i), bs, s, jx) == g[i]
    end

    # Lower wrap: period shift -L; u[1] ~ u[N].
    @test M.bcoord(CartesianIndex(1), bs, s, jx) ≈ g[N] - L
    @test M.bcoord(CartesianIndex(1), bs, s, jx) ≈ g[1]
    @test M.bcoord(CartesianIndex(0), bs, s, jx) ≈ g[N - 1] - L
    @test M.bcoord(CartesianIndex(-1), bs, s, jx) ≈ g[N - 2] - L

    # Upper wrap: period shift +L.
    @test M.bcoord(CartesianIndex(N + 1), bs, s, jx) ≈ g[2] + L

    # Monotonic coords at seam-adjacent points.
    for II in (CartesianIndex(2), CartesianIndex(3), CartesianIndex(N - 1), CartesianIndex(N))
        coords = [M.bcoord(II + CartesianIndex(i), bs, s, jx) for i in -2:2]
        @test all(diff(coords) .> 0)
    end

    F = WENOScheme()
    for II in (CartesianIndex(2), CartesianIndex(N))
        f, Itap, Iraw = M.get_f_taps_coords(F, II, s, bs, jx, u_field)
        @test f === F.interior
        discx = [M.bcoord(I, bs, s, jx) for I in Iraw]
        @test all(diff(discx) .> 0)
    end
end

@testset "Periodic stencil-wrap guard (N-1 >= interior_points)" begin
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    function guard_setup(n)
        g = perturbed_grid(0.0, 2.0, n; amp = 0.002)
        eq = Dt(u(t, x)) ~ -Dx(u(t, x))
        bcs = [u(0, x) ~ sinpi(x), u(t, 0.0) ~ u(t, 2.0)]
        domains = [t ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 2.0)]
        @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
        disc = MOLFiniteDifference([x => g], t; advection_scheme = WENOScheme())
        im, s, bmap = build_discrete_system(pdesys, disc)
        u_field = im.var[first(keys(im.I))]
        j = M.x2i(s, u_field, x)
        bs = M.PDEBase.filter_interfaces(bmap[only(keys(bmap))][x])
        return s, bs, (j, x), u_field
    end

    F = WENOScheme()

    # N = 5: only 4 distinct physical nodes (u[1] ~ u[N]); wrap must be rejected.
    s5, bs5, jx5, u5 = guard_setup(5)
    @test_throws "requires at least" M.get_f_taps_coords(F, CartesianIndex(3), s5, bs5, jx5, u5)

    # N = 6: minimal admissible periodic grid; interior path must engage cleanly.
    s6, bs6, jx6, u6 = guard_setup(6)
    f6, _, Iraw6 = M.get_f_taps_coords(F, CartesianIndex(3), s6, bs6, jx6, u6)
    @test f6 === F.interior
    discx6 = [M.bcoord(I, bs6, s6, jx6) for I in Iraw6]
    @test all(diff(discx6) .> 0)
end

@testset "Contiguous two-domain chart transition (bcoord)" begin
    @parameters t x1 x2
    @variables u1(..) u2(..)
    Dt = Differential(t)
    Dx1 = Differential(x1)
    Dx2 = Differential(x2)

    g1 = perturbed_grid(0.0, 0.5, 11; amp = 0.002)
    g2 = perturbed_grid(0.5, 1.0, 16; amp = 0.002, seedmul = 2.0)
    N1 = length(g1)

    eqs = [
        Dt(u1(t, x1)) ~ -Dx1(u1(t, x1)),
        Dt(u2(t, x2)) ~ -Dx2(u2(t, x2)),
    ]
    bcs = [
        u1(0, x1) ~ sinpi(2x1),
        u2(0, x2) ~ sinpi(2x2),
        u1(t, 0.0) ~ sinpi(-2t),
        u1(t, 0.5) ~ u2(t, 0.5),
        Dx2(u2(t, 1.0)) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, 0.2),
        x1 ∈ Interval(0.0, 0.5),
        x2 ∈ Interval(0.5, 1.0),
    ]
    @named pdesys = PDESystem(
        eqs, bcs, domains, [t, x1, x2], [u1(t, x1), u2(t, x2)]
    )

    disc = MOLFiniteDifference(
        [x1 => g1, x2 => g2], t; advection_scheme = WENOScheme()
    )
    im, s, bmap = build_discrete_system(pdesys, disc)

    # u1 upper wrap: coords from g2, zero shift.
    bs1 = M.PDEBase.filter_interfaces(bmap[operation(M.unwrap(u1(t, x1)))][x1])
    @test !isempty(bs1)
    j1 = M.x2i(s, M.depvar(M.unwrap(u1(t, x1)), s), x1)
    @test M.bcoord(CartesianIndex(N1), bs1, s, (j1, x1)) == g1[N1]
    @test M.bcoord(CartesianIndex(N1 + 1), bs1, s, (j1, x1)) ≈ g2[2]
    @test M.bcoord(CartesianIndex(N1 + 2), bs1, s, (j1, x1)) ≈ g2[3]
    coords1 = [M.bcoord(CartesianIndex(N1 + i), bs1, s, (j1, x1)) for i in -2:2]
    @test all(diff(coords1) .> 0)

    # u2 lower wrap: coords from g1, zero shift.
    bs2 = M.PDEBase.filter_interfaces(bmap[operation(M.unwrap(u2(t, x2)))][x2])
    @test !isempty(bs2)
    j2 = M.x2i(s, M.depvar(M.unwrap(u2(t, x2)), s), x2)
    @test M.bcoord(CartesianIndex(2), bs2, s, (j2, x2)) == g2[2]
    @test M.bcoord(CartesianIndex(1), bs2, s, (j2, x2)) ≈ g1[N1]
    @test M.bcoord(CartesianIndex(0), bs2, s, (j2, x2)) ≈ g1[N1 - 1]
    @test M.bcoord(CartesianIndex(-1), bs2, s, (j2, x2)) ≈ g1[N1 - 2]
    coords2 = [M.bcoord(CartesianIndex(2 + i), bs2, s, (j2, x2)) for i in -2:2]
    @test all(diff(coords2) .> 0)
end
