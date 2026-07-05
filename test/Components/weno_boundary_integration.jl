# WENO discretizer integration: extent dispatch and interior map.

using Test
using ModelingToolkit, MethodOfLines, DomainSets, OrdinaryDiffEq, SciMLBase

const M = MethodOfLines

@parameters t x
@variables u(..)
Dt = Differential(t)
Dx = Differential(x)

@testset "WENONonUniformBoundary scalar dx callable" begin
    xs = collect(range(0.0, 1.0, length = 5))
    u = sin.(2π .* xs)
    result = @inferred M.WENONonUniformBoundary{1}()(u, (1.0e-6,), 0.0, xs, 0.1)
    @test result isa Float64
    @test isfinite(result)
end

@testset "Grid-dispatched extent" begin
    s = WENOScheme()
    @test M.extent(s, 1, 1.0) == 2             # uniform dx
    @test M.extent(s, 1, [0.1, 0.2, 0.3]) == 0  # vector dx

    # WENO specialization; not AbstractScheme fallback.
    m_nonuniform = which(M.extent, (typeof(s), Int, Vector{Float64}))
    m_fallback = which(M.extent, (M.AbstractScheme, Int, Any))
    @test m_nonuniform !== m_fallback

    # Grid-agnostic schemes ignore dx.
    @test M.extent(UpwindScheme(), 1, [0.1, 0.2]) == M.extent(UpwindScheme(), 1)
end

@testset "Interior map routes boundary nodes to Val WENO (non-uniform only)" begin
    eq = Dt(u(t, x)) ~ -Dx(u(t, x))
    bcs = [u(0, x) ~ sinpi(x), u(t, 0.0) ~ 0.0, u(t, 1.0) ~ 0.0]
    domains = [t ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    function build_interiormap(disc)
        v = M.VariableMap(pdesys, disc)
        bcorders = Dict(
            map(xx -> xx => M.d_orders(xx, M.get_bcs(pdesys)), M.PDEBase.all_ivs(v))
        )
        bmap = M.PDEBase.parse_bcs(M.get_bcs(pdesys), v, bcorders)
        s = M.construct_discrete_space(v, disc)
        eqs = M.get_eqs(pdesys)
        eqs = eqs isa AbstractVector ? Vector{Equation}(eqs) : Equation[eqs]
        return M.PDEBase.construct_var_equation_mapping(eqs, bmap, s, disc)
    end

    nonuniform_grid = collect(range(0.0, 1.0, length = 21))
    nonuniform_grid[2:(end - 1)] .+= 0.004 .* sin.(1:19)

    im_nu = build_interiormap(MOLFiniteDifference([x => nonuniform_grid], t; advection_scheme = WENOScheme()))
    im_u = build_interiormap(MOLFiniteDifference([x => 1 / 20], t; advection_scheme = WENOScheme()))

    pde_nu = first(keys(im_nu.I))
    pde_u = first(keys(im_u.I))

    @test im_nu.stencil_extents[pde_nu] == ([0], [0])
    @test im_u.stencil_extents[pde_u] == ([2], [2])

    interior_nu = vec(collect(im_nu.I[pde_nu]))
    @test !(CartesianIndex(1) in interior_nu)  # wall: Dirichlet BC
    @test CartesianIndex(2) in interior_nu      # penultimate lower
    @test CartesianIndex(20) in interior_nu     # penultimate upper
    @test !(CartesianIndex(21) in interior_nu) # wall: Dirichlet BC

    interior_u = vec(collect(im_u.I[pde_u]))
    @test !(CartesianIndex(1) in interior_u)
    @test !(CartesianIndex(2) in interior_u)
    @test !(CartesianIndex(20) in interior_u)
    @test !(CartesianIndex(21) in interior_u)
end

@testset "Non-uniform WENO discretize + solve smoke (MMS)" begin
    T_END = 0.2
    u_exact(x, t) = sin(2π * (x - t))

    nonuniform_grid = collect(range(0.0, 1.0, length = 21))
    nonuniform_grid[2:(end - 1)] .+= 0.004 .* sin.(1:19)
    x0, xL = nonuniform_grid[1], nonuniform_grid[end]

    eq = Dt(u(t, x)) ~ -Dx(u(t, x))
    bcs = [
        u(0.0, x) ~ sin(2π * x),
        u(t, x0) ~ u_exact(x0, t),
        u(t, xL) ~ u_exact(xL, t),
    ]
    domains = [t ∈ Interval(0.0, T_END), x ∈ Interval(x0, xL)]
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    disc = MOLFiniteDifference([x => nonuniform_grid], t; advection_scheme = WENOScheme())
    prob = discretize(pdesys, disc)
    sol = solve(prob, Tsit5(); abstol = 1e-8, reltol = 1e-8, saveat = [T_END])

    @test SciMLBase.successful_retcode(sol)

    xs = sol[x]
    u_num = sol[u(t, x)][end, :]
    @test all(isfinite, u_num)

    u_ref = u_exact.(xs, T_END)
    w = similar(xs, Float64)
    w[1] = (xs[2] - xs[1]) / 2
    w[end] = (xs[end] - xs[end - 1]) / 2
    for i in 2:(length(xs) - 1)
        w[i] = (xs[i + 1] - xs[i - 1]) / 2
    end
    err = sqrt(sum(w .* abs2.(u_num .- u_ref)))
    ref = max(sqrt(sum(w .* abs2.(u_ref))), eps())
    rel_l2 = err / ref
    # measured rel_l2 ≈ 3.2e-3 at N=21, t=0.2
    @test rel_l2 < 0.01
end
