# WENO discretizer integration: extent dispatch and interior map.

using Test
using ModelingToolkit, MethodOfLines, DomainSets

const M = MethodOfLines

@parameters t x
@variables u(..)
Dt = Differential(t)
Dx = Differential(x)

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
    @test CartesianIndex(2) in interior_nu   # Val(2)
    @test CartesianIndex(20) in interior_nu  # Val(4)

    interior_u = vec(collect(im_u.I[pde_u]))
    @test !(CartesianIndex(2) in interior_u)
    @test !(CartesianIndex(20) in interior_u)
end
