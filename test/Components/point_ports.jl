using ModelingToolkit, MethodOfLines, DomainSets, Test
using MethodOfLines: select_port_nodes, port_spec
include(joinpath(@__DIR__, "..", "shared", "heat_ports.jl"))

@parameters x y
@variables T(..)

function port_space(port, dxs; dims = [x])
    args = [t; dims]
    eq = Dt(T(args...)) ~ sum(d -> (Differential(d)^2)(T(args...)), dims)
    bcs = [T(0, dims...) ~ 0.0]
    for d in dims
        push!(bcs, T(t, map(z -> isequal(z, d) ? 0.0 : z, dims)...) ~ 0.0)
        push!(bcs, T(t, map(z -> isequal(z, d) ? 1.0 : z, dims)...) ~ 0.0)
    end
    domains = [t ∈ Interval(0.0, 1.0); [d ∈ Interval(0.0, 1.0) for d in dims]]
    @named pdesys = PDESystem(eq, bcs, domains, args, [T(args...)]; systems = [port])
    disc = MOLFiniteDifference(dxs, t)
    v = MethodOfLines.VariableMap(pdesys, disc)
    return port_spec(port), MethodOfLines.construct_discrete_space(v, disc)
end

grid_of(s, z) = collect(s.grid[z])
interp(f, s, nodes, w; dims = [x]) =
    sum(w[k] * f(map(d -> grid_of(s, dims[d])[nodes[k][d]], eachindex(dims))...) for k in eachindex(w))

@testset "1-D uniform grid" begin
    @named p = PointPort(HeatPort; var = T(t, x), at = [x => 0.33], across = :T, select = Interpolate(2))
    spec, s = port_space(p, [x => 0.1])
    nodes, w, V = select_port_nodes(spec, s, 2)
    @test length(nodes) == 3
    @test sum(w) ≈ 1
    f(z) = 1 + 2z - 3z^2
    @test interp(f, s, nodes, w) ≈ f(0.33)
    @test all(≈(0.1), V)

    @named p = PointPort(HeatPort; var = T(t, x), at = [x => 0.33], across = :T, select = Nearest())
    nodes, w, _ = select_port_nodes(port_spec(p), s, 2)
    @test nodes == [CartesianIndex(4)] && w == [1.0]
    @named p = PointPort(HeatPort; var = T(t, x), at = [x => 0.35], across = :T, select = Nearest())
    @test first(select_port_nodes(port_spec(p), s, 2)) == [CartesianIndex(4)]

    @named p = PointPort(HeatPort; var = T(t, x), at = [x => 0.5], across = :T)
    nodes, w, _ = select_port_nodes(port_spec(p), s, 2)
    @test nodes == [CartesianIndex(6)] && w == [1.0]

    @named p = PointPort(HeatPort; var = T(t, x), at = [x => 0.0], across = :T, select = Nearest())
    _, _, V = select_port_nodes(port_spec(p), s, 2)
    @test V ≈ [0.05]

    @named p = PointPort(HeatPort; var = T(t, x), at = [x => 0.12], across = :T)
    @test first(select_port_nodes(port_spec(p), s, 2)) == CartesianIndex.(1:3)
    @test first(select_port_nodes(port_spec(p), s, 2; restrict_to = (2:10,))) == CartesianIndex.(2:4)

    @named p = PointPort(HeatPort; var = T(t, x), at = [x => 0.05], across = :T)
    @test_throws r"flux boundary condition" select_port_nodes(port_spec(p), s, 2; restrict_to = (2:10,))
    @named p = PointPort(HeatPort; var = T(t, x), at = [x => 1.2], across = :T)
    @test_throws r"outside the domain" select_port_nodes(port_spec(p), s, 2)
end

@testset "1-D nonuniform grid" begin
    xs = [0.0, 0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0]
    @named p = PointPort(HeatPort; var = T(t, x), at = [x => 0.42], across = :T, select = Interpolate(2))
    spec, s = port_space(p, [x => xs])
    nodes, w, V = select_port_nodes(spec, s, 2)
    @test sum(w) ≈ 1
    f(z) = 1 + 2z - 3z^2
    @test interp(f, s, nodes, w) ≈ f(0.42)
    @test V ≈ [(xs[i + 1] - xs[i - 1]) / 2 for i in Tuple.(nodes) .|> first]

    @named p = PointPort(HeatPort; var = T(t, x), at = [x => 0.42], across = :T, select = Interpolate(3))
    nodes, w, _ = select_port_nodes(port_spec(p), s, 2)
    g(z) = 1 - z + 2z^2 + 5z^3
    @test length(nodes) == 4
    @test interp(g, s, nodes, w) ≈ g(0.42)
end

@testset "2-D tensor product" begin
    @named p = PointPort(
        HeatPort; var = T(t, x, y), at = [y => 0.47, x => 0.33], across = :T,
        select = Interpolate(1)
    )
    spec, s = port_space(p, [x => 0.1, y => 0.1]; dims = [x, y])
    nodes, w, V = select_port_nodes(spec, s, 2)
    @test sort(nodes) == sort(vec([CartesianIndex(i, j) for i in 4:5, j in 5:6]))
    @test sum(w) ≈ 1
    f(a, b) = 1 + a + 2b + 3a * b
    @test interp(f, s, nodes, w; dims = [x, y]) ≈ f(0.33, 0.47)
    @test w[findfirst(==(CartesianIndex(4, 5)), nodes)] ≈ 0.7 * 0.3
    @test all(≈(0.01), V)

    @named p = PointPort(HeatPort; var = T(t, x, y), at = [x => 0.33], across = :T)
    @test_throws r"exactly the spatial arguments" select_port_nodes(port_spec(p), s, 2)
end

@testset "PointPort validation" begin
    @named p = PointPort(HeatPort; var = T(t, x), at = x => 0.3, across = :T, through = :Q_flow => 2)
    @test port_spec(p).through == (:Q_flow, 2)
    @test ModelingToolkit.isconnector(p)
    @named rod = PDESystem(
        [Dt(T(t, x)) ~ (Differential(x)^2)(T(t, x))], [T(0, x) ~ 0.0],
        [t ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0)], [t, x], [T(t, x)]; systems = [p]
    )
    @test port_spec(only(ModelingToolkit.get_systems(rod))) === port_spec(p)

    @test_throws r"not an unknown" PointPort(HeatPort; name = :p, var = T(t, x), at = [x => 0.3], across = :Temp)
    @test_throws r"Flow variable" PointPort(HeatPort; name = :p, var = T(t, x), at = [x => 0.3], across = :Q_flow)
    @test_throws r"connect = Flow" PointPort(
        HeatPort; name = :p, var = T(t, x), at = [x => 0.3], across = :T, through = :T => 1
    )
    @test_throws r"did not return a connector" PointPort(
        HeatCapacitor; name = :p, var = T(t, x), at = [x => 0.3], across = :T
    )
    @test_throws r"not an argument" PointPort(HeatPort; name = :p, var = T(t, x), at = [y => 0.3], across = :T)
    @test_throws r"more than once" PointPort(
        HeatPort; name = :p, var = T(t, x), at = [x => 0.3, x => 0.4], across = :T
    )
    @test_throws ArgumentError Interpolate(-2)
end
