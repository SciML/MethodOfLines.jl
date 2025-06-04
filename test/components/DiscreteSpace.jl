using ModelingToolkit, MethodOfLines, DomainSets, Test, Symbolics, SymbolicUtils

@testset "Test 01: discretization of variables, center aligned grid" begin
    # Test centered order

    @parameters x, y, t
    @variables u(..) v(..)

    t_min = 0.0
    t_max = 2.0
    x_min = 0.0
    x_max = 2.0
    y_min = 0.0
    y_max = 2.0
    dx = 0.1
    dy = 0.2
    order = 2

    domains = [
        t ∈ Interval(t_min, t_max), x ∈ Interval(x_min, x_max), y ∈ Interval(y_min, y_max)]

    pde = u(t, x, y) ~ v(t, x, y) + Differential(x)(u(t, x, y)) +
                       Differential(y)(v(t, x, y))

    bcs = [u(t_min, x, y) ~ 0, u(t_max, x, y) ~ 0, v(t_min, x, y) ~ 0, v(t_max, x, y) ~ 0]

    @named pdesys = PDESystem(pde, bcs, domains, [t, x, y], [u(t, x, y), v(t, x, y)])

    disc = MOLFiniteDifference([x => dx, y => dy], t; approx_order = order)
    depvar_ops = map(x -> operation(x.val), getfield(pdesys, :depvars))

    v = MethodOfLines.VariableMap(pdesys, disc)

    s = MethodOfLines.construct_discrete_space(v, disc)

    discx = x_min:dx:x_max
    discy = y_min:dy:y_max

    @test s.grid[x] == discx
    @test s.grid[y] == discy

    @test s.axies[x] == s.grid[x]
    @test s.axies[y] == s.grid[y]

    #@test all([all(I[i] .∈ (collect(s.Igrid),)) for I in values(s.Iedges), i in [1,2]])

end

@testset "Test 02: discretization of variables, edge aligned grid" begin
    # Test centered order
    @parameters x, y, t
    @variables u(..) v(..)

    t_min = 0.0
    t_max = 2.0
    x_min = 0.0
    x_max = 2.0
    y_min = 0.0
    y_max = 2.0
    dx = 0.1
    dy = 0.2
    order = 2

    domains = [
        t ∈ Interval(t_min, t_max), x ∈ Interval(x_min, x_max), y ∈ Interval(y_min, y_max)]

    pde = u(t, x, y) ~ v(t, x, y) + Differential(x)(u(t, x, y)) +
                       Differential(y)(v(t, x, y))
    bcs = [u(t_min, x, y) ~ 0, u(t_max, x, y) ~ 0, v(t_min, x, y) ~ 0, v(t_max, x, y) ~ 0]

    @named pdesys = PDESystem(pde, bcs, domains, [t, x, y], [u(t, x, y), v(t, x, y)])

    disc = MOLFiniteDifference(
        [x => dx, y => dy], t; approx_order = order, grid_align = edge_align)
    v = MethodOfLines.VariableMap(pdesys, disc)

    s = MethodOfLines.construct_discrete_space(v, disc)
    discx = (x_min - dx / 2):dx:(x_max + dx / 2)
    discy = (y_min - dy / 2):dy:(y_max + dy / 2)

    grid = Dict([x => discx, y => discy])

    @test s.grid[x] == discx
    @test s.grid[y] == discy

    @test s.axies[x] != s.grid[x]
    @test s.axies[y] != s.grid[y]

    #@test all([all(I[i] .∈ (collect(s.Igrid),)) for I in values(s.Iedges), i in [1,2]])

end
