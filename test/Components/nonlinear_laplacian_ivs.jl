using ModelingToolkit, MethodOfLines, DomainSets, Test

@testset "Transverse IV substitutions follow argument order, not s.x̄" begin
    @parameters t x y
    @variables u(..)

    t_min = 0.0
    t_max = 2.0
    x_min = 0.0
    x_max = 2.0
    y_min = 0.0
    y_max = 2.0
    # Equal spacing: the historical `s.x̄[k] => grid[s.x̄[k]][II[k]]` mix-up does
    # not BoundsError when nx == ny; it silently substitutes the wrong coordinate.
    dx = dy = 0.2

    domains = [
        t ∈ Interval(t_min, t_max), x ∈ Interval(x_min, x_max), y ∈ Interval(y_min, y_max),
    ]
    pde = Differential(t)(u(t, x, y)) ~
        Differential(x)(x * Differential(x)(u(t, x, y))) +
        Differential(y)(y * Differential(y)(u(t, x, y)))
    bcs = [
        u(t_min, x, y) ~ 0,
        u(t, x_min, y) ~ 0,
        u(t, x_max, y) ~ 0,
        u(t, x, y_min) ~ 0,
        u(t, x, y_max) ~ 0,
    ]
    @named pdesys = PDESystem([pde], bcs, domains, [t, x, y], [u(t, x, y)])
    disc = MOLFiniteDifference([x => dx, y => dy], t)
    s = MethodOfLines.construct_discrete_space(MethodOfLines.VariableMap(pdesys, disc), disc)
    uvar = s.ū[1]

    function assert_argument_ordered(s, uvar, x, y)
        II = CartesianIndex(8, 3)
        y_rules = MethodOfLines.transverse_iv_substitutions(s, uvar, II, y)
        x_rules = MethodOfLines.transverse_iv_substitutions(s, uvar, II, x)
        @test length(y_rules) == 1
        @test isequal(first(y_rules[1]), x)
        @test last(y_rules[1]) == s.grid[x][8]
        @test length(x_rules) == 1
        @test isequal(first(x_rules[1]), y)
        @test last(x_rules[1]) == s.grid[y][3]
    end

    assert_argument_ordered(s, uvar, x, y)

    # Force the historically-broken VariableMap order so this test does not
    # depend on SymbolicUtils Set iteration.
    xbar = s.x̄
    if xbar isa AbstractVector
        reverse!(xbar)
    end
    assert_argument_ordered(s, uvar, x, y)

    # Unequal grids: the CI failure indexed y's 11-point grid at 12.
    disc2 = MOLFiniteDifference([x => 0.1, y => 0.2], t)
    s2 = MethodOfLines.construct_discrete_space(MethodOfLines.VariableMap(pdesys, disc2), disc2)
    u2 = s2.ū[1]
    II = CartesianIndex(12, 3)
    y_rules = MethodOfLines.transverse_iv_substitutions(s2, u2, II, y)
    @test isequal(first(y_rules[1]), x)
    @test last(y_rules[1]) == s2.grid[x][12]
    @test last(y_rules[1]) != s2.grid[y][3]
end
