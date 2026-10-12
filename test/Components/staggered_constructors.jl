using ModelingToolkit, MethodOfLines, DomainSets, Test, Symbolics, SymbolicUtils,
    LinearAlgebra

@testset "staggered constructor" begin
    @parameters t x
    @variables ρ(..) ϕ(..)

    Dt = Differential(t)
    Dx = Differential(x)

    a = 5.0 #1.0/2.0;
    L = 8.0
    dx = 1.0 #0.125;
    dt = dx / a
    tmax = 1000.0

    initialFunction(x) = exp(-(x)^2)
    eq = [
        Dt(ρ(t, x)) + Dx(ϕ(t, x)) ~ 0,
        Dt(ϕ(t, x)) + a^2 * Dx(ρ(t, x)) ~ 0,
    ]
    bcs = [
        ρ(0, x) ~ initialFunction(x),
        ϕ(0.0, x) ~ 0.0,
        ρ(t, -L) ~ initialFunction(-L) * exp(-(t^2)),
        ρ(t, L) ~ 0.0,
        ϕ(t, -L) ~ exp((t^2)),
        Dt(ϕ(t, L)) ~ -a^2 * Dx(ρ(t, L)),
    ]

    domains = [
        t in Interval(0.0, tmax),
        x in Interval(-L, L),
    ]

    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [ρ(t, x), ϕ(t, x)])

    discretization = MOLFiniteDifference(
        [x => dx], t, grid_align = MethodOfLines.StaggeredGrid(),
        edge_aligned_var = ϕ(t, x)
    )
    @test operation(Symbolics.unwrap(discretization.kwargs[:edge_aligned_var])) ===
        operation(Symbolics.unwrap(ϕ(t, x)))

    v = MethodOfLines.VariableMap(pdesys, discretization)
    disc_space = MethodOfLines.construct_discrete_space(v, discretization)
    xs = Symbolics.unwrap(x)
    @test MethodOfLines.staggered_alignment(disc_space, Symbolics.unwrap(ρ(t, x)), xs) ==
        MethodOfLines.CenterAlignedVar
    @test MethodOfLines.staggered_alignment(disc_space, Symbolics.unwrap(ϕ(t, x)), xs) ==
        MethodOfLines.EdgeAlignedVar

    orders = Dict(map(xx -> xx => [1; 2], v.x̄))
    diff_disc = MethodOfLines.construct_differential_discretizer(
        pdesys, disc_space, discretization, orders
    )
end

@testset "staggered alignment map covers every dependent variable" begin
    @parameters t x
    @variables E(..) H(..) J(..)
    Dt = Differential(t)
    Dx = Differential(x)

    eqs = [
        Dt(E(t, x)) ~ -Dx(H(t, x)) - J(t, x),
        Dt(H(t, x)) ~ -Dx(E(t, x)),
        Dt(J(t, x)) ~ 4.0 * E(t, x),
    ]
    bcs = [
        E(0, x) ~ exp(-(x - 0.5)^2 / 0.01), H(0, x) ~ 0, J(0, x) ~ 0,
        E(t, 0) ~ 0, E(t, 1) ~ 0,
    ]
    domains = [t ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eqs, bcs, domains, [t, x], [E(t, x), H(t, x), J(t, x)])
    op(u) = operation(Symbolics.unwrap(u))

    function alignments(edge_aligned_var)
        disc = MOLFiniteDifference(
            [x => 0.05], t; grid_align = MethodOfLines.StaggeredGrid(), edge_aligned_var
        )
        v = MethodOfLines.VariableMap(pdesys, disc)
        s = MethodOfLines.construct_discrete_space(v, disc)
        @test length(s.staggeredvars) == 3
        return u -> MethodOfLines.staggered_alignment(
            s, Symbolics.unwrap(u), Symbolics.unwrap(x)
        )
    end
    Edge, Center = MethodOfLines.EdgeAlignedVar, MethodOfLines.CenterAlignedVar

    # a single variable: every other variable is center aligned
    a = alignments(H(t, x))
    @test a(H(t, x)) == Edge
    @test a(E(t, x)) == Center
    @test a(J(t, x)) == Center

    # a collection edge-aligns each of its variables
    for edge in ([H(t, x), J(t, x)], (H(t, x), J(t, x)), [H(t, x) => x, J(t, x)])
        a = alignments(edge)
        @test a(H(t, x)) == Edge
        @test a(J(t, x)) == Edge
        @test a(E(t, x)) == Center
    end

    @variables B(..)
    @test_throws ArgumentError alignments(B(t, x))
end

@testset "staggered alignment per direction" begin
    @parameters t x y
    @variables Hz(..) Ex(..) Ey(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dy = Differential(y)
    X = (t, x, y)
    eqs = [
        Dt(Hz(X...)) ~ Dy(Ex(X...)) - Dx(Ey(X...)),
        Dt(Ex(X...)) ~ Dy(Hz(X...)),
        Dt(Ey(X...)) ~ -Dx(Hz(X...)),
    ]
    bcs = [
        Hz(0, x, y) ~ cos(π * x) * cos(π * y), Ex(0, x, y) ~ 0.0, Ey(0, x, y) ~ 0.0,
        Ex(t, x, 0) ~ 0, Ex(t, x, 1) ~ 0, Ey(t, 0, y) ~ 0, Ey(t, 1, y) ~ 0,
    ]
    domains = [t ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eqs, bcs, domains, [t, x, y], [Hz(X...), Ex(X...), Ey(X...)])

    function space(edge_aligned_var)
        disc = MOLFiniteDifference(
            [x => 0.1, y => 0.1], t; grid_align = MethodOfLines.StaggeredGrid(),
            edge_aligned_var
        )
        v = MethodOfLines.VariableMap(pdesys, disc)
        return MethodOfLines.construct_discrete_space(v, disc)
    end
    uw = Symbolics.unwrap
    Edge, Center = MethodOfLines.EdgeAlignedVar, MethodOfLines.CenterAlignedVar

    # the Yee layout: Hz is staggered in both directions, Ex only in x, Ey only in y
    s = space([Hz(X...), Ex(X...) => x, Ey(X...) => (y,)])
    for (u, ax, ay) in ((Hz, Edge, Edge), (Ex, Edge, Center), (Ey, Center, Edge))
        @test MethodOfLines.staggered_alignment(s, uw(u(X...)), uw(x)) == ax
        @test MethodOfLines.staggered_alignment(s, uw(u(X...)), uw(y)) == ay
        # edge aligned points sit half a step past the grid points
        @test MethodOfLines.staggered_shift(s, uw(u(X...)), uw(x)) == (ax == Edge ? 0.05 : 0)
        @test MethodOfLines.staggered_shift(s, uw(u(X...)), uw(y)) == (ay == Edge ? 0.05 : 0)
    end
    # initial conditions are evaluated at those points
    vals = Dict(MethodOfLines.gridvals(s, uw(Hz(X...)), CartesianIndex(1, 1)))
    @test vals[uw(x)] ≈ 0.05 && vals[uw(y)] ≈ 0.05

    # a direction the variable does not depend on is rejected
    @parameters w
    @test_throws ArgumentError space([Ex(X...) => w])
end
