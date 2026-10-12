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
    @test disc_space.staggeredvars[operation(Symbolics.unwrap(ρ(t, x)))] ==
        MethodOfLines.CenterAlignedVar
    @test disc_space.staggeredvars[operation(Symbolics.unwrap(ϕ(t, x)))] ==
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
        return MethodOfLines.construct_discrete_space(v, disc).staggeredvars
    end
    Edge, Center = MethodOfLines.EdgeAlignedVar, MethodOfLines.CenterAlignedVar

    # a single variable: every other variable is center aligned
    a = alignments(H(t, x))
    @test length(a) == 3
    @test a[op(H(t, x))] == Edge
    @test a[op(E(t, x))] == Center
    @test a[op(J(t, x))] == Center

    # a collection edge-aligns each of its variables
    for edge in ([H(t, x), J(t, x)], (H(t, x), J(t, x)))
        a = alignments(edge)
        @test length(a) == 3
        @test a[op(H(t, x))] == Edge
        @test a[op(J(t, x))] == Edge
        @test a[op(E(t, x))] == Center
    end

    @variables B(..)
    @test_throws ArgumentError alignments(B(t, x))
end
