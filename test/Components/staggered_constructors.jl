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
