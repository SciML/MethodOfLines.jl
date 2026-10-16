using ModelingToolkit, MethodOfLines, DomainSets, Test
using SciMLBase: DAEProblem
using SymbolicIndexingInterface: variable_index
using SymbolicUtils: operation
using StableRNGs

# The names of the variables whose states are in `states`.
function variable_names(prob, states)
    sys = prob.f.sys
    return Set(
        nameof(operation(u)) for u in unknowns(sys)
            if !isempty(intersect(vec(collect(variable_index(sys, u))), states))
    )
end

# Fill `du` from the groups and evaluate the full residual there. Every row that holds a
# time derivative (found by perturbing `du`) must vanish, to rounding; return the largest
# such row relative to the residual at `du = 0`.
function groups_residual(prob, sg; t = 0.37)
    rng = StableRNG(1)
    N = length(prob.u0)
    u = randn(rng, N)
    du = zeros(N)
    for g in sg.groups
        dv = zeros(length(g.states))
        g.f(dv, u, prob.p, t)
        du[g.states] = dv
    end
    F(du) = (r = zeros(N); prob.f(r, du, u, prob.p, t); r)
    F0 = F(zeros(N))
    rows = F(ones(N)) .!= F0
    # one row per differential state, and the states split into groups, constant, algebraic
    @test count(rows) == N - length(sg.algebraic)
    @test sort([reduce(vcat, [g.states for g in sg.groups]); sg.constant; sg.algebraic]) == 1:N
    @test sg.algebraic == findall(!, prob.differential_vars)
    return maximum(abs, F(du)[rows]) / maximum(abs, F0[rows])
end

@testset "1D wave equation: groups ρ | ϕ, Neumann row algebraic" begin
    @parameters t x
    @variables ρ(..) ϕ(..)
    Dt = Differential(t)
    Dx = Differential(x)
    eq = [Dt(ρ(t, x)) + Dx(ϕ(t, x)) ~ 0, Dt(ϕ(t, x)) + 25 * Dx(ρ(t, x)) ~ 0]
    bcs = [
        ρ(0, x) ~ exp(-x^2), ϕ(0.0, x) ~ 0.0,
        Dx(ρ(t, 8.0)) ~ 0.0, ϕ(t, -8.0) ~ 0.0,
    ]
    @named pdesys = PDESystem(
        eq, bcs, [t ∈ Interval(0.0, 1.0), x ∈ Interval(-8.0, 8.0)], [t, x],
        [ρ(t, x), ϕ(t, x)]
    )
    disc = MOLFiniteDifference(
        [x => 0.5], t, grid_align = MethodOfLines.StaggeredGrid(),
        edge_aligned_var = ϕ(t, x)
    )
    prob = DAEProblem(pdesys, disc)
    sg = MethodOfLines.staggered_groups(prob)
    @test variable_names(prob, sg.groups[1].states) == Set([:ρ])
    @test variable_names(prob, sg.groups[2].states) == Set([:ϕ])
    @test variable_names(prob, sg.algebraic) == Set([:ρ, :ϕ])
    @test isempty(sg.constant)
    @test groups_residual(prob, sg) < 1.0e-14
end

@testset "1D Maxwell–Drude with ε ≠ 1: groups E | H, J" begin
    t = ModelingToolkit.t_nounits
    @parameters x
    @variables E(..) H(..) J(..)
    Dt = Differential(t)
    Dx = Differential(x)
    # the coefficient of Dt(E) is divided out of the group's right hand side
    eqs = [
        2.0 * Dt(E(t, x)) ~ -Dx(H(t, x)) - J(t, x),
        Dt(H(t, x)) ~ -Dx(E(t, x)),
        Dt(J(t, x)) ~ 4.0 * E(t, x),
    ]
    bcs = [
        E(0, x) ~ exp(-(x - 0.5)^2 / 0.01), H(0, x) ~ 0, J(0, x) ~ 0,
        E(t, 0) ~ 0, E(t, 1) ~ 0,
    ]
    @named pdesys = PDESystem(
        eqs, bcs, [t ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0)], [t, x],
        [E(t, x), H(t, x), J(t, x)]
    )
    disc = MOLFiniteDifference(
        [x => 0.1], t; grid_align = MethodOfLines.StaggeredGrid(),
        edge_aligned_var = H(t, x)
    )
    prob = DAEProblem(pdesys, disc)
    sg = MethodOfLines.staggered_groups(prob)
    @test variable_names(prob, sg.groups[1].states) == Set([:E])
    @test variable_names(prob, sg.groups[2].states) == Set([:H, :J])
    @test variable_names(prob, sg.algebraic) == Set([:E])
    @test groups_residual(prob, sg) < 1.0e-14
end

@testset "A coupling graph with an odd cycle has no two groups" begin
    t = ModelingToolkit.t_nounits
    @parameters x
    @variables a(..) b(..) c(..)
    Dt = Differential(t)
    Dx = Differential(x)
    # a, b and c are coupled pairwise
    eqs = [
        Dt(a(t, x)) ~ -Dx(b(t, x)) + c(t, x),
        Dt(b(t, x)) ~ -Dx(a(t, x)),
        Dt(c(t, x)) ~ b(t, x),
    ]
    bcs = [a(0, x) ~ sin(π * x), b(0, x) ~ 0, c(0, x) ~ 0, a(t, 0) ~ 0, a(t, 1) ~ 0]
    @named pdesys = PDESystem(
        eqs, bcs, [t ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0)], [t, x],
        [a(t, x), b(t, x), c(t, x)]
    )
    disc = MOLFiniteDifference(
        [x => 0.1], t; grid_align = MethodOfLines.StaggeredGrid(),
        edge_aligned_var = b(t, x)
    )
    @test_throws "not bipartite" MethodOfLines.staggered_groups(DAEProblem(pdesys, disc))
end

# 2D TM Maxwell (ε = μ = 1) in the unit square with PEC walls, on the Yee layout: Ez on
# the nodes, Hx staggered along y and Hy along x. `drude` adds a Drude–Lorentz current J
# with its integral G, and a permittivity field Eps held fixed by Dt(Eps) ~ 0.
function tm_yee(dx; drude = false)
    @parameters t x y
    @variables Ez(..) Hx(..) Hy(..) J(..) G(..) Eps(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dy = Differential(y)
    X = (t, x, y)
    curl = Dx(Hy(X...)) - Dy(Hx(X...))
    eqs = [
        drude ? Dt(Ez(X...)) ~ (curl - J(X...)) / Eps(X...) : Dt(Ez(X...)) ~ curl,
        Dt(Hx(X...)) ~ -Dy(Ez(X...)),
        Dt(Hy(X...)) ~ Dx(Ez(X...)),
    ]
    bcs = [
        Ez(0, x, y) ~ sin(π * x) * sin(π * y), Hx(0, x, y) ~ 0.0, Hy(0, x, y) ~ 0.0,
        Ez(t, 0, y) ~ 0, Ez(t, 1, y) ~ 0, Ez(t, x, 0) ~ 0, Ez(t, x, 1) ~ 0,
    ]
    dvs = [Ez(X...), Hx(X...), Hy(X...)]
    if drude
        append!(
            eqs, [
                Dt(J(X...)) ~ 4.0 * Ez(X...) - 0.1 * J(X...) - G(X...),
                Dt(G(X...)) ~ J(X...),
                Dt(Eps(X...)) ~ 0,
            ]
        )
        append!(bcs, [J(0, x, y) ~ 0.0, G(0, x, y) ~ 0.0, Eps(0, x, y) ~ 1 + x * y])
        append!(dvs, [J(X...), G(X...), Eps(X...)])
    end
    domains = [t ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eqs, bcs, domains, [t, x, y], dvs)
    disc = MOLFiniteDifference(
        [x => dx, y => dx], t; grid_align = MethodOfLines.StaggeredGrid(),
        edge_aligned_var = [Hx(X...) => y, Hy(X...) => x]
    )
    return pdesys, disc
end

@testset "2D TM Yee with PEC walls: groups Ez | Hx, Hy" begin
    prob = DAEProblem(tm_yee(1 / 10)...)
    sg = MethodOfLines.staggered_groups(prob)
    @test variable_names(prob, sg.groups[1].states) == Set([:Ez])
    @test variable_names(prob, sg.groups[2].states) == Set([:Hx, :Hy])
    @test variable_names(prob, sg.algebraic) == Set([:Ez])
    @test isempty(sg.constant)
    @test groups_residual(prob, sg) < 1.0e-14
end

@testset "2D TM Yee with a Drude current: groups Ez, G | Hx, Hy, J" begin
    prob = DAEProblem(tm_yee(1 / 10; drude = true)...)
    sg = MethodOfLines.staggered_groups(prob)
    @test variable_names(prob, sg.groups[1].states) == Set([:Ez, :G])
    @test variable_names(prob, sg.groups[2].states) == Set([:Hx, :Hy, :J])
    # every Eps state is constant
    Eps = only(u for u in unknowns(prob.f.sys) if nameof(operation(u)) == :Eps)
    @test sort(sg.constant) == sort(vec(collect(variable_index(prob.f.sys, Eps))))
    @test variable_names(prob, sg.algebraic) == Set([:Ez])
    @test groups_residual(prob, sg) < 1.0e-14
end

# Velocity Verlet with the groups is Yee's leapfrog. Started from the (1, 1) cavity mode
# with H = 0, Ez after n steps is exactly cos(nΩdt) times its start, where Ω is the fully
# discrete Yee frequency (2/dt) asin(dt ω̃/2) of the mode's semi-discrete frequency ω̃.
@testset "The groups step a 2D cavity mode as Yee's leapfrog" begin
    dx = 1 / 10
    prob = DAEProblem(tm_yee(dx)...)
    (; groups, algebraic) = MethodOfLines.staggered_groups(prob)
    E, H = groups
    u = copy(prob.u0)
    u[algebraic] .= 0
    E0 = u[E.states]
    ω̃ = sqrt(2) * 2 / dx * sin(π * dx / 2)
    dt = 0.9 * dx / sqrt(2)
    Ω = 2 / dt * asin(dt * ω̃ / 2)
    dE, dH = zeros(length(E.states)), zeros(length(H.states))
    kick!() = (E.f(dE, u, prob.p, 0.0); u[E.states] .+= dt / 2 .* dE)
    nsteps = 200
    for _ in 1:nsteps
        kick!()
        H.f(dH, u, prob.p, 0.0)
        u[H.states] .+= dt .* dH
        kick!()
    end
    @test maximum(abs, u[E.states] .- cos(Ω * nsteps * dt) .* E0) < 1.0e-12
    @test all(iszero, u[algebraic])
end

@testset "3D Yee cavity: groups E | H" begin
    @parameters t x y z
    @variables Ex(..) Ey(..) Ez(..) Hx(..) Hy(..) Hz(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dy = Differential(y)
    Dz = Differential(z)
    X = (t, x, y, z)
    eqs = [
        Dt(Ex(X...)) ~ Dy(Hz(X...)) - Dz(Hy(X...)),
        Dt(Ey(X...)) ~ Dz(Hx(X...)) - Dx(Hz(X...)),
        Dt(Ez(X...)) ~ Dx(Hy(X...)) - Dy(Hx(X...)),
        Dt(Hx(X...)) ~ Dz(Ey(X...)) - Dy(Ez(X...)),
        Dt(Hy(X...)) ~ Dx(Ez(X...)) - Dz(Ex(X...)),
        Dt(Hz(X...)) ~ Dy(Ex(X...)) - Dx(Ey(X...)),
    ]
    ics = [
        Ex(0, x, y, z) ~ cos(π * x) * sin(π * y) * sin(π * z),
        Ey(0, x, y, z) ~ -sin(π * x) * cos(π * y) * sin(π * z),
        Ez(0, x, y, z) ~ 0.0, Hx(0, x, y, z) ~ 0.0, Hy(0, x, y, z) ~ 0.0,
        Hz(0, x, y, z) ~ 0.0,
    ]
    walls = [
        Ex(t, x, 0, z) ~ 0, Ex(t, x, 1, z) ~ 0, Ex(t, x, y, 0) ~ 0, Ex(t, x, y, 1) ~ 0,
        Ey(t, 0, y, z) ~ 0, Ey(t, 1, y, z) ~ 0, Ey(t, x, y, 0) ~ 0, Ey(t, x, y, 1) ~ 0,
        Ez(t, 0, y, z) ~ 0, Ez(t, 1, y, z) ~ 0, Ez(t, x, 0, z) ~ 0, Ez(t, x, 1, z) ~ 0,
    ]
    domains = [
        t ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0),
        z ∈ Interval(0.0, 1.0),
    ]
    @named pdesys = PDESystem(
        eqs, [ics; walls], domains, [t, x, y, z],
        [Ex(X...), Ey(X...), Ez(X...), Hx(X...), Hy(X...), Hz(X...)]
    )
    dx = 1 / 5
    disc = MOLFiniteDifference(
        [x => dx, y => dx, z => dx], t; grid_align = MethodOfLines.StaggeredGrid(),
        edge_aligned_var = [
            Ex(X...) => x, Ey(X...) => y, Ez(X...) => z,
            Hx(X...) => (y, z), Hy(X...) => (x, z), Hz(X...) => (x, y),
        ]
    )
    prob = DAEProblem(pdesys, disc)
    sg = MethodOfLines.staggered_groups(prob)
    @test variable_names(prob, sg.groups[1].states) == Set([:Ex, :Ey, :Ez])
    @test variable_names(prob, sg.groups[2].states) == Set([:Hx, :Hy, :Hz])
    @test variable_names(prob, sg.algebraic) == Set([:Ex, :Ey, :Ez])
    @test groups_residual(prob, sg) < 1.0e-14
end
