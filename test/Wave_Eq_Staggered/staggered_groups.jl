using ModelingToolkit, MethodOfLines, DomainSets, Test
using SciMLBase: DAEProblem
using SymbolicIndexingInterface: variable_index
using SymbolicUtils: SymbolicUtils, operation
using Symbolics: Symbolics
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

# `update!` steps each group in place exactly as `u[states] .+= dt .* dv` does with the
# group's `f` (or `.= a .* u .+ b .* dv` when `lossy`), bit for bit, and writes no other
# state: constant and algebraic states keep their values.
function updates_match(prob, sg; lossy = false, t = 0.37, dt = 0.013)
    rng = StableRNG(2)
    N = length(prob.u0)
    u, a, b = randn(rng, N), 1 .+ 0.1 .* randn(rng, N), dt .* (1 .+ 0.1 .* randn(rng, N))
    return all(sg.groups) do g
        dv = zeros(length(g.states))
        g.f(dv, u, prob.p, t)
        expected = copy(u)
        if lossy
            expected[g.states] .= a[g.states] .* u[g.states] .+ b[g.states] .* dv
        else
            expected[g.states] .+= dt .* dv
        end
        stepped = copy(u)
        lossy ? g.update!(stepped, prob.p, t, dt, a, b) : g.update!(stepped, prob.p, t, dt)
        stepped == expected
    end
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
    @test updates_match(prob, sg)
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
    @test updates_match(prob, sg)
    @test updates_match(prob, MethodOfLines.staggered_groups(prob; lossy = true); lossy = true)
end

@testset "Periodic Maxwell–Drude with pointwise interiors: groups E | H, J" begin
    t = ModelingToolkit.t_nounits
    @parameters x
    @variables E(..) H(..) J(..)
    Dt = Differential(t)
    Dx = Differential(x)
    eqs = [
        2.0 * Dt(E(t, x)) ~ -Dx(H(t, x)) - J(t, x),
        0.5 * Dt(H(t, x)) ~ -Dx(E(t, x)),
        Dt(J(t, x)) ~ 4.0 * E(t, x),
    ]
    # J has no periodic condition, so the E and J interiors are discretized point by point,
    # as scalar equations, while H keeps its slice form
    bcs = [
        E(0, x) ~ cos(2π * x), H(0, x) ~ 0.0, J(0, x) ~ 0.0,
        E(t, 0) ~ E(t, 1), H(t, 0) ~ H(t, 1),
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
    @test variable_names(prob, sg.algebraic) == Set([:E, :H])
    @test groups_residual(prob, sg) < 1.0e-14
    @test updates_match(prob, sg)
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
    @test updates_match(prob, sg)
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
    @test updates_match(prob, sg)
    @test updates_match(prob, MethodOfLines.staggered_groups(prob; lossy = true); lossy = true)
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
    @test updates_match(prob, sg)
end

@testset "An in-place update must not read other states of its own group" begin
    @variables t x(t)[1:5]
    D = Differential(t)
    slice(r) = Symbolics.unwrap(x[r])
    # spelled as MOL spells a discretized equation, a broadcast over slices
    eq(r, rhs...) = (lhs = Symbolics.unwrap(D(x[r])), rhs = MethodOfLines._broadcast(rhs...))
    @test MethodOfLines._group_race([eq(2:4, *, 2, slice(2:4))]) === nothing
    shifted = eq(2:4, +, slice(1:3), slice(2:4))
    race = MethodOfLines._group_race([shifted])
    @test race !== nothing
    # the slice read, compared by value: `isequal` tells equal ranges apart by their type
    @test SymbolicUtils.unwrap_const.(SymbolicUtils.arguments(race[2])[2:end]) == [1:3]
end

@testset "Staggered updates read the complete group entry state" begin
    for n in (5, 20), scalar_seam in (false, true), periodic in (false, true),
            scalarize in (false, true), lossy in (false, true)
        @testset "n=$n scalar=$scalar_seam periodic=$periodic scalarize=$scalarize lossy=$lossy" begin
            @variables t v(t)[1:n] h(t)[1:n] c(t)[1:n]
            D = Differential(t)
            seam = if periodic
                scalar_seam ? D(v[n]) ~ v[1] - v[n] : D(v[n:n]) ~ v[1:1] .- v[n:n]
            else
                D(v[n:n]) ~ zeros(1)
            end
            eqs = [D(v[1:(n - 1)]) ~ v[2:n] .- v[1:(n - 1)], seam, D(h) ~ v, D(c) ~ zeros(n)]
            @named raw = System(eqs, t, [v, h, c], [])
            sys = complete(raw)
            prob = DAEProblem(
                sys, [
                    v => collect(1.0:n), h => zeros(n), c => fill(3.0, n),
                    D(v) => zeros(n), D(h) => zeros(n), D(c) => zeros(n),
                ],
                (0.0, 1.0); build_initializeprob = false
            )
            sg = MethodOfLines.staggered_groups(
                prob; lossy, scalarize_single_element_regions = scalarize
            )
            iv = vec(collect(variable_index(sys, v)))
            for T in (Float32, Float64), coefficient_alias in (false, true)
                storage = [T(-777); T.(prob.u0); T(-888)]
                u = view(storage, 2:(length(storage) - 1))
                a = coefficient_alias ? u : fill(T(0.75), length(u))
                b = fill(T(0.125), length(u))
                dt = T(0.125)
                for iteration in 1:2, group in sg.groups
                    entry = copy(u)
                    aa = copy(a)
                    dv = zeros(T, length(group.states))
                    group.f(dv, entry, prob.p, T(0.25))
                    expected = copy(entry)
                    if lossy
                        expected[group.states] .= aa[group.states] .* entry[group.states] .+
                            b[group.states] .* dv
                    else
                        expected[group.states] .+= dt .* dv
                    end
                    lossy ? group.update!(u, prob.p, T(0.25), dt, a, b) :
                        group.update!(u, prob.p, T(0.25), dt)
                    @test u == expected
                    untouched = setdiff(eachindex(u), group.states)
                    @test u[untouched] == entry[untouched]
                    @test storage[1] == -777 && storage[end] == -888
                    if periodic && !lossy && iteration == 1 && first(iv) in group.states
                        @test sum(u[iv]) == sum(entry[iv])
                    end
                end
            end
        end
    end
end

# A periodic forward difference on 20 points, whose seam reads the state the first region
# writes, spelled as a slice and as a scalar: one step must read the group's state on entry,
# and so keep the total at 210.
@testset "update! reads its group's state on entry across a periodic seam" begin
    @variables t v(t)[1:20] h(t)[1:20]
    D = Differential(t)
    dt = 1 / 8
    for seam in (D(v[20:20]) ~ v[1:1] .- v[20:20], D(v[20]) ~ v[1] - v[20])
        @named raw = System([D(v[1:19]) ~ v[2:20] .- v[1:19], seam, D(h) ~ v], t, [v, h], [])
        sys = complete(raw)
        prob = DAEProblem(
            sys, [v => collect(1.0:20.0), h => zeros(20), D(v) => zeros(20), D(h) => zeros(20)],
            (0.0, 1.0); build_initializeprob = false
        )
        ids = vec(collect(variable_index(sys, v)))
        expected = copy(prob.u0)
        expected[ids] .+= dt .* (circshift(prob.u0[ids], -1) .- prob.u0[ids])
        for fuse_broadcasts in (true, false), scalarize_single_element_regions in (true, false),
                lossy in (false, true)
            sg = MethodOfLines.staggered_groups(
                prob; lossy, fuse_broadcasts, scalarize_single_element_regions
            )
            g = only(filter(g -> ids[1] in g.states, sg.groups))
            u = copy(prob.u0)
            lossy ? g.update!(u, prob.p, 0.0, dt, ones(length(u)), fill(dt, length(u))) :
                g.update!(u, prob.p, 0.0, dt)
            @test u == expected
            @test sum(u[ids]) == 210.0
        end
    end
end

# Only a group that reads its own states at other elements copies `u` on entry, into the
# buffer its `update!` holds as `entry`; Yee's groups, plain or lossy, read `u` itself. With
# fused code generation no `update!` allocates. The seam is kept a one-element region: its
# scalarized write allocates 64 bytes in the generated code itself, entry copy or not.
function seam_problem()
    @variables t v(t)[1:20] h(t)[1:20]
    D = Differential(t)
    eqs = [D(v[1:19]) ~ v[2:20] .- v[1:19], D(v[20]) ~ v[1] - v[20], D(h) ~ v]
    @named raw = System(eqs, t, [v, h], [])
    return DAEProblem(
        complete(raw),
        [v => collect(1.0:20.0), h => zeros(20), D(v) => zeros(20), D(h) => zeros(20)],
        (0.0, 1.0); build_initializeprob = false
    )
end
fused = isdefined(Symbolics, :CodegenFunctionOptions) &&
    hasfield(Symbolics.CodegenFunctionOptions, :fuse_broadcasts)
@testset "Only a racing group's update! copies its entry state" begin
    allocations(update!, args...) = (update!(args...); @allocated update!(args...))
    for (prob, racing, scalarize_single_element_regions) in (
            (DAEProblem(tm_yee(1 / 10)...), (false, false), true),
            (seam_problem(), (true, false), false),
        )
        N = length(prob.u0)
        for lossy in (false, true)
            groups = MethodOfLines.staggered_groups(
                prob; lossy, scalarize_single_element_regions
            ).groups
            @test map(g -> g.update!.entry isa Vector{Float64}, groups) == racing
            @test all(g -> g.update!.entry === nothing || length(g.update!.entry) == N, groups)
            fused || continue
            for g in groups
                args = lossy ? (copy(prob.u0), prob.p, 0.0, 0.01, ones(N), fill(0.01, N)) :
                    (copy(prob.u0), prob.p, 0.0, 0.01)
                @test allocations(g.update!, args...) == 0
            end
        end
    end
end
