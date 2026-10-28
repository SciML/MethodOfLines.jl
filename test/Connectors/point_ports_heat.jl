# PointPorts couple a discretized PDE to lumped MTK components through `connect`.
using ModelingToolkit, MethodOfLines, DomainSets, OrdinaryDiffEq, Symbolics, Test
using MethodOfLines: select_port_nodes, port_spec
include(joinpath(@__DIR__, "..", "shared", "heat_ports.jl"))

@parameters x y ρc κ
@variables T(..) v(..)
Dx = Differential(x)
Dy = Differential(y)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

const α = 0.01
const T0 = 300.0
const Tcap = 400.0
const Ccap = 10.0

neumann_bcs(ic) = [T(0, x) ~ ic, Dx(T(t, 0)) ~ 0, Dx(T(t, 1)) ~ 0]
periodic_bcs(ic) = [T(0, x) ~ ic, T(t, 0) ~ T(t, 1)]

function rod_pdesys(
        ports; tmax = 5.0, ic = T0, pde = Dt(T(t, x)) ~ α * Dxx(T(t, x)), bcs = neumann_bcs(ic)
    )
    return PDESystem(
        [pde], bcs, [t ∈ Interval(0.0, tmax), x ∈ Interval(0.0, 1.0)], [t, x], [T(t, x)],
        [ρc => 2.0]; systems = ports, name = :rod
    )
end

function plate_pdesys(ports)
    bcs = [
        T(0, x, y) ~ T0, Dx(T(t, 0, y)) ~ 0, Dx(T(t, 1, y)) ~ 0,
        Dy(T(t, x, 0)) ~ 0, Dy(T(t, x, 1)) ~ 0,
    ]
    domains = [t ∈ Interval(0.0, 0.5), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]
    return PDESystem(
        Dt(T(t, x, y)) ~ α * (Dxx(T(t, x, y)) + Dyy(T(t, x, y))), bcs, domains,
        [t, x, y], [T(t, x, y)], [ρc => 2.0]; systems = ports, name = :plate
    )
end

discrete_space(pdesys, disc) =
    MethodOfLines.construct_discrete_space(MethodOfLines.VariableMap(pdesys, disc), disc)

# The source per unit flow that each node's equation adds to `Dt(u_I)`, read from the
# scalarized equations: -(∂R/∂Q) / (∂R/∂Dt(u_I)).
function effective_source(sys, port, dims; values = Dict())
    Q = Symbolics.unwrap(port.Q_flow)
    field = first(ModelingToolkit.get_unknowns(sys))
    eff = zeros(dims...)
    num(c) = Float64(Symbolics.unwrap_const(Symbolics.value(Symbolics.substitute(c, values))))
    for eq in equations(sys)
        eqs = Symbolics.scalarize(eq)
        for e in (eqs isa Equation ? [eqs] : vec(eqs))
            I = MethodOfLines.time_derivative_index(e, field, Symbolics.unwrap(t))
            I === nothing && continue
            R = e.lhs - e.rhs
            a = first(Symbolics.linear_expansion(R, Differential(t)(field[I...])))
            eff[I...] += -num(Symbolics.derivative(R, Q)) / num(a)
        end
    end
    return eff
end

trapezoid(n, dx) = [i == 1 || i == n ? dx / 2 : dx for i in 1:n]

function compose_with_capacitor(sys)
    @named cond = ThermalConductor()
    @named cap = HeatCapacitor(C = Ccap, T0 = Tcap)
    @named model = System(
        [connect(sys.heater, cond.port_a), connect(cond.port_b, cap.port)], t;
        systems = [sys, cond, cap]
    )
    return model, cap
end

function solve_with_capacitor(sys, tspan; op = nothing)
    model, cap = compose_with_capacitor(sys)
    prob = ODEProblem(mtkcompile(model), op, tspan)
    return solve(prob, FBDF(); reltol = 1.0e-10, abstol = 1.0e-10), cap
end

# Relative energy imbalance between the capacitor and the field, with control volumes V.
function imbalance(sol, sys, cap, V; heat_capacity = 2.0)
    Efield = heat_capacity * sum(V .* (sol[sys.T][end] .- sol[sys.T][1]))
    Ecap = Ccap * (sol[cap.T][end] - Tcap)
    return abs(Efield + Ecap) / abs(Ecap)
end

heater_1d(; select = Interpolate(), at = 0.31, scale = 1 / ρc) = PointPort(
    HeatPort; name = :heater, var = T(t, x), at = [x => at], across = :T,
    through = :Q_flow => scale, select
)

@testset "1-D rod: ported System" begin
    heater = heater_1d(scale = 0.5)
    @named probe = PointPort(HeatPort; var = T(t, x), at = [x => 0.71], across = :T)
    pdesys = rod_pdesys([heater, probe])
    disc = MOLFiniteDifference([x => 0.02], t)
    sys, _ = symbolic_discretize(pdesys, disc)
    @test nameof.(ModelingToolkit.get_systems(sys)) == [:heater, :probe]
    @test any(eq -> MethodOfLines.is_array_valued(eq.lhs), equations(sys))

    nodes, w, V = select_port_nodes(port_spec(heater), discrete_space(pdesys, disc); restrict_to = (2:50,))
    eff = effective_source(sys, heater, 51)
    @test findall(!iszero, eff) == first.(Tuple.(nodes))
    @test eff[first.(Tuple.(nodes))] ≈ 0.5 .* w ./ V
    @test sum(trapezoid(51, 0.02) .* eff) ≈ 0.5
end

@testset "Source does not depend on how the PDE is written" begin
    disc = MOLFiniteDifference([x => 0.02], t)
    forms = [
        Dt(T(t, x)) ~ α * Dxx(T(t, x)),
        α * Dxx(T(t, x)) ~ Dt(T(t, x)),
        κ * Dt(T(t, x)) ~ κ * α * Dxx(T(t, x)),
        0 ~ α * Dxx(T(t, x)) - Dt(T(t, x)),
    ]
    effs = map(forms) do pde
        heater = heater_1d(scale = 0.5)
        sys, _ = symbolic_discretize(rod_pdesys([heater]; pde), disc)
        effective_source(sys, heater, 51; values = Dict(κ => 3.0))
    end
    @test all(eff -> eff ≈ effs[1], effs)
    @test sum(trapezoid(51, 0.02) .* effs[1]) ≈ 0.5

    @test_throws r"constant or parameter coefficient" symbolic_discretize(
        rod_pdesys([heater_1d()]; pde = x * Dt(T(t, x)) ~ α * Dxx(T(t, x))), disc
    )
end

# Rate added to [Dt(T[k]), Dt(v[k])] per unit flow, from the residuals R that evolve node k:
# A = ∂R/∂Dt, b = -∂R/∂Q_flow, rate = A \ b.
function coupled_rate(sys, port, k)
    fields = ModelingToolkit.get_unknowns(sys)
    field(n) = only(filter(u -> startswith(string(u), n), fields))
    dts = [Differential(t)(Symbolics.unwrap(field(n))[k]) for n in ("T(t)", "v(t)")]
    rows = Equation[]
    for eq in equations(sys)
        eqs = Symbolics.scalarize(eq)
        for e in (eqs isa Equation ? [eqs] : vec(eqs))
            any(d -> any(isequal(d), dts), Symbolics.get_differential_vars(e)) && push!(rows, e)
        end
    end
    num(c) = Float64(Symbolics.unwrap_const(Symbolics.value(c)))
    R = [e.lhs - e.rhs for e in rows]
    A = [num(first(Symbolics.linear_expansion(r, d))) for r in R, d in dts]
    b = [-num(Symbolics.derivative(r, Symbolics.unwrap(port.Q_flow))) for r in R]
    return A \ b
end

@testset "Flow into a variable whose time derivative appears in several PDEs" begin
    # The flow must add `scale * Q` to Dt(T) of the coupled system, not to the residual of
    # the one PDE assigned to T: here A = [1 1; -1 1] in [T, v] order.
    eqs = [
        Dt(T(t, x)) + Dt(v(t, x)) ~ 0.01Dx(Dx(T(t, x))),
        Dt(v(t, x)) - Dt(T(t, x)) ~ 0.01Dx(Dx(v(t, x))),
    ]
    bcs = [
        T(0, x) ~ 300.0, v(0, x) ~ 300.0, Dx(T(t, 0)) ~ 0, Dx(T(t, 1)) ~ 0,
        Dx(v(t, 0)) ~ 0, Dx(v(t, 1)) ~ 0,
    ]
    disc = MOLFiniteDifference([x => 0.1], t)
    for (x0, weights) in ((0.5, [6 => 1.0]), (0.53, [6 => 0.7, 7 => 0.3]))
        @named hmass = PointPort(
            HeatPort; var = T(t, x), at = [x => x0], across = :T, through = :Q_flow => 1
        )
        @named cpl = PDESystem(
            eqs, bcs, [t ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0)], [t, x],
            [T(t, x), v(t, x)]; systems = [hmass]
        )
        sys, _ = symbolic_discretize(cpl, disc)
        for (k, wk) in weights
            @test 0.1 .* coupled_rate(sys, hmass, k) ≈ [wk, 0.0] atol = 1.0e-12
        end
    end
end

@testset "1-D rod: read-out port at top level" begin
    # At top level the rod's own ports are outer connectors, whose flows MTK sets to zero.
    heater = heater_1d()
    @named probe = PointPort(HeatPort; var = T(t, x), at = [x => 0.71], across = :T)
    pdesys = rod_pdesys([heater, probe]; ic = T0 + 50cospi(x))
    disc = MOLFiniteDifference([x => 0.02], t)
    sys, tspan = symbolic_discretize(pdesys, disc)
    sol = solve(ODEProblem(mtkcompile(sys), nothing, tspan), FBDF(); reltol = 1.0e-10, abstol = 1.0e-10)
    @test SciMLBase.successful_retcode(sol)
    nodes, w, _ = select_port_nodes(port_spec(probe), discrete_space(pdesys, disc))
    Tend = sol[T(t, x)][end, :]
    @test sol.original_sol[probe.T][end] ≈ sum(w .* Tend[first.(Tuple.(nodes))])
    @test all(iszero, sol.original_sol[heater.Q_flow])
end

@testset "1-D rod: pointwise host, parameter only in scale" begin
    # at approx_order = 4 node 2 has its own (frame) equation rather than the slice's
    heater = heater_1d(select = Nearest(), at = 0.021, scale = 1 / κ)
    disc = MOLFiniteDifference([x => 0.02], t; approx_order = 4)
    sys, _ = symbolic_discretize(rod_pdesys([heater]), disc)
    @test any(isequal(κ), parameters(sys))
    eff = effective_source(sys, heater, 51; values = Dict(κ => 2.0))
    @test findall(!iszero, eff) == [2]
    @test eff[2] ≈ 1 / (2.0 * 0.02)
end

@testset "Periodic rod: ports across the seam" begin
    disc = MOLFiniteDifference([x => 0.02], t)
    for at in (0.99, 0.005)
        heater = heater_1d(; at, scale = 1)
        sys, _ = symbolic_discretize(rod_pdesys([heater]; bcs = periodic_bcs(T0)), disc)
        eff = effective_source(sys, heater, 51)
        @test eff[1] == 0
        @test sum(0.02 .* eff[2:51]) ≈ 1
    end
end

@testset "2-D plate: ported System" begin
    @named heater = PointPort(
        HeatPort; var = T(t, x, y), at = [x => 0.53, y => 0.47], across = :T,
        through = :Q_flow => 1
    )
    sys, _ = symbolic_discretize(plate_pdesys([heater]), MOLFiniteDifference([x => 0.1, y => 0.1], t))
    @test any(eq -> MethodOfLines.is_array_valued(eq.lhs), equations(sys))
    eff = effective_source(sys, heater, (11, 11))
    @test count(!iszero, eff) == 4
    @test all(>=(0), eff)
    @test sum(trapezoid(11, 0.1) * trapezoid(11, 0.1)' .* eff) ≈ 1
end

@testset "Nested composition with lumped components" begin
    @testset "1-D rod, Interpolate" begin
        heater = heater_1d()
        @named probe = PointPort(HeatPort; var = T(t, x), at = [x => 0.71], across = :T)
        pdesys = rod_pdesys([heater, probe])
        disc = MOLFiniteDifference([x => 0.02], t)
        sys, tspan = symbolic_discretize(pdesys, disc)
        sol, cap = solve_with_capacitor(sys, tspan)
        @test SciMLBase.successful_retcode(sol)
        # positive into the rod while the capacitor is hotter
        @test sol[sys.heater.Q_flow][1] ≈ Tcap - T0
        s = discrete_space(pdesys, disc)
        Tend = sol[sys.T][end]
        for (port, interior) in ((heater, (2:50,)), (probe, nothing))
            nodes, w, _ = select_port_nodes(port_spec(port), s; restrict_to = interior)
            @test sol[getproperty(sys, nameof(port)).T][end] ≈ sum(w .* Tend[first.(Tuple.(nodes))])
        end
        @test imbalance(sol, sys, cap, trapezoid(51, 0.02)) < 1.0e-3
    end

    @testset "PDE written as α Dxx(T) ~ Dt(T)" begin
        pde = α * Dxx(T(t, x)) ~ Dt(T(t, x))
        sys, tspan = symbolic_discretize(rod_pdesys([heater_1d()]; pde), MOLFiniteDifference([x => 0.02], t))
        sol, cap = solve_with_capacitor(sys, tspan)
        @test SciMLBase.successful_retcode(sol)
        @test sol[cap.T][end] < Tcap
        @test imbalance(sol, sys, cap, trapezoid(51, 0.02)) < 1.0e-3
    end

    @testset "Edge-aligned rod conserves to solver tolerance" begin
        disc = MOLFiniteDifference([x => 0.02], t; grid_align = edge_align)
        sys, tspan = symbolic_discretize(rod_pdesys([heater_1d(at = 0.32)]), disc)
        sol, cap = solve_with_capacitor(sys, tspan)
        @test SciMLBase.successful_retcode(sol)
        V = [i == 1 || i == 52 ? 0.0 : 0.02 for i in 1:52]
        @test imbalance(sol, sys, cap, V) < 1.0e-8
    end

    @testset "Periodic rod conserves across the seam" begin
        for at in (0.99, 0.005)
            pdesys = rod_pdesys([heater_1d(; at)]; bcs = periodic_bcs(T0))
            sys, tspan = symbolic_discretize(pdesys, MOLFiniteDifference([x => 0.02], t))
            sol, cap = solve_with_capacitor(sys, tspan)
            @test SciMLBase.successful_retcode(sol)
            V = [i == 1 ? 0.0 : 0.02 for i in 1:51]
            @test imbalance(sol, sys, cap, V) < 1.0e-8
        end
    end

    @testset "1-D rod, Nearest on a pointwise host" begin
        heater = heater_1d(select = Nearest(), at = 0.021, scale = 1 / κ)
        disc = MOLFiniteDifference([x => 0.02], t; approx_order = 4)
        sys, tspan = symbolic_discretize(rod_pdesys([heater]), disc)
        sol, _ = solve_with_capacitor(sys, tspan; op = [sys.κ => 2.0])
        @test SciMLBase.successful_retcode(sol)
        @test sol[sys.heater.T] ≈ getindex.(sol[sys.T], 2)
        @test sol[sys.heater.Q_flow][1] ≈ Tcap - T0
    end

    @testset "2-D plate" begin
        @named heater = PointPort(
            HeatPort; var = T(t, x, y), at = [x => 0.53, y => 0.47], across = :T,
            through = :Q_flow => 1 / ρc
        )
        sys, tspan = symbolic_discretize(
            plate_pdesys([heater]), MOLFiniteDifference([x => 0.1, y => 0.1], t)
        )
        sol, cap = solve_with_capacitor(sys, tspan)
        @test SciMLBase.successful_retcode(sol)
        V = trapezoid(11, 0.1) * trapezoid(11, 0.1)'
        @test imbalance(sol, sys, cap, V) < 1.0e-2
    end
end

@testset "Errors" begin
    disc = MOLFiniteDifference([x => 0.02], t)
    @test_throws ArgumentError discretize(rod_pdesys([heater_1d()]), disc)
    @test_throws ArgumentError DAEProblem(rod_pdesys([heater_1d()]), disc)
    @test_throws r"flux boundary condition" symbolic_discretize(rod_pdesys([heater_1d(at = 0.01)]), disc)

    # Dirichlet ends on a grid whose first cells are far smaller than the domain: a flow
    # at or just inside a boundary is still outside the interior nodes
    stretched = MOLFiniteDifference([x => [0.0, 1.0e-10, 2.0e-10, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]], t)
    dirichlet = [T(0, x) ~ T0, T(t, 0) ~ T0, T(t, 1) ~ T0]
    for x0 in (0.0, 5.0e-11, 1.0)
        @test_throws r"flux boundary condition" symbolic_discretize(
            rod_pdesys([heater_1d(at = x0)]; bcs = dirichlet), stretched
        )
    end
    @test symbolic_discretize(rod_pdesys([heater_1d(at = 1.0e-10)]; bcs = dirichlet), stretched) isa Tuple

    # a flow into a variable without a time derivative
    @named hv = PointPort(HeatPort; var = v(t, x), at = [x => 0.5], across = :T, through = :Q_flow => 1)
    eqs = [Dt(T(t, x)) ~ Dxx(T(t, x)) + v(t, x), 0 ~ Dxx(v(t, x)) - v(t, x) + T(t, x)]
    bcs = [
        T(0, x) ~ T0, T(t, 0) ~ T0, T(t, 1) ~ T0, v(0, x) ~ 0.0, v(t, 0) ~ 0.0, v(t, 1) ~ 0.0,
    ]
    @named coupled = PDESystem(
        eqs, bcs, [t ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0)], [t, x],
        [T(t, x), v(t, x)]; systems = [hv]
    )
    @test_throws r"no PDE contains" symbolic_discretize(coupled, disc)
end

@testset "Port-free systems are unchanged" begin
    disc = MOLFiniteDifference([x => 0.02], t)
    sys, _ = symbolic_discretize(rod_pdesys([]), disc)
    generic, _ = invoke(
        symbolic_discretize,
        Tuple{PDESystem, MethodOfLines.PDEBase.AbstractEquationSystemDiscretization},
        rod_pdesys([]), disc
    )
    @test isempty(ModelingToolkit.get_systems(sys))
    @test isequal(equations(sys), equations(generic))
    @test isempty(disc.ports)
end
