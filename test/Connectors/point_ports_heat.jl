# PointPorts couple a discretized PDE to lumped MTK components through `connect`.
using ModelingToolkit, MethodOfLines, DomainSets, OrdinaryDiffEq, Symbolics, Test
using MethodOfLines: select_port_nodes, port_spec
include(joinpath(@__DIR__, "..", "shared", "heat_ports.jl"))

@parameters x y ρc κ
@variables T(..)
Dx = Differential(x)
Dy = Differential(y)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

const α = 0.01
const T0 = 300.0
const Tcap = 400.0
const Ccap = 10.0

# Nesting the discretized system as a subsystem namespaces its slice-form equations,
# which needs SciML/ModelingToolkit.jl#5133.
const nested = MethodOfLines._renamespace_supports_slices()

function rod_pdesys(ports; tmax = 5.0, ic = T0)
    return PDESystem(
        [Dt(T(t, x)) ~ α * Dxx(T(t, x))],
        [T(0, x) ~ ic, Dx(T(t, 0)) ~ 0, Dx(T(t, 1)) ~ 0],
        [t ∈ Interval(0.0, tmax), x ∈ Interval(0.0, 1.0)], [t, x], [T(t, x)], [ρc => 2.0];
        systems = ports, name = :rod
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

# d(residual of node I)/d(Q_flow) for every node, read from the scalarized equations.
function source_coefficients(sys, port, dims; values = Dict())
    Q = Symbolics.unwrap(port.Q_flow)
    field = first(ModelingToolkit.get_unknowns(sys))
    coef = zeros(dims...)
    for eq in equations(sys)
        eqs = Symbolics.scalarize(eq)
        for e in (eqs isa Equation ? [eqs] : vec(eqs))
            I = MethodOfLines.time_derivative_index(e, field, Symbolics.unwrap(t))
            I === nothing && continue
            c = Symbolics.substitute(Symbolics.derivative(e.lhs - e.rhs, Q), values)
            coef[I...] += Float64(Symbolics.unwrap_const(Symbolics.value(c)))
        end
    end
    return coef
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

    nodes, w, V = select_port_nodes(port_spec(heater), discrete_space(pdesys, disc), 2; restrict_to = (2:50,))
    coef = source_coefficients(sys, heater, 51)
    @test findall(!iszero, coef) == first.(Tuple.(nodes))
    @test coef[first.(Tuple.(nodes))] ≈ -0.5 .* w ./ V
    @test sum(trapezoid(51, 0.02) .* coef) ≈ -0.5
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
    nodes, w, _ = select_port_nodes(port_spec(probe), discrete_space(pdesys, disc), 2)
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
    coef = source_coefficients(sys, heater, 51; values = Dict(κ => 2.0))
    @test findall(!iszero, coef) == [2]
    @test coef[2] ≈ -1 / (2.0 * 0.02)
end

@testset "2-D plate: ported System" begin
    @named heater = PointPort(
        HeatPort; var = T(t, x, y), at = [x => 0.53, y => 0.47], across = :T,
        through = :Q_flow => 1
    )
    sys, _ = symbolic_discretize(plate_pdesys([heater]), MOLFiniteDifference([x => 0.1, y => 0.1], t))
    @test any(eq -> MethodOfLines.is_array_valued(eq.lhs), equations(sys))
    coef = source_coefficients(sys, heater, (11, 11))
    @test count(!iszero, coef) == 9
    @test sum(trapezoid(11, 0.1) * trapezoid(11, 0.1)' .* coef) ≈ -1
end

@testset "Nested composition with lumped components" begin
    if !nested
        heater = heater_1d()
        sys, _ = symbolic_discretize(rod_pdesys([heater]), MOLFiniteDifference([x => 0.02], t))
        compiles = try
            mtkcompile(first(compose_with_capacitor(sys)))
            true
        catch
            false
        end
        @test_broken compiles
    else
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
                nodes, w, _ = select_port_nodes(port_spec(port), s, 2; restrict_to = interior)
                @test sol[getproperty(sys, nameof(port)).T][end] ≈ sum(w .* Tend[first.(Tuple.(nodes))])
            end
            Erod = 2.0 * sum(trapezoid(51, 0.02) .* (Tend .- sol[sys.T][1]))
            Ecap = Ccap * (sol[cap.T][end] - Tcap)
            @test abs(Erod + Ecap) / abs(Ecap) < 1.0e-3
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
            Eplate = 2.0 * sum(V .* reshape(sol[sys.T][end] .- sol[sys.T][1], 11, 11))
            Ecap = Ccap * (sol[cap.T][end] - Tcap)
            @test abs(Eplate + Ecap) / abs(Ecap) < 1.0e-2
        end
    end
end

@testset "Errors" begin
    disc = MOLFiniteDifference([x => 0.02], t)
    @test_throws ArgumentError discretize(rod_pdesys([heater_1d()]), disc)
    @test_throws r"flux boundary condition" symbolic_discretize(rod_pdesys([heater_1d(at = 0.01)]), disc)
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
