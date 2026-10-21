"""
    AbstractPortSelection

How a [`PointPort`](@ref) picks the grid nodes and weights that couple it to the field.
"""
abstract type AbstractPortSelection end

"""
    Nearest()

Couple a [`PointPort`](@ref) to the single grid node nearest its coordinates, in every
spatial direction. Ties go to the lower index.
"""
struct Nearest <: AbstractPortSelection end

"""
    Interpolate(order = 1)

Couple a [`PointPort`](@ref) to the field by Lagrange interpolation of degree `order` in
every spatial direction: `order + 1` nodes per direction, grown outward from the two nodes
that bracket the port, so it never extrapolates. A port that lies exactly on a node uses
that node alone. The default linear interpolation has non-negative weights; `order >= 2`
is more accurate on smooth fields but has negative weights, so a positive flow also
removes a little of the quantity at some nodes (a non-monotone source).
"""
struct Interpolate <: AbstractPortSelection
    order::Int
    function Interpolate(order::Integer)
        order >= 0 || throw(ArgumentError("Interpolate order must be >= 0, got $order"))
        return new(order)
    end
end
Interpolate() = Interpolate(1)

port_npoints(::Nearest) = 1
port_npoints(sel::Interpolate) = sel.order + 1

struct PortSpec
    var::Any
    at::Vector{Pair{Any, Float64}}
    across::Symbol
    through::Union{Nothing, Tuple{Symbol, Any}}
    select::AbstractPortSelection
end

struct PortSpecKey end

port_spec(sys) = getmetadata(sys, PortSpecKey, nothing)

"""
    PointPort(connector; name, var, at, across, through = nothing,
              select = Interpolate(), kwargs...)

Declare an MTK connector port on the dependent variable `var` of a `PDESystem`, at the
spatial point `at`, to be placed in the `systems` of that `PDESystem`.

`connector(; name, kwargs...)` must return a connector (a `System` built with
`@connector`). `at` gives one `x => value` pair for every spatial argument of `var`.
After discretization the connector's unknown `across` equals the selected field value
`Σ w_I u_I`. If `through = :Q => scale` names a `Flow` unknown of the connector, the port
adds `scale * Q * w_I / V_I` (`V_I` the control volume of node `I`) to the time derivative
of `var` at each selected node, as if the PDE were solved for `Dt(var)`, however it is
written: `ρ * c * Dt(T) ~ k * Dxx(T)` and `Dt(T) ~ k / (ρ * c) * Dxx(T)` take the same
`scale`. The PDE must contain `Dt(var)` linearly, with a constant or parameter
coefficient. `Q` is positive into the PDE, the MTK convention for a component's own port;
`scale` converts the flow into a rate of `var`, e.g. `1 / (ρ * c * A)` for a temperature
rod of cross-section `A`. Without `through` the port only reads the field.

`select` is [`Interpolate`](@ref) or [`Nearest`](@ref). A port carrying a flow selects
among interior nodes only, since boundary nodes hold boundary conditions; flow into a
boundary belongs in a flux boundary condition instead.

Compose the discretized system rather than calling `discretize`:

```julia
@named heater = PointPort(HeatPort; var = T(t, x), at = [x => 0.3], across = :T,
                          through = :Q_flow => 1 / (ρ * c * A))
@named rod = PDESystem(eqs, bcs, domains, [t, x], [T(t, x)], ps; systems = [heater])
sys, tspan = symbolic_discretize(rod, MOLFiniteDifference([x => 0.01], t))
@named model = System([connect(sys.heater, cap.port)], t; systems = [sys, cap])
prob = ODEProblem(mtkcompile(model), nothing, tspan)
```

A read-out port also works on `sys` compiled on its own, where its flow is zero.
"""
function PointPort(
        connector; name, var, at, across::Symbol, through = nothing,
        select::AbstractPortSelection = Interpolate(), kwargs...
    )
    sys = connector(; name, kwargs...)
    ModelingToolkitBase.has_connector_type(sys) &&
        ModelingToolkitBase.get_connector_type(sys) !== nothing ||
        throw(ArgumentError("PointPort $name: `$connector` did not return a connector; define it with `@connector`."))
    portvars = Dict(SymbolicIndexingInterface.getname(v) => v for v in get_unknowns(sys))
    haskey(portvars, across) ||
        throw(ArgumentError("PointPort $name: `across = :$across` is not an unknown of the connector, which has $(collect(keys(portvars)))."))
    ModelingToolkit.getconnect(portvars[across]) === ModelingToolkit.Flow &&
        throw(ArgumentError("PointPort $name: `across = :$across` is a Flow variable; name the potential variable instead."))
    if through !== nothing
        through isa Symbol && (through = through => 1)
        through isa Pair{Symbol} ||
            throw(ArgumentError("PointPort $name: `through` must be `:name => scale`, got $through."))
        flow = through.first
        haskey(portvars, flow) && ModelingToolkit.getconnect(portvars[flow]) === ModelingToolkit.Flow ||
            throw(ArgumentError("PointPort $name: `through = :$flow` must name a `connect = Flow` unknown of the connector."))
        through = (flow, through.second)
    end

    u = safe_unwrap(var)
    iscall(u) || throw(ArgumentError("PointPort $name: `var` must be a dependent variable such as `u(t, x)`, got $var."))
    at = at isa Pair ? [at] : collect(at)
    isempty(at) && throw(ArgumentError("PointPort $name: `at` must give a coordinate for each spatial argument of $var."))
    keys_ = map(p -> safe_unwrap(p.first), at)
    for (i, x) in enumerate(keys_)
        any(a -> isequal(a, x), arguments(u)) ||
            throw(ArgumentError("PointPort $name: `$x` in `at` is not an argument of $var."))
        any(y -> isequal(y, x), keys_[1:(i - 1)]) &&
            throw(ArgumentError("PointPort $name: `$x` is given more than once in `at`."))
    end
    for p in at
        unwrap_const(safe_unwrap(p.second)) isa Real ||
            throw(ArgumentError("PointPort $name: coordinates in `at` must be numbers, got $(p.first) => $(p.second)."))
    end
    spec = PortSpec(
        u, Pair{Any, Float64}[x => Float64(unwrap_const(safe_unwrap(p.second))) for (x, p) in zip(keys_, at)],
        across, through, select
    )
    return setmetadata(sys, PortSpecKey, spec)
end

"""
The `PointPort`s among the subsystems of `pdesys`.
"""
point_port_systems(pdesys) = filter(sys -> port_spec(sys) !== nothing, ModelingToolkit.get_systems(pdesys))

function reject_point_ports(pdesys, f)
    isempty(point_port_systems(pdesys)) && return
    throw(
        ArgumentError(
            "`$f` compiles the PDE system on its own, which would leave its PointPorts " *
                "unconnected and pin their flows to zero. Compose instead:\n" *
                "    sys, tspan = symbolic_discretize(pdesys, discretization)\n" *
                "    @named model = System([connect(sys.port, component.port), ...], t; systems = [sys, component, ...])\n" *
                "    prob = ODEProblem(mtkcompile(model), nothing, tspan)"
        )
    )
end
