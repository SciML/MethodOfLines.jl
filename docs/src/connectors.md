# Connector Ports

A `PointPort` turns the value of a PDE field at a point into an
[MTK connector](https://docs.sciml.ai/ModelingToolkit/stable/basics/Composition/), so a
discretized PDE composes with lumped components through `connect`. A heated rod can
exchange heat with a `HeatCapacitor`, or a probe can feed a field value to a controller.

Ports are declared on the `PDESystem`, before discretization, at coordinates rather than
grid indices. The discretization chooses the grid nodes.

## Declaring a port

```julia
PointPort(connector; name, var, at, across, through = nothing, select = Interpolate())
```

  - `connector` constructs the connector, called as `connector(; name, kwargs...)`. Any
    `@connector` works, for example `ModelingToolkitStandardLibrary.Thermal.HeatPort`.
  - `var` is the dependent variable the port couples to, such as `T(t, x)`.
  - `at` gives one coordinate per spatial argument of `var`, such as `[x => 0.3]`.
  - `across` names the potential variable of the connector, such as `:T`. After
    discretization it equals the selected field value.
  - `through = :Q_flow => scale` names a `connect = Flow` variable of the connector. The
    flow enters the PDE of `var` as a source term `scale * Q_flow`. Omit `through` for a
    read-out port, which only observes the field.

Put the ports in the `systems` of the `PDESystem`:

```julia
using ModelingToolkit, MethodOfLines, DomainSets, OrdinaryDiffEq
using ModelingToolkit: t_nounits as t, D_nounits as Dt
using ModelingToolkitStandardLibrary.Thermal

@parameters x k ρ c A
@variables T(..)
Dx = Differential(x)
Dxx = Differential(x)^2

@named heater = PointPort(HeatPort; var = T(t, x), at = [x => 0.31], across = :T,
    through = :Q_flow => 1 / (ρ * c * A))
@named probe = PointPort(HeatPort; var = T(t, x), at = [x => 0.7], across = :T,
    select = Nearest())

@named rod = PDESystem([Dt(T(t, x)) ~ k / (ρ * c) * Dxx(T(t, x))],
    [T(0, x) ~ 300.0, Dx(T(t, 0)) ~ 0, Dx(T(t, 1)) ~ 0],
    [t ∈ Interval(0.0, 10.0), x ∈ Interval(0.0, 1.0)], [t, x], [T(t, x)],
    [k => 50.0, ρ => 8000.0, c => 500.0, A => 1e-4]; systems = [heater, probe])
```

## Composing and solving

`discretize` refuses a system with ports, since compiling the rod alone would leave the
ports unconnected. Discretize symbolically, connect, then compile the composed model:

```julia
sys, tspan = symbolic_discretize(rod, MOLFiniteDifference([x => 0.01], t))

@named cap = HeatCapacitor(C = 10.0, T = 400.0)
@named contact = ThermalConductor(G = 1.0)
@named sensor = TemperatureSensor()
@named model = System(
    [connect(sys.heater, contact.port_a), connect(contact.port_b, cap.port),
        connect(sys.probe, sensor.port)], t;
    systems = [sys, contact, cap, sensor])

prob = ODEProblem(mtkcompile(model), nothing, tspan)
sol = solve(prob, FBDF())
sol[sys.heater.Q_flow]  # heat flow into the rod
sol[sys.T]              # the discretized field, one vector per time step
```

The initial condition of the PDE travels with the discretized system, so `ODEProblem`
needs no `u0` for it.

!!! note "Requires SciML/ModelingToolkit.jl#5133"
    Placing the discretized system inside another system namespaces its array (slice)
    equations, which released ModelingToolkitBase cannot do yet. Until
    [ModelingToolkit.jl#5133](https://github.com/SciML/ModelingToolkit.jl/pull/5133) is
    released, `mtkcompile(model)` above fails with

    ```
    matching non-exhaustive
    ```

    raised from `renamespace`. MethodOfLines keeps the array form rather than scalarizing.
    Keeping the discretized system at top level and adding the components with `extend`
    does not help: the system's own ports are then outer connectors, and MTK sets the flow
    of a top-level outer connector to zero.

Read-out ports work today on the discretized system compiled on its own. Its ports are
then outer connectors, so their flows are zero, and their variables are read from the
underlying ODE solution with the port objects you declared:

```julia
sol = solve(ODEProblem(mtkcompile(sys), nothing, tspan), FBDF())
sol.original_sol[probe.T]
```

## Point selection

The `select` keyword decides which grid nodes `I` and weights `w_I` represent the point.
The weights sum to one and form a tensor product across spatial directions.

  - `Interpolate(p)` uses Lagrange interpolation of degree `p` over `p + 1` nodes per
    direction, centred on the point where the grid allows. It reproduces polynomials of
    degree `p`. The default `Interpolate()` takes `p` from the `approx_order` of the
    discretization. A point that lies on a node uses that node alone.
  - `Nearest()` uses the single nearest node per direction. Ties go to the lower index.

The across variable is `Σ w_I u_I`. The flow is spread over the same nodes, with source
density `scale * Q_flow * w_I / V_I` at node `I`, where `V_I` is the node's control volume
(half a cell at the ends of a grid). One weight table serves both directions, so the
coupling conserves the exchanged quantity, `Σ V_I q_I = scale * Q_flow`, and is passive:
the power the PDE receives is `Q_flow` times the value the component sees.

## Sign and scale

`Q_flow` is positive into the PDE. This is the MTK convention for a component's own port,
and it matches `ModelingToolkitStandardLibrary` components such as `HeatCapacitor`.

`scale` converts the flow into the units of the PDE's source term. A rod with
cross-section `A`, density `ρ` and heat capacity `c`, whose state is temperature, takes
`1 / (ρ * c * A)`. A PDE for a conserved density per unit length takes `1`. `scale` may
contain parameters of the `PDESystem`.

## Limitations

  - A port that carries a flow selects interior nodes only, because boundary nodes hold
    boundary conditions. A point closer to a boundary than the first interior node is an
    error. Write flow through a boundary as a flux boundary condition instead.
  - Ports on boundary faces, ports spanning a whole face, and ports embedded in boundary
    conditions are not supported.
  - Stationary (time-independent) systems, `StaggeredGrid` and
    `PseudospectralDiscretization` do not support ports.
  - The composed model must go through `mtkcompile` and `ODEProblem`. A `DAEProblem` of
    the composed model fails initialization.
  - The `PDESolution` interface (`sol[T(t, x)]`) does not cover composed models. Index the
    namespaced unknowns instead, such as `sol[sys.T]`.
  - Composing a discretized system with lumped components needs
    SciML/ModelingToolkit.jl#5133, see above.
