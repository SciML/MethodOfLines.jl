# Minimal thermal connector and lumped components, mirroring the ModelingToolkitStandardLibrary
# conventions: `Q_flow` is a Flow variable, positive into the component that owns the port.
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as Dt

@connector function HeatPort(; name, T_guess = 300.0, Q_flow_guess = 0.0)
    vars = @variables begin
        T(t), [guess = T_guess]
        Q_flow(t), [guess = Q_flow_guess, connect = Flow]
    end
    return System(Equation[], t, vars, []; name)
end

@component function HeatCapacitor(; name, C = 10.0, T0 = 400.0)
    @named port = HeatPort()
    pars = @parameters C = C
    vars = @variables T(t) = T0
    eqs = [T ~ port.T, Dt(T) ~ port.Q_flow / C]
    return System(eqs, t, vars, pars; systems = [port], name)
end

@component function ThermalConductor(; name, G = 1.0)
    @named port_a = HeatPort()
    @named port_b = HeatPort()
    pars = @parameters G = G
    vars = @variables Q_flow(t)
    eqs = [
        Q_flow ~ G * (port_a.T - port_b.T),
        port_a.Q_flow ~ Q_flow,
        port_b.Q_flow ~ -Q_flow,
    ]
    return System(eqs, t, vars, pars; systems = [port_a, port_b], name)
end

# Whether ModelingToolkitBase can namespace the slice-form equations of a discretized system
# nested in another system: SciML/ModelingToolkit.jl#5133. Without it `mtkcompile` of the
# enclosing system fails with "matching non-exhaustive" in `renamespace`.
function renamespace_supports_slices()
    @parameters τ
    @variables p(τ)[1:3]
    D = Differential(τ)
    inner = System([D(p[1:2]) ~ p[2:3] .+ 1], τ; name = :inner)
    outer = System(Equation[], τ; systems = [inner], name = :outer)
    return try
        equations(outer)
        true
    catch
        false
    end
end
