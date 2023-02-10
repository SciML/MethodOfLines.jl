using OrdinaryDiffEq, ModelingToolkit, MethodOfLines, DomainSets
using Symbolics: @register_symbolic
 
@parameters x t
@variables ρ(..) v(..)
Dt = Differential(t)
Dx = Differential(x)
 
GM = 4e20
cs = 2.87e5
cs2 = 8.24e10
 
eq = [
        Dt(ρ(x,t)) ~ -exp(-x) * (2*ρ(x,t)*v(x,t) + v(x,t)*Dx(ρ(x,t)) + ρ(x,t)*Dx(v(x,t))),
        Dt(v(x,t)) ~ -v(x,t)*exp(-x) * Dx(v(x,t)) - cs2/(ρ(x,t)) * exp(-x) * Dx(ρ(x,t)) - GM*exp(-2*x)
    ]
 
x_min, x_max = 20.2, 21.8
t_min, t_max = 0.0, 30_000.0
 
domains = [x ∈ Interval(x_min, x_max), t ∈ Interval(t_min, t_max)]
 
ρ_init(x) = 5e-15 * exp(4.84e9 * exp(-x))
function v_init(x)
    if (x > 21.6)
        return 2 * cs
    else
        return 0.0
    end
end
 
@register_symbolic v_init(x)
 
bcs = [
    ρ(x,t_min) ~ ρ_init(x),
    ρ(x_min,t) ~ ρ_init(x_min),
    v(x,t_min) ~ v_init(x),
    v(x_min,t) ~ v_init(x_min),
    Dx(v(x_min,t)) ~ 0.0,
    Dx(ρ(x_min,t)) ~ 0.0,
    Dx(v(x_max,t)) ~ 0.0,
    Dx(ρ(x_max,t)) ~ 0.0,
    #Dt(v(x,t_min)) ~ 0.0,
    #Dt(ρ(x,t_min)) ~ 0.0,
] 
 
@named pdesys = PDESystem(eq, bcs, domains, [x,t], [ρ(x,t),v(x,t)])
 
N = 20
dx = (x_max-x_min)/N
order = 2
discretization = MOLFiniteDifference([x=>dx], t, advection_scheme=WENOScheme())
 
prob = discretize(pdesys, discretization)
 
sol = solve(prob, ImplicitEuler())