# Steady state of SIS (suspected-infected-suspected) reaction-diffusion  model

Considering the following SIS reaction diffusion model:

```math
\left\{\begin{array}{l}
S_{t} = d_{S} S_{x x}-\beta(x) \frac{S I}{S+I}+\gamma(x) I=0, \quad 0<x<1 \\
I_{t} = d_{I} I_{x x}+\beta(x) \frac{S I}{S+I}-\gamma(x) I=0, \quad 0<x<1 \\
S_{x}=I_{x}=0, \quad x=0,1,
\end{array}\right.
```

where ``\int_{0}^{1} S(x,t)+I(x,t)dx = 1``. ``S(x,t)`` and ``I(x,t)``  denote the density of susceptible and  infected  populations at location ``x`` and time ``t``,  ``d_{S}`` and ``d_{I}`` represent the  diffusion coefficients for susceptible and infected  individuals, and  ``\beta(x)``, ``\gamma(x)`` are transmission  and recovery rates at ``x``, respectively.

We want to solve the steady state problem (same notations for convenience):

```math
\left\{\begin{array}{l}
d_{S} S_{x x}-\beta(x) \frac{S I}{S+I}+\gamma(x) I=0, \quad 0<x<1 \\
d_{I} I_{x x}+\beta(x) \frac{S I}{S+I}-\gamma(x) I=0, \quad 0<x<1 \\
S_{x}=I_{x}=0, \quad x=0,1,
\end{array}\right.
```

where ``\int_{0}^{1} S(x)+I(x)dx = 1``.

Note here elliptic problem has condition ``\int_{0}^{1} S(x)+I(x)dx = 1``.

```@example sispde
using DifferentialEquations, ModelingToolkit, MethodOfLines, DomainSets, Plots

# Parameters, variables, and derivatives
@parameters t x
@parameters dS dI brn ϵ
@variables S(..) I(..)
Dt = Differential(t)
Dx = Differential(x)
Dxx = Differential(x)^2

# Define functions
function γ(x)
    y = x + 1.0
    return y
end

function ratio(x, brn, ϵ)
    y = brn + ϵ * sin(2 * pi * x)
    return y
end

# 1D PDE and boundary conditions
eq = [
    Dt(S(t, x)) ~
    dS * Dxx(S(t, x)) -
    ratio(x, brn, ϵ) * γ(x) * S(t, x) * I(t, x) / (S(t, x) + I(t, x)) +
    γ(x) * I(t, x),
    Dt(I(t, x)) ~
    dI * Dxx(I(t, x)) +
    ratio(x, brn, ϵ) * γ(x) * S(t, x) * I(t, x) / (S(t, x) + I(t, x)) -
    γ(x) * I(t, x)]
bcs = [S(0, x) ~ 0.9 + 0.1 * sin(2 * pi * x),
    I(0, x) ~ 0.1 + 0.1 * cos(2 * pi * x),
    Dx(S(t, 0)) ~ 0.0,
    Dx(S(t, 1)) ~ 0.0,
    Dx(I(t, 0)) ~ 0.0,
    Dx(I(t, 1)) ~ 0.0]

# Space and time domains
domains = [t ∈ Interval(0.0, 10.0),
    x ∈ Interval(0.0, 1.0)]

# PDE system
@named pdesys = PDESystem(eq, bcs, domains, [t, x], [S(t, x), I(t, x)], [dS, dI, brn, ϵ];
    defaults = Dict(dS => 0.5, dI => 0.1, brn => 3, ϵ => 0.1))

# Method of lines discretization
# Need a small dx here for accuracy
dx = 0.01
order = 2
discretization = MOLFiniteDifference([x => dx], t)

# Convert the PDE problem into an ODE problem
prob = discretize(pdesys, discretization);
```

### Solving time-dependent SIS epidemic model

```@example sispde
# Solving SIS reaction diffusion model
sol = solve(prob, Tsit5(), saveat = 0.2);

# Retrieving the results
discrete_x = sol[x]
discrete_t = sol[t]
S_solution = sol[S(t, x)]
I_solution = sol[I(t, x)]

p = surface(discrete_x, discrete_t, S_solution)
display(p)
```

### Solving steady state problem

Change the elliptic problem to steady state problem of reaction diffusion equation.

See more solvers in [Steady State Solvers · DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/solvers/steady_state_solve/)

```@example sispde
steadystateprob = SteadyStateProblem(prob)
steadystate = solve(steadystateprob, DynamicSS(Tsit5()))
```

### The effect of human mobility on endemic size

Set the endemic size
$$f(d_{S},d_{I}) = \int_{0}^{1}I(x;d_{S},d_{I}).$$

```@example sispde
function episize!(dS, dI)
    newprob = remake(prob, p = [dS, dI, 3, 0.1])
    steadystateprob = SteadyStateProblem(newprob)
    sol = solve(steadystateprob, DynamicSS(Tsit5()))
    if sol === nothing || !SciMLBase.successful_retcode(sol)
        @warn "Failed to find steady state for dS=$dS, dI=$dI"
        return NaN
    end
    state = sol.u
    y = sum(state[100:end]) / 99
    return y
end
episize!(exp(1.0), exp(0.5))
```

References:

  - Allen L J S, Bolker B M, Lou Y, et al. Asymptotic profiles of the steady states for an SIS epidemic reaction-diffusion model[J]. Discrete & Continuous Dynamical Systems, 2008, 21(1): 1.
