# Linear Complex PDEs

MethodOfLines can solve many linear complex PDEs like Telegraph equation, Schrödinger equation, Dirac equation, et cetera.

## Examples

### 1. Schrödinger Equation
```@example schro
using MethodOfLines, OrdinaryDiffEq, Plots, DomainSets, ModelingToolkit

@parameters t, x
@variables ψ(..)

Dt = Differential(t)
Dxx = Differential(x)^2

xmin = 0
xmax = 1

V(x) = 0.0

eq = [im * Dt(ψ(t, x)) ~ Dxx(ψ(t, x)) + V(x) * ψ(t, x)] # You must enclose complex equations in a vector, even if there is only one equation

ψ0 = x -> ((1 + im)/sqrt(2))*sinpi(2*x)

bcs = [ψ(0, x) => ψ0(x), # Initial condition must be marked with a => operator
    ψ(t, xmin) ~ 0,
    ψ(t, xmax) ~ 0]

domains = [t ∈ Interval(0, 1), x ∈ Interval(xmin, xmax)]

@named sys = PDESystem(eq, bcs, domains, [t, x], [ψ(t, x)])

disc = MOLFiniteDifference([x => 100], t)

prob = discretize(sys, disc)

sol = solve(prob, TRBDF2(), saveat = 0.01)

discx = sol[x]
disct = sol[t]

discψ = sol[ψ(t, x)]
anim = @animate for i in 1:length(disct)
    u = discψ[i, :]
    plot(discx, [real.(u), imag.(u)], ylim = (-1.5, 1.5), title = "t = $(disct[i])",
        xlabel = "x", ylabel = "ψ(t,x)", label = ["re(ψ)" "im(ψ)"], legend = :topleft)
end
gif(anim, "schroedinger.gif", fps = 10)
```

Note that complex initial conditions are supported, but must be marked with a `=>` operator.

This represents the second from ground state of a particle in an infinite quantum well. Try changing the potential `V(x)`, initial conditions and boundary conditions to see how extremely interesting the wave function evolves even for nonphysical combinations. Be sure to post interesting results on the discourse!


### 2. Dirac Equation
```@example dirac

# Dirac Equation in 1+1 Dimensions
using MethodOfLines, OrdinaryDiffEq, DomainSets, ModelingToolkit, Plots

@parameters t, x
@variables u1(..) u2(..)  # Components of the Dirac spinor

Dt = Differential(t)
Dx = Differential(x)

m = 1.0  # Mass term

# Dirac equation in 1+1 dimensions (natural units: c = ħ = 1)
eqs = [
    Dt(u1(t, x)) ~ -Dx(u2(t, x)) - m * u1(t, x),
    Dt(u2(t, x)) ~ Dx(u1(t, x)) - m * u2(t, x)
]

# Initial conditions
u1_0(x) = exp(-10 * (x - 0.5)^2)
u2_0(x) = 0.0

xmin, xmax = 0, 1
bcs = [
    u1(0, x) ~ u1_0(x),
    u2(0, x) ~ u2_0(x),
    u1(t, xmin) ~ 0,
    u1(t, xmax) ~ 0,
    u2(t, xmin) ~ 0,
    u2(t, xmax) ~ 0
]

domains = [t ∈ Interval(0, 1), x ∈ Interval(xmin, xmax)]

@named sys = PDESystem(eqs, bcs, domains, [t, x], [u1(t, x), u2(t, x)])

disc = MOLFiniteDifference([x => 100], t)
prob = discretize(sys, disc)
sol = solve(prob, TRBDF2(), saveat=0.01)

discx = sol[x]
disct = sol[t]
disc_u1 = sol[u1(t, x)]
disc_u2 = sol[u2(t, x)]

anim = @animate for i in 1:length(disct)
    plot(discx, [disc_u1[i, :], disc_u2[i, :]], ylim=(-1, 1), title="t = $(disct[i])",
        xlabel="x", ylabel="Ψ(t,x)", label=["u1 = Re(Ψ)" "u2 = Im(Ψ)"], legend=:topleft)
end
gif(anim, "dirac_equation.gif", fps=10)
```
This shows the time evolution of the two components (`u_1` and `u_2`) of a Dirac spinor in 1+1 dimensions. By solving the Dirac equation with specific initial conditions and boundary conditions, we observe how the particle's wavefunction evolves over time. Experimenting with different initial conditions or mass values can reveal interesting behaviors of relativistic particles.

Note that the Dirac equation can be viewed as a relativistic extension of Schrödinger equation.
