using ModelingToolkit, Symbolics

# Variables, parameters, and derivatives
@parameters t x Δx u[1:3]
@variables a(..) v(..)
Dx = Differential(x)

g(x) = x^2 *(u[3] - 2u[2] - u[1])/(2*Δx^2) + x *(-u[3] + 4*u[2] - 3*u[1])/(2*Δx) + u[1]
gprime(x) = Symbolics.derivative(g(x), x)

fdisc = simplify(substitute(gprime(x), x => substitute(a(x)*gprime(x), x => Δx)))