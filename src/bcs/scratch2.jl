using ModelingToolkit, MethodOfLines

@parameters t x
@variables u(..)
Dx = Differential(x)
Dt = Differential(t)
t_min= 0.
t_max = 2.
x_min = 0.
x_max = 2.
c = 1.0
a = 1.0

# Equation
eq = Dt(u(t,x)) ~ Dx(u(t,x) * Dx(u(t,x)))

terms = MethodOfLines.split_additive_terms(eq)

r = @rule ($(Differential(x))(u(t,x) * $(Differential(x))(u(t,x)))) => 0

@show r(terms[2])