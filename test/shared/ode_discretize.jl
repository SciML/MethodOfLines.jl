function ode_discretize(pdesys, disc)
    sys, tspan = symbolic_discretize(pdesys, disc)
    return ODEProblem(mtkcompile(sys), nothing, tspan)
end
