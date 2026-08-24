using ModelingToolkit, MethodOfLines, DomainSets
include(joinpath(@__DIR__, "..", "shared", "ode_discretize.jl"))

@testset "MTK metadata should be passed to discretized variables" begin
    # Parameters, variables, and derivatives
    @parameters t x
    @variables u(..) [input = true] g(..) [irreducible = true] y(..) z(..) [bounds = (0.0, Inf), output = true]
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    # 1D PDE and boundary conditions
    eq = [Dt(u(t, x)) ~ Dxx(u(t, x)) + g(t, x) + y(t, x) + z(t, x),
        Dt(g(t, x)) ~ 0,
        Dt(y(t, x)) ~ 0,
        Dt(z(t, x)) ~ 0]
    bcs = [u(0, x) ~ sin(pi * x),
        u(t, 0) ~ 0,
        u(t, 1) ~ 0,
        g(0, x) ~ 0,
        Dx(g(t, 0)) ~ 0,
        Dx(g(t, 1)) ~ 0,
        y(0, x) ~ 0,
        Dx(y(t, 0)) ~ 0,
        Dx(y(t, 1)) ~ 0,
        z(0, x) ~ 0,
        Dx(z(t, 0)) ~ 0,
        Dx(z(t, 1)) ~ 0]

    # Space and time domains
    domains = [t ∈ Interval(0.0, 0.01),
        x ∈ Interval(0.0, 1.0)]

    # PDE system
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [g(t, x), u(t, x), y(t, x), z(t, x)])

    # Dependent variables have metadata
    @test hasmetadata(pdesys.u, ModelingToolkit.VariableInput)
    @test hasmetadata(pdesys.g, ModelingToolkit.VariableIrreducible)
    @test hasmetadata(pdesys.z, ModelingToolkit.VariableBounds)
    @test hasmetadata(pdesys.z, ModelingToolkit.VariableOutput)
    @test getmetadata(pdesys.u, ModelingToolkit.VariableInput) == true
    @test getmetadata(pdesys.g, ModelingToolkit.VariableIrreducible) == true
    @test getmetadata(pdesys.z, ModelingToolkit.VariableBounds) == (0.0, Inf)
    @test getmetadata(pdesys.z, ModelingToolkit.VariableOutput) == true

    dx = 0.25
    # Method of lines discretization
    discretization = MOLFiniteDifference([x => dx], t)

    # Convert the PDE problem into an ODE problem
    prob = ode_discretize(pdesys, discretization)

    @test hasmetadata(prob.f.sys.g, ModelingToolkit.VariableIrreducible)
    @test hasmetadata(prob.f.sys.u, ModelingToolkit.VariableInput)
    @test hasmetadata(prob.f.sys.z, ModelingToolkit.VariableBounds)
    @test hasmetadata(prob.f.sys.z, ModelingToolkit.VariableOutput)
    @test getmetadata(prob.f.sys.g, ModelingToolkit.VariableIrreducible) == true
    @test getmetadata(prob.f.sys.u, ModelingToolkit.VariableInput) == true
    @test getmetadata(prob.f.sys.z, ModelingToolkit.VariableBounds) == (0.0, Inf)
    @test getmetadata(prob.f.sys.z, ModelingToolkit.VariableOutput) == true

    @test length(analytically_integrated(prob)) == 6 # 3 for y and 3 for z
end
