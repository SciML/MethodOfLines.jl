include("MOL_original.jl")

@testset "Regression Test Linear 2D" begin
        # Variables, parameters, and derivatives
        @parameters t x y
        @variables u(..)
        Dxx = Differential(x)^2
        Dyy = Differential(y)^2
        Dt = Differential(t)
        t_min= 0.
        t_max = 2.0
        x_min = 0.
        x_max = 2.
        y_min = 0.
        y_max = 2.
        dx = 0.1; dy = 0.2
        order = 2
    
        # Analytic solution
        analytic_sol_func(t,x,y) = exp(x+y)*cos(x+y+4t)
    
        # Equation
        eq  = Dt(u(t,x,y)) ~ Dxx(u(t,x,y)) + Dyy(u(t,x,y))
    
        # Initial and boundary conditions
        bcs = [u(t_min,x,y) ~ analytic_sol_func(t_min,x,y),
                u(t,x_min,y) ~ analytic_sol_func(t,x_min,y),
                u(t,x_max,y) ~ analytic_sol_func(t,x_max,y),
                u(t,x,y_min) ~ analytic_sol_func(t,x,y_min),
                u(t,x,y_max) ~ analytic_sol_func(t,x,y_max)]
    
        # Space and time domains
        domains = [t ∈ Interval(t_min,t_max),
                x ∈ Interval(x_min,x_max),
                y ∈ Interval(y_min,y_max)]
    
        # Space and time domains
        @named pdesys = PDESystem([eq],bcs,domains,[t,x,y],[u(t,x,y)])

        # Method of lines discretization
        discretization = MOLFiniteDifference([x=>dx,y=>dy],t;centered_order=order)
        prob = ModelingToolkit.discretize(pdesys,discretization)

        # Test equivalance with original
        discretization_original = MOLFiniteDifference_origial([x=>dx,y=>dy],t;centered_order=order)
        prob_original = discretize_original(pdesys,discretization)
        
        @test isequal(prob, prob_original)
end