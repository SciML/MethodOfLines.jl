using ModelingToolkit, MethodOfLines, LinearAlgebra, OrdinaryDiffEq
using ModelingToolkit: operation, istree, arguments
using DomainSets
using NonlinearSolve

# # Define some variables
@testset "Heat Equation 1D 2 variables" begin
       @parameters t x
       @variables u(..) v(..)
       Dt = Differential(t)
       Dx = Differential(x)
       Dxx = Differential(x)^2
       eqs  = [Dt(u(t,x)) ~ Dxx(u(t,x)),
              Dt(v(t,x)) ~ Dxx(v(t,x))]
       bcs = [u(0,x) ~ - x * (x-1) * sin(x),
              v(0,x) ~ - x * (x-1) * sin(x),
              u(t,0) ~ 0.0, u(t,1) ~ 0.0,
              v(t,0) ~ 0.0, v(t,1) ~ 0.0]

       domains = [t ∈ Interval(0.0,1.0),
              x ∈ Interval(0.0,1.0)]

       @named pdesys = PDESystem(eqs,bcs,domains,[t,x],[u(t,x),v(t,x)])
       discretization = MOLFiniteDifference([x=>0.1],t;grid_align=edge_align)
       prob = discretize(pdesys,discretization) # This gives an ODEProblem since it's time-dependent
       sol = solve(prob,Tsit5())
end

@testset "Heat Equation 2D 1 variable" begin
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

       # 3D PDE
       eq  = Dt(u(t,x,y)) ~ Dxx(u(t,x,y)) + Dyy(u(t,x,y))

       analytic_sol_func(t,x,y) = exp(x+y)*cos(x+y+4t)
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
       @named pdesys = PDESystem([eq],bcs,domains,[t,x,y],[u(t,x,y)])
       # Method of lines discretization
       dx = 0.1; dy = 0.2
       discretization = MOLFiniteDifference([x=>dx,y=>dy],t)
       prob = ModelingToolkit.discretize(pdesys,discretization)
       sol = solve(prob,Tsit5())
end

# Diffusion in a sphere
@testset "Spherical Diffusion" begin

       @parameters t r
       @variables u(..)
       Dt = Differential(t)
       Dr = Differential(r)
       Drr = Dr^2
       eq  = Dt(u(t,r)) ~ 1/r^2 * Dr(r^2 * Dr(u(t,r)))
       bcs = [u(0,r) ~ - r * (r-1) * sin(r),
              Dr(u(t,0)) ~ 0.0, u(t,1) ~ sin(1)]

       domains = [t ∈ Interval(0.0,1.0),
              r ∈ Interval(0.0,1.0)]

       @named pdesys = PDESystem(eq,bcs,domains,[t,r],[u(t,r)])
       discretization = MOLFiniteDifference([r=>0.1],t)
       prob = discretize(pdesys,discretization) # This gives an ODEProblem since it's time-dependent
       sol = solve(prob,Tsit5())
end

@testset "RHS = 0" begin
       @parameters x, t
       @variables u(..)
       Dt = Differential(t)
       Dx = Differential(x)
       Dx2 = Differential(x)^2
       Dx3 = Differential(x)^3
       Dx4 = Differential(x)^4

       α = 1
       β = 4
       γ = 1
       eq = Dt(u(x,t)) + u(x,t)*Dx(u(x,t)) + α*Dx2(u(x,t)) + β*Dx3(u(x,t)) + γ*Dx4(u(x,t)) ~ 0

       du(x,t;z = -x/2+t) = 15/2*(tanh(z) + 1)*(3*tanh(z) - 1)*sech(z)^2

       bcs = [u(x,0) ~ x^2,
              Dx(u(-10,t)) ~ du(-10,t),
              Dx(u(10,t)) ~ du(10,t)]

       # Space and time domains
       domains = [x ∈ Interval(-10.0,10.0),
              t ∈ Interval(0.0,0.5)]
       # Discretization
       dx = 0.4; dt = 0.2

       discretization = MOLFiniteDifference([x=>dx],t;approx_order=4)
       @named pdesys = PDESystem(eq,bcs,domains,[x,t],[u(x,t)])
       prob = discretize(pdesys,discretization)
end

@testset "2D Neumann Hydrogen flame" begin

    # the two spatial dimensions and time
    @parameters x y t
    # the mass fractions MF (respectively H₂, O₂, and H₂O), temperature,
    # and the source terms of the mass fractions and temperature
    @variables YF(..) YO(..) YP(..) T(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dy = Differential(y)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    # laplace operator
    ∇²(u) = Dxx(u) + Dyy(u)

    ## Define domains

    xmin = ymin = 0.0 # cm
    xmax = 1.8 # cm
    ymax = 0.9 # cm
    tmin = 0.0
    tmax = 0.06 # s

    domains = [
        x ∈ Interval(xmin, xmax),
        y ∈ Interval(ymin, ymax),
        t ∈ Interval(tmin, tmax)
    ]

    ## Define parameters

    # diffusivity coefficient (for temperature and mass fractions)
    κ =  2.0 #* cm^2/s
    # constant (divergence-free) velocity field
    U = [50.0, 0.0] #.* cm/s
    # density of the mixture
    ρ = 1.39e-3 #* g/cm^3
    # molecular weights (respectively H₂, O₂, and H₂O)
    W = [2.016, 31.9, 18.0] #.* g/mol
    # stoichiometric coefficients
    ν = [2, 1, 2]
    # heat of the reaction
    Q = 9800 #* K
    # universal gas constant
    R = 8.314472 * 100 #* 1e-2 J/mol/K -> because J = Nm = 100 N cm
    # y coordinates of the inlet area on the left side of the domain
    inlety = (0.3, 0.6) # mm

    # TODO try out different values (systematically)
    # pre-exponential factor in source term
    A = 5.5e11 # dimensionless
    # activation energy in source term
    E = 5.5e13 * 100 # 1e-2 J/mol -> J = Nm = 100 N cm

    ## Define the model

    # diffusion and convection terms for the mass fractions and temperature
    eqs = [
        Dt( YF(x,y,t) ) ~ κ * ∇²( YF(x,y,t) ) - U[1] * Dx( YF(x,y,t) ) - U[2] * Dy( YF(x,y,t) )
        Dt( YO(x,y,t) ) ~ κ * ∇²( YO(x,y,t) ) - U[1] * Dx( YO(x,y,t) ) - U[2] * Dy( YO(x,y,t) )
        Dt( YP(x,y,t) ) ~ κ * ∇²( YP(x,y,t) ) - U[1] * Dx( YP(x,y,t) ) - U[2] * Dy( YP(x,y,t) )
        Dt( T(x,y,t) )  ~ κ * ∇²( T(x,y,t) )  - U[1] * Dx( T(x,y,t) )  - U[2] * Dy( T(x,y,t) )
    ]

    function atInlet(x,y,t)
        return (inlety[1] < y) * (y < inlety[2])
    end

    bcs = [
        #
        # initial conditions
        #

        T(x, y, 0) ~ 300.0, # K; around 26 C
        # domain is empty at the start
        YF(x,y,0) ~ 0.0,
        YO(x,y,0) ~ 0.0,
        YP(x,y,0) ~ 0.0,

        #
        # boundary conditions
        #

        # left side
        T(xmin,y,t) ~ atInlet(xmin,y,t) * 950 + (1-atInlet(xmin,y,t)) *  300, # K
        YF(xmin,y,t) ~ atInlet(xmin,y,t) * 0.0282, # mass fraction
        YO(xmin,y,t) ~ atInlet(xmin,y,t) * 0.2259,
        YP(xmin,y,t) ~ 0.0,

        # bottom
        Dt( T(x,ymin,t) ) ~ 0.0, # K/s
        Dt( YF(x,ymin,t) ) ~ 0.0, # mass fraction/s
        Dt( YO(x,ymin,t) ) ~ 0.0,
        Dt( YP(x,ymin,t) ) ~ 0.0,
        # right side
        Dt( T(xmax,y,t) ) ~ 0.0, # K/s
        Dt( YF(xmax,y,t) ) ~ 0.0, # mass fraction/s
        Dt( YO(xmax,y,t) ) ~ 0.0,
        Dt( YP(xmax,y,t) ) ~ 0.0,

        # top
        Dt( T(x,ymax,t) ) ~ 0.0, # K/s
        Dt( YF(x,ymax,t) ) ~ 0.0, # mass fraction/s
        Dt( YO(x,ymax,t) ) ~ 0.0,
        Dt( YP(x,ymax,t) ) ~ 0.0,

    ]

    @named pdesys = PDESystem(eqs, bcs, domains, [x,y,t],[YF(x,y,t), YO(x,y,t), YP(x,y,t), T(x,y,t)])
    ## Discretize the system

    N = 4
    dx = 1/N
    dy = 1/N
    order = 2

    discretization = MOLFiniteDifference(
        [x=>dx, y=>dy], t, approx_order=order
    )

    # this creates an ODEProblem or a NonlinearProblem, depending on the system
    problem = discretize(pdesys, discretization)

end

@testset "Wave Equation" begin
       # Parameters, variables, and derivatives
       @parameters t x
       @variables u(..)
       Dt = Differential(t)             # Required in ICs and equation
       Dtt = Differential(t)^2           # required in equation
       Dxxxx = Differential(x)^4       # required in equation
       Dxx = Differential(x)^2         # required in BCs
       # some parameters
       EI  = 291.6667;
       m = 1.3850;
       c = 1.;
       p = 0.01;
       L = 2.0;

       # 1D PDE and boundaru conditions
       eq  = m*Dtt(u(t,x)) + c*Dt(u(t,x)) + EI*Dxxxx(u(t,x)) ~0

       ic_bc = [u(0,x) ~ (p*x*(x^3 + L^3 -2*L*x^2)/(24*EI)), #for all 0 < u < L
              Dt(u(0,x)) ~ 0.,        # for all 0 < u < L
              u(t,0) ~ 0.,            # for all t > 0,, displacement zero at u=0
              u(t,2) ~ 0.,            # for all t > 0,, displacement zero at u=L
              Dxx(u(t,0)) ~ 0.,       # for all t > 0,, curvature zero at u=0
              Dxx(u(t,2)) ~ 0.]       # for all t > 0,, curvature zero at u=L

       # Space and time domains
       domains = [t ∈ Interval(0.0,2.0),
              x ∈ Interval(0.0, L)]

       dt = 0.1   # dt related to saving the data.. not actual dt
       # PDE sustem
       @named pdesys = PDESystem(eq,ic_bc,domains,[t,x],[u(t,x)])

       # Method of lines discretization
       dx = 0.1
       order = 2
       discretization = MOLFiniteDifference([x=>dx, t=>dt], approx_order=order)

       # Convert the PDE problem into an ODE problem
       prob = discretize(pdesys,discretization)

       # Solve the ODE problem
       sol = NonlinearSolve.solve(prob, NewtonRaphson())
end

@testset "Rearranged Robin" begin
       @parameters x t

	@variables c(..)

	∂t  = Differential(t)
	∂x  = Differential(x)
	∂²x = Differential(x) ^ 2

       D₀ = 1.
       R = 0.5
       cₑ = 2.5
       ℓ = 2.0
       α = 1/3

	diff_eq = ∂t(c(x, t)) ~ ∂x((D₀ + α * c(x, t)) * ∂x(c(x, t)))

	bcs = [c(x, 0) 	 ~ 0.0,  	       # initial condition
                  R * (D₀ + α * c(0, t)) * ∂x(c(0, t)) ~ c(0, t) - cₑ, # Robin
                  ∂x(c(ℓ, t)) ~ 0.0]   # no flux

       domains = [t ∈ Interval(0.0,10.0),
                  x ∈ Interval(0.0, ℓ)]

       @named pdesys = PDESystem(diff_eq,bcs,domains,[t,x],[c(x,t)])

       # Method of lines discretization
       dx = 0.1
       order = 2
       discretization = MOLFiniteDifference([x=>dx],t,approx_order=order)

       # Convert the PDE problem into an ODE problem
       prob = discretize(pdesys,discretization)

       sol = solve(prob,Rodas4())
end

@testset "Array u" begin
	# Dependencies
	N = 6 # number of dependent variables

	# Variables, parameters, and derivatives
	@parameters x
	@variables u[1:N](..)
	Dx = Differential(x)
	Dxx = Differential(x)^2

	# Domain edges
	x_min= 0.
	x_max = 1.

	# Discretization parameters
	dx = 0.1
	order = 2

       #u = collect(u)

	# Equations
	eqs  = Vector{ModelingToolkit.Equation}(undef, N)
	for i = 1:N
		eqs[i] = Dxx(u[i](x)) ~ u[i](x)
	end

	# Initial and boundary conditions
	bcs = Vector{ModelingToolkit.Equation}(undef, 2*N)
	for i = 1:N
		bcs[i] = Dx(u[i](x_min)) ~ 0.
	end

	for i = 1:N
		bcs[i+N] = u[i](x_max) ~ rand()
	end

	# Space and time domains
	domains = [x ∈ Interval(x_min, x_max)]

	# PDE system
	@named pdesys = PDESystem(eqs, bcs, domains, [x], collect([u[i](x) for i = 1:N]))

	# Method of lines discretization
	discretization = MOLFiniteDifference([x=>dx], nothing, approx_order=order)
	prob = ModelingToolkit.discretize(pdesys,discretization)

	# # Solution of the ODE system
	sol = NonlinearSolve.solve(prob,NewtonRaphson())
end

@testset "2D variable connected to 1D variable at boundary #33" begin
       @parameters t x r
       @variables u(..) v(..)
       Dt = Differential(t)
       Dx = Differential(x)
       Dxx = Differential(x)^2
       Dr = Differential(r)
       Drr = Differential(r)^2

       s = u(t,x) + v(t,x,1)

       eqs  = [Dt(u(t,x)) ~ Dxx(u(t,x)) + s,
              Dt(v(t,x,r)) ~ Drr(v(t,x,r))]
       bcs = [u(0,x) ~ 1,
       v(0,x,r) ~ 1,
       Dx(u(t,0)) ~ 0.0, Dx(u(t,1)) ~ 0.0,
       Dr(v(t,x,0)) ~ 0.0, Dr(v(t,x,1)) ~ s]

       domains = [t ∈ Interval(0.0,1.0),
                  x ∈ Interval(0.0,1.0),
                  r ∈ Interval(0.0,1.0)]

       @named pdesys = PDESystem(eqs,bcs,domains,[t,x,r],[u(t,x),v(t,x,r)])

       # Method of lines discretization
       dx = 0.1
       dr = 0.1
       order = 2
       discretization = MOLFiniteDifference([x=>dx,r=>dr], t, approx_order=order)

       # Convert the PDE problem into an ODE problem
       prob = discretize(pdesys,discretization)

       sol = solve(prob, Tsit5())
end




@testset "Testing discretization of varied systems" begin
	@parameters x t

	@variables c(..)

	∂t  = Differential(t)
	∂x  = Differential(x)
	∂²x = Differential(x) ^ 2

	D₀ = 1.5
	α = 0.15
	χ = 1.2
	R = 0.1
	cₑ = 2.0
	ℓ = 1.0
	Δx = 0.1

	bcs = [
              # initial condition
		c(x, 0) ~ 0.0,
		# Robin BC
		∂x(c(0.0, t)) / (1 + exp(α * (c(0.0, t) - χ))) * R * D₀ + cₑ - c(0.0, t) ~ 0.0,
		# no flux BC
		∂x(c(ℓ, t)) ~ 0.0]


       # define space-time plane
       domains = [x ∈ Interval(0.0, ℓ), t ∈ Interval(0.0, 5.0)]

       @testset "Test 01: ∂t(c(x, t)) ~ ∂x(D * ∂x(c(x, t)))" begin
              D = D₀ / (1.0 + exp(α * (c(x, t) - χ)))
              diff_eq = ∂t(c(x, t)) ~ ∂x(D * ∂x(c(x, t)))
              @named pdesys = PDESystem(diff_eq, bcs, domains, [x, t], [c(x, t)]);
              discretization = MOLFiniteDifference([x=>Δx], t)
       end

       @testset "Test 02: ∂t(c(x, t)) ~ ∂x(D * ∂x(c(x, t)))" begin
              D = 1.0 / (1.0 + exp(α * (c(x, t) - χ)))
              diff_eq = ∂t(c(x, t)) ~ ∂x(D * ∂x(c(x, t)))
              @named pdesys = PDESystem(diff_eq, bcs, domains, [x, t], [c(x, t)]);
              discretization = MOLFiniteDifference([x=>Δx], t)
       end

       @testset "Test 03: ∂t(c(x, t)) ~ ∂x(1.0 / (1.0/D₀ + exp(α * (c(x, t) - χ))/D₀) * ∂x(c(x, t)))" begin
              diff_eq = ∂t(c(x, t)) ~ ∂x(1.0 / (1.0/D₀ + exp(α * (c(x, t) - χ))/D₀) * ∂x(c(x, t)))
              @named pdesys = PDESystem(diff_eq, bcs, domains, [x, t], [c(x, t)]);
              discretization = MOLFiniteDifference([x=>Δx], t)
       end

       @testset "Test 04: ∂t(c(x, t)) ~ ∂x(D₀ / (1.0 + exp(α * (c(x, t) - χ))) * ∂x(c(x, t)))" begin
              diff_eq = ∂t(c(x, t)) ~ ∂x(D₀ / (1.0 + exp(α * (c(x, t) - χ))) * ∂x(c(x, t)))
              @named pdesys = PDESystem(diff_eq, bcs, domains, [x, t], [c(x, t)]);
              discretization = MOLFiniteDifference([x=>Δx], t)
       end

       @testset "Test 05: ∂t(c(x, t)) ~ ∂x(1/x * ∂x(c(x, t)))" begin
              diff_eq = ∂t(c(x, t)) ~ ∂x(1/x * ∂x(c(x, t)))
              @named pdesys = PDESystem(diff_eq, bcs, domains, [x, t], [c(x, t)]);
              discretization = MOLFiniteDifference([x=>Δx], t)
       end

       @testset "Test 06: ∂t(c(x, t)) ~ ∂x(x*∂x(c(x, t)))/c(x,t)" begin
              diff_eq = ∂t(c(x, t)) ~ ∂x(x*∂x(c(x, t)))/c(x,t)
              @named pdesys = PDESystem(diff_eq, bcs, domains, [x, t], [c(x, t)]);
              discretization = MOLFiniteDifference([x=>Δx], t)
       end

       @testset "Test 07: ∂t(c(x, t)) ~ ∂x(1/(1+c(x,t)) ∂x(c(x, t)))" begin
              diff_eq = ∂t(c(x, t)) ~ ∂x(1/(1+c(x,t)) * ∂x(c(x, t)))
              @named pdesys = PDESystem(diff_eq, bcs, domains, [x, t], [c(x, t)]);
              discretization = MOLFiniteDifference([x=>Δx], t)
       end

       @testset "Test 08: ∂t(c(x, t)) ~ c(x, t) * ∂x(c(x,t) * ∂x(c(x, t)))" begin
              diff_eq = ∂t(c(x, t)) ~ c(x, t)*∂x(c(x,t) * ∂x(c(x, t)))
              @named pdesys = PDESystem(diff_eq, bcs, domains, [x, t], [c(x, t)]);
              discretization = MOLFiniteDifference([x=>Δx], t)
       end

       @testset "Test 09: ∂t(c(x, t)) ~ c(x, t) * ∂x(c(x,t) * ∂x(c(x, t)))/(1+c(x,t))" begin
              diff_eq = c(x, t) * ∂x(c(x,t) * ∂x(c(x, t)))/(1+c(x,t))
              @named pdesys = PDESystem(diff_eq, bcs, domains, [x, t], [c(x, t)]);
              discretization = MOLFiniteDifference([x=>Δx], t)
       end

       @testset "Test 10: ∂t(c(x, t)) ~ c(x, t) * ∂x(c(x,t) * ∂x(c(x, t)))/(1+c(x,t))" begin
              diff_eq = c(x, t) * ∂x(c(x,t)^(-1) * ∂x(c(x, t)))
              @named pdesys = PDESystem(diff_eq, bcs, domains, [x, t], [c(x, t)]);
              discretization = MOLFiniteDifference([x=>Δx], t)
       end

       @testset "Test 11: ∂t(c(x, t)) ~ ∂x(1/(1+c(x,t)^2) ∂x(c(x, t)))" begin
              diff_eq = ∂t(c(x, t)) ~ ∂x(1/(1+c(x,t)^2) * ∂x(c(x, t)))
              @named pdesys = PDESystem(diff_eq, bcs, domains, [x, t], [c(x, t)]);
              discretization = MOLFiniteDifference([x=>Δx], t)
       end

end
