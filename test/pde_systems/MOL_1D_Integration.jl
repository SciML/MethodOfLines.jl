using MethodOfLines, ModelingToolkit, LinearAlgebra, Test, OrdinaryDiffEq, DomainSets

@testset "Test 00: Test simple integration case (0 .. x), no transformation" begin
    # test integrals
    @parameters t, x
    @variables integrand(..) cumuSum(..)
    Dt = Differential(t)
    Dx = Differential(x)
    xmin = 0.0
    xmax = 2.0 * pi

    Ix = Integral(x in DomainSets.ClosedInterval(xmin, x)) # basically cumulative sum from 0 to x

    eqs = [cumuSum(t, x) ~ Ix(integrand(t, x))
           integrand(t, x) ~ t * cos(x)]

    bcs = [cumuSum(0, x) ~ 0.0,
        integrand(0, x) ~ 0.0]

    domains = [t ∈ Interval(0.0, 1.0),
        x ∈ Interval(xmin, xmax)]

    @named pde_system = PDESystem(
        eqs, bcs, domains, [t, x], [integrand(t, x), cumuSum(t, x)])

    asf(t, x) = t * sin(x)

    disc = MOLFiniteDifference([x => 120], t)

    prob = discretize(pde_system, disc)

    sol = solve(prob, Tsit5())

    xdisc = sol[x]
    tdisc = sol[t]

    cumuSumsol = sol[cumuSum(t, x)]

    exact = [asf(t_, x_) for t_ in tdisc, x_ in xdisc]

    @test cumuSumsol≈exact atol=0.36
end

@testset "Test 00: Test simple integration case (0 .. x), with sys transformation" begin
    # test integrals
    @parameters t, x
    @variables integrand(..) cumuSum(..)
    Dt = Differential(t)
    Dx = Differential(x)
    xmin = 0.0
    xmax = 2.0 * pi

    Ix = Integral(x in DomainSets.ClosedInterval(xmin, x)) # basically cumulative sum from 0 to x

    eqs = [cumuSum(t, x) ~ Ix(t * cos(x))]

    bcs = [cumuSum(0, x) ~ 0.0]

    domains = [t ∈ Interval(0.0, 1.0),
        x ∈ Interval(xmin, xmax)]

    @named pde_system = PDESystem(
        eqs, bcs, domains, [t, x], [integrand(t, x), cumuSum(t, x)])

    asf(t, x) = t * sin(x)

    disc = MOLFiniteDifference([x => 120], t)

    @test_broken (discretize(pde_system, disc) isa ODEProblem)
    # prob = discretize(pde_system, disc)
    # sol = solve(prob, Tsit5())

    # xdisc = sol[x]
    # tdisc = sol[t]

    # cumuSumsol = sol[cumuSum(t, x)]

    # exact = [asf(t_, x_) for t_ in tdisc, x_ in xdisc]

    # @test cumuSumsol ≈ exact atol = 0.36
end

@testset "Test 01: Test integration over whole domain, (xmin .. xmax)" begin
    # test integrals
    @parameters t, x
    @variables integrand(..) cumuSum(..)
    Dt = Differential(t)
    Dx = Differential(x)
    xmin = 0.0
    xmax = 2.0 * pi

    Ix = Integral(x in DomainSets.ClosedInterval(xmin, xmax)) # integral over domain

    eqs = [cumuSum(t) ~ Ix(integrand(t, x))
           integrand(t, x) ~ t * cos(x)]

    bcs = [cumuSum(0) ~ 0.0,
        integrand(0, x) ~ 0.0]

    domains = [t ∈ Interval(0.0, 1.0),
        x ∈ Interval(xmin, xmax)]

    @named pde_system = PDESystem(eqs, bcs, domains, [t, x], [integrand(t, x), cumuSum(t)])

    asf(t) = 0.0

    disc = MOLFiniteDifference([x => 120], t)

    prob = discretize(pde_system, disc)

    sol = solve(prob, Tsit5())

    xdisc = sol[x]
    tdisc = sol[t]

    cumuSumsol = sol[cumuSum(t)]

    exact = [asf(t_) for t_ in tdisc]

    @test cumuSumsol≈exact atol=0.3
end

@testset "Test 02: Test integration with arbitrary limits, (a .. b)" begin
    # test integrals
    @parameters t, x
    @variables integrand(..) cumuSum(..)
    Dt = Differential(t)
    Dx = Differential(x)
    xmin = 0.0
    xmax = 2.0 * pi

    Ix = Integral(x in DomainSets.ClosedInterval(0.5, 3.0)) # integral over interval

    eqs = [cumuSum(t) ~ Ix(integrand(t, x))
           integrand(t, x) ~ t * cos(x)]

    bcs = [cumuSum(0) ~ 0.0,
        integrand(0, x) ~ 0.0]

    domains = [t ∈ Interval(0.0, 1.0),
        x ∈ Interval(xmin, xmax)]

    @named pde_system = PDESystem(eqs, bcs, domains, [t, x], [integrand(t, x), cumuSum(t)])

    asf(t, x) = t * sin(x)

    disc = MOLFiniteDifference([x => 120], t)

    @test_broken (discretize(pde_system, disc) isa ODEProblem)
    # prob = discretize(pde_system, disc)
    # sol = solve(prob, Tsit5(), saveat=0.1)

    # xdisc = sol[x]
    # tdisc = sol[t]

    # cumuSumsol = sol[cumuSum(t)]

    # exact = [asf(t_, 3.0) - asf(t_, 0.5) for t_ in tdisc]

    # @test cumuSumsol ≈ exact atol = 0.36
end

@testset "Test 03: Partial integro-differential equation with time derivative, with sys transformation" begin
    # Equation: ∂u/∂t + u(t,x) + ∫₀ˣ u(t,ξ) dξ = x + x²/2 * (1 - exp(-t - τ))   with τ = 0.2 (alternatively τ = 0)
    # On domain: x ∈ [0,2] and t ∈ [0,1]
    # Initial condition: u(0,x) = x * (1 - exp(-τ))
    # Boundary condition: u(t,0) = 0
    # Analytical solution: u(t,x) = x * (1 - exp(-t - τ))
    
    @parameters t, x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    xmin = 0.0
    xmax = 2.0

    Ix_0_to_x = Integral(x in DomainSets.ClosedInterval(xmin, x)) # basically cumulative sum from 0 to x

    eqs = [Dt(u(t,x)) + u(t,x) + Ix_0_to_x(u(t,x)) ~ x + x^2/2*(1-exp(-t - 0.2))]
    bcs = [
        u(0,x) ~ x * (1 - exp(-0.2)),   # Initial condition
        u(t,0) ~ 0.0]                   # Boundary condition
    domains = [
        t ∈ Interval(0.0, 1.0),
        x ∈ Interval(xmin, xmax)]
    @named pde_system = PDESystem(eqs, bcs, domains, [t, x], [u(t, x)])

    # Discretize and compute numerical solution
    disc = MOLFiniteDifference([x => 50], t)
    prob = discretize(pde_system, disc)
    sol = solve(prob, Tsit5(), saveat=0.1)

    # Extract numerical solution at grid points
    xdisc = sol[x]
    tdisc = sol[t]
    usol = sol[u(t,x)]

    # Calculate analytical solution at grid points
    analytical_solution(t, x) = x * (1 - exp(-t - 0.2) )
    exact = [analytical_solution(t_, x_) for t_ in tdisc, x_ in xdisc]

    # Compare numerical and analytical solutions
    @test usol ≈ exact atol=0.1

    # # Create animated plot
    # using Plots  # Add this import
    # anim = @animate for (i, t_val) in enumerate(tdisc)
    #     plot(xdisc, usol[i, :], 
    #          label="Numerical", 
    #          linewidth=2,
    #          xlabel="x", 
    #          ylabel="u(t,x)", 
    #          title="Solution at t = $(round(t_val, digits=3))",
    #          ylim=(0, maximum(usol)*1.1),
    #          xlim=(xmin, xmax))
        
    #     plot!(xdisc, exact[i, :], 
    #           label="Analytical", 
    #           linewidth=2, 
    #           linestyle=:dash)

    # end
    # gif(anim, "integro_diff_solution.gif", fps=10) # Save the animation
end

@testset "Test 03b: Mirrored version of: Partial integro-differential equation with time derivative, with sys transformation" begin
    # Original equation: ∂u/∂t + u(t,x) + ∫₀ˣ u(t,ξ) dξ = x + x²/2 * (1 - exp(-t - τ))   with τ = 0
    #   Mirrored version with transformation y = -x, x = -y:
    # Equation:          ∂u/∂t + u(t,y) + ∫_y ^⁰ u(t,ξ) dξ = -y + y²/2 * (1 - exp(-t))
    # On domain:         y ∈ [-2, 0] and t ∈ [0,1]
    # Initial condition: u(0,y) = 0
    # Boundary condition: u(t,0) = 0
    # Analytical solution: u(t,y) = -y * (1 - exp(-t))
    
    @parameters t, y
    @variables u(..)
    Dt = Differential(t)
    Dy = Differential(y)
    ymin = -2.0
    ymax = 0.0

    # Integral from y to 0 (note the reversed limits)
    Iy_to_0 = Integral(y in DomainSets.ClosedInterval(y, ymax))

    eqs = [Dt(u(t,y)) + u(t,y) + Iy_to_0(u(t,y)) ~ -y + y^2/2*(1-exp(-t))]
    bcs = [
        u(0,y) ~ 0.0,   # Initial condition
        u(t,0) ~ 0.0]   # Boundary condition (at y = 0, which corresponds to x = 0)
    domains = [
        t ∈ Interval(0.0, 1.0),
        y ∈ Interval(ymin, ymax)]
    @named pde_system = PDESystem(eqs, bcs, domains, [t, y], [u(t, y)])

    # Discretize and compute numerical solution
    disc = MOLFiniteDifference([y => 50], t)
    prob = discretize(pde_system, disc)
    sol = solve(prob, Tsit5(), saveat=0.1)

    # Extract numerical solution at grid points
    ydisc = sol[y]
    tdisc = sol[t]
    usol = sol[u(t,y)]

    # Calculate analytical solution at grid points
    # Since u(t,x) = x * (1 - exp(-t)) and y = -x, then u(t,y) = -y * (1 - exp(-t))
    analytical_solution(t, y) = -y * (1 - exp(-t))
    exact = [analytical_solution(t_, y_) for t_ in tdisc, y_ in ydisc]

    # Compare numerical and analytical solutions
    @test usol ≈ exact atol=0.1

    # # Create animated plot
    # using Plots  # Add this import
    # anim = @animate for (i, t_val) in enumerate(tdisc)
    #     plot(ydisc, usol[i, :], 
    #          label="Numerical", 
    #          linewidth=2,
    #          xlabel="y", 
    #          ylabel="u(t,y)", 
    #          title="Solution at t = $(round(t_val, digits=3))",
    #          ylim=(0, maximum(usol)*1.1),
    #          xlim=(ymin, ymax))
        
    #     plot!(ydisc, exact[i, :], 
    #           label="Analytical", 
    #           linewidth=2, 
    #           linestyle=:dash)
    # end
    # gif(anim, "integro_diff_solution.gif", fps=10) # Save the animation
end

@testset "Test 04: PIDE with spatial derivative and source term - domain [0, 4π]" begin
    # Equation:  ∂u/∂t = ∂u/∂x - u(t,x) - ∫₀ˣ u(t,ξ) dξ + h(x,t)
    #      where h(x,t) = -sin(x)sin(t) + sin(x)cos(t) - cos(x)cos(t) + cos(t)(1 - cos(x))
    # On domain: x ∈ [0,L]=[0,2π] and t ∈ [0,T]=[0,1]
    # Initial condition: u(0,x) = sin(x)
    # Boundary conditions: u(t,0) = 0, u(t,2π) = sin(2π)cos(t) = 0
    # Analytical solution: u(t,x) = sin(x)cos(t)

    @parameters t, x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    xmin = 0.0
    xmax = 4π

    I0_to_x = Integral(x in DomainSets.ClosedInterval(xmin, x)) # integral from 0 to x

    # Source term h(x,t) that admits the analytical solution
    h(x, t) = -sin(x)*sin(t) + sin(x)*cos(t) - cos(x)*cos(t) + cos(t)*(1 - cos(x))

    # PIDE: ∂u/∂t = ∂u/∂x - u(t,x) - ∫₀ˣ u(t,ξ) dξ + h(x,t)
    eqs = [Dt(u(t,x)) ~ Dx(u(t,x)) - u(t,x) - I0_to_x(u(t,x)) + h(x, t)]
    bcs = [
        u(0, x) ~ sin(x),              # Initial condition
        u(t, 0) ~ 0.0,                 # Boundary condition at x = 0
        u(t, xmax) ~ sin(xmax)*cos(t)  # Boundary condition at x = 4π (sin(4π) = 0)
        ]
    domains = [
        t ∈ Interval(0.0, 1.0),
        x ∈ Interval(xmin, xmax)]
    @named pde_system = PDESystem(eqs, bcs, domains, [t, x], [u(t, x)])

    # Discretize and compute numerical solution
    disc = MOLFiniteDifference([x => 200], t)
    prob = discretize(pde_system, disc)
    sol = solve(prob, Tsit5(), saveat=0.1)

    # Extract numerical solution at grid points
    xdisc = sol[x]
    tdisc = sol[t]
    usol  = sol[u(t,x)]

    # Calculate analytical solution at grid points
    analytical_solution(t, x) = sin(x) * cos(t)
    exact = [analytical_solution(t_, x_) for t_ in tdisc, x_ in xdisc]

    # Compare numerical and analytical solutions
    @test usol ≈ exact atol=0.36

    # # Create animated plot
    # using Plots  # Add this import
    # anim = @animate for (i, t_val) in enumerate(tdisc)
    #     plot(xdisc, usol[i, :], 
    #          label="Numerical", 
    #          linewidth=2,
    #          xlabel="x", 
    #          ylabel="u(t,x)", 
    #          title="PIDE Solution [0,4π] at t = $(round(t_val, digits=3))",
    #          ylim=(-1.1, 1.1),
    #          xlim=(xmin, xmax))
        
    #     plot!(xdisc, exact[i, :], 
    #           label="Analytical", 
    #           linewidth=2, 
    #           linestyle=:dash)
        
    # end
    # gif(anim, "pide_solution_4pi.gif", fps=10) # Save the animation
end


@testset "Test 04b: PIDE with spatial derivative and source term - mirrored domain [-4π, 0]" begin
    # Mirrored version with transformation y = -x, so x = -y
    # Original equation:    ∂u/∂t =  ∂u/∂x - u(t,x) - ∫₀ˣ u(t,ξ) dξ + h(x,t)
    # Transformed equation: ∂u/∂t = -∂u/∂y - u(t,y) - ∫ₓ⁰ u(t,η) dη + h(-y,t)
    # where h(-y,t) = -sin(-y)sin(t) + sin(-y)cos(t) - cos(-y)cos(t) + cos(t)(1 - cos(-y))
    #               = sin(y)sin(t) - sin(y)cos(t) - cos(y)cos(t) + cos(t)(1 - cos(y))
    # On domain: y ∈ [-4π, 0] and t ∈ [0,1]
    # Initial condition: u(0,y) = sin(-y) = -sin(y)
    # Boundary conditions: u(t,0) = 0, u(t,-4π) = sin(-(-4π))cos(t) = sin(4π)cos(t) = 0
    # Analytical solution: u(t,y) = sin(-y)cos(t) = -sin(y)cos(t)
    
    @parameters t, y
    @variables u(..)
    Dt = Differential(t)
    Dy = Differential(y)
    ymin = -4π
    ymax = 0.0

    # Integral from y to 0 (note the reversed limits)
    Iy_to_0 = Integral(y in DomainSets.ClosedInterval(y, ymax))

    # Source term h(-y,t) transformed for y coordinate
    h_transformed(y, t) = sin(y)*sin(t) - sin(y)*cos(t) - cos(y)*cos(t) + cos(t)*(1 - cos(y))

    # PIDE: ∂u/∂t = -∂u/∂y - u(t,y) - ∫ʸ⁰ u(t,η) dη + h(-y,t)
    eqs = [Dt(u(t,y)) ~ -Dy(u(t,y)) - u(t,y) - Iy_to_0(u(t,y)) + h_transformed(y, t)]
    bcs = [
        u(0, y) ~ -sin(y),             # Initial condition: u(0,y) = sin(-y) = -sin(y)
        u(t, 0) ~ 0.0,                 # Boundary condition at y = 0
        u(t, ymin) ~ -sin(ymin)*cos(t) # Boundary condition at y = -4π (sin(-4π) = 0)
        ]
    domains = [
        t ∈ Interval(0.0, 1.0),
        y ∈ Interval(ymin, ymax)]
    @named pde_system = PDESystem(eqs, bcs, domains, [t, y], [u(t, y)])

    # Discretize and compute numerical solution
    disc = MOLFiniteDifference([y => 200], t)
    prob = discretize(pde_system, disc)
    sol = solve(prob, Tsit5(), saveat=0.1)

    # Extract numerical solution at grid points
    ydisc = sol[y]
    tdisc = sol[t]
    usol  = sol[u(t,y)]

    # Calculate analytical solution at grid points
    # Analytical solution: u(t,y) = sin(-y)cos(t) = -sin(y)cos(t)
    analytical_solution(t, y) = -sin(y) * cos(t)
    exact = [analytical_solution(t_, y_) for t_ in tdisc, y_ in ydisc]

    # Compare numerical and analytical solutions
    @test usol ≈ exact atol=0.36

    # # Create animated plot
    # using Plots  # Add this import
    # anim = @animate for (i, t_val) in enumerate(tdisc)
    #     plot(ydisc, usol[i, :], 
    #          label="Numerical", 
    #          linewidth=2,
    #          xlabel="y", 
    #          ylabel="u(t,y)", 
    #          title="PIDE Solution [-4π,0] at t = $(round(t_val, digits=3))",
    #          ylim=(-1.1, 1.1),
    #          xlim=(ymin, ymax))
        
    #     plot!(ydisc, exact[i, :], 
    #           label="Analytical", 
    #           linewidth=2, 
    #           linestyle=:dash)
        
    # end
    # gif(anim, "pide_solution_mirrored_4pi.gif", fps=10) # Save the animation
end
