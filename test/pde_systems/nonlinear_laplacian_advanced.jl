using ModelingToolkit, MethodOfLines, LinearAlgebra, Test, OrdinaryDiffEq, DomainSets

function halfar_dome(t, x, y, R0, H0, ρ, A=1e-16)
    n = 3.0
    grav = 9.8101
    alpha = 1.0/9.0
    beta = 1.0/18.0

    Gamma = 2.0/(n+2.0) * A * (ρ * grav)^n
  
    xcenter = 0.0
    ycenter = 0.0
  
    t0 = (beta/Gamma) * (7.0/4.0)^3 * (R0^4/H0^7)  # Note: this line assumes n=3!
    tr=(t+t0)/t0 
  
    H=zeros(length(y), length(x))
    for i in eachindex(x)
      for j in eachindex(y)
        r = sqrt((x[i]-xcenter)^2 + (y[j]-ycenter)^2)
        r=r/R0
        inside = max(0.0, 1.0 - (r / tr^beta)^((n+1.0) / n))
  
        H[i,j] = H0 * inside^(n / (2.0*n+1.0)) / tr^alpha
        end
    end
    return H
  
end
@testset "Halfar ice dome glacier model." begin

    rmax = 2*1000
    rmin = -rmax

    H0 = 100
    R0 = 1000

    asf = (t, x, y) -> halfar_dome(t, x, y, R0, H0, 917)

    @parameters x, y, t

    @variables H(..) inHx(..) inHy(..)

    Dx = Differential(x)
    Dy = Differential(y)
    Dt = Differential(t)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    n = 3.0
    grav = 9.8101
    A = 1e-16
    ρ = 910.0

    Γ = 2.0/(n+2.0) * A * (ρ * grav)^n

    eqs = [Dt(H(t,x,y)) ~ Dx(Γ*H(t, x, y)^(n+2)*(abs(Dx(H(t, x, y)))^(n-1))*Dx(H(t, x, y))) 
                        + Dy(Γ*H(t, x, y)^(n+2)*(abs(Dy(H(t, x, y)))^(n-1))*Dy(H(t, x, y)))]

    bcs = [H(0.0, x, y) ~ asf(0.0, x, y),
        H(t, xmin, y) ~ 0.0,
        H(t, xmax, y) ~ 0.0,
        H(t, x, ymin) ~ 0.0,
        H(t, x, ymax) ~ 0.0]

    domains = [x ∈ IntervalDomain(rmin, rmax),
            y ∈ IntervalDomain(rmin, rmax),
            t ∈ IntervalDomain(0.0, 1.0e6)]

    @named pdesys = PDESystem(eqs, bcs, domains, [x, y, t], [H(t, x, y)])
    
    disc = MOLFiniteDifference([x => 24, y => 24], t)

    prob = discretize(pdesys, disc)

    sol = solve(prob, FBDF())

    @test sol.retcode == :Success

    solx = sol.x
    soly = sol.y
    solt = sol.t

    solexact = [asf(dt, dx, dy) for dt in solt, dx in solx, dy in soly]

    @test sum(abs2, sol[H(t, x, y)] .- solexact) < 1e-2

end
