using ModelingToolkit, MethodOfLines, LinearAlgebra, OrdinaryDiffEq
using DomainSets

using Plots

local sol
@time begin #@testset "Test 01: Brusselator equation 2D" begin
       @parameters x y t
       @variables u(..) v(..)
       Dt = Differential(t)
       Dx = Differential(x)
       Dy = Differential(y)
       Dxx = Differential(x)^2
       Dyy = Differential(y)^2

       brusselator_f(x, y, t) = (((x-0.3)^2 + (y-0.6)^2) <= 0.1^2) * (t >= 1.1) * 5.

       x_min = y_min = t_min = 0.0
       x_max = y_max = 1.0
       t_max = 1.0

       α = 10.

       u0(x,y,t) = 22(y*(1-y))^(3/2)
       v0(x,y,t) = 27(x*(1-x))^(3/2)

       eq = [Dt(u(x,y,t)) ~ 1. + v(x,y,t)*u(x,y,t)^2 - 4.4*u(x,y,t) + α*(Dxx(u(x,y,t)) + Dyy(u(x,y,t))) + brusselator_f(x, y, t),
             Dt(v(x,y,t)) ~ 3.4*u(x,y,t) - v(x,y,t)*u(x,y,t)^2 + α*(Dxx(u(x,y,t)) + Dyy(u(x,y,t)))]

       domains = [x ∈ Interval(x_min, x_max),
                  y ∈ Interval(y_min, y_max),
                  t ∈ Interval(t_min, t_max)]

       bcs = [u(x,y,0) ~ u0(x,y,0),
              u(0,y,t) ~ u(1,y,t),
              u(x,0,t) ~ u(x,1,t),

              v(x,y,0) ~ v0(x,y,0),
              v(0,y,t) ~ v(1,y,t),
              v(x,0,t) ~ v(x,1,t)] 
       
       @named pdesys = PDESystem(eq,bcs,domains,[x,y,t],[u(x,y,t),v(x,y,t)])

       # Method of lines discretization
       dx = 1/32
       dy = 1/32

       order = 2

       discretization = MOLFiniteDifference([x=>dx, y=>dy], t, approx_order=order)

       # Convert the PDE problem into an ODE problem
       @time prob = discretize(pdesys,discretization)

       @time sol = solve(prob, TRBDF2(),saveat=0.01)

       Nx = floor(Int64, (x_max - x_min) / dx) + 1
       Ny = floor(Int64, (y_max - y_min) / dy) + 1

       #  @variables u[1:Nx,1:Ny](t)
       #  @variables v[1:Nx,1:Ny](t)
       #  t = sol[t]
       #   anim = @animate for k in 1:length(t)
       #          solu = real.(reshape([sol[u[(i-1)*Ny+j]][k] for i in 1:Nx for j in 1:Ny],(Nx,Ny)))
       #          solv = real.(reshape([sol[v[(i-1)*Ny+j]][k] for i in 1:Nx for j in 1:Ny],(Nx,Ny)))
       #          heatmap(solu[2:end, 2:end], title="$(t[k])")
       #   end
       #   gif(anim, "plots/Brusselator2Dsol.gif", fps = 5)


    #    solu′ = reshape([sol[u[(i-1)*Ny+j]][end] for i in 1:Nx for j in 1:Ny],(Nx,Ny))
    #    solv′ = reshape([sol[v[(i-1)*Ny+j]][end] for i in 1:Nx for j in 1:Ny],(Nx,Ny))

    #    r_space_x = x_min:dx:x_max
    #    r_space_y = y_min:dy:y_max

    #    asfu = reshape([u_exact(t_max,r_space_x[i],r_space_y[j]) for j in 1:Ny for i in 1:Nx],(Nx,Ny))
    #    asfv = reshape([v_exact(t_max,r_space_x[i],r_space_y[j]) for j in 1:Ny for i in 1:Nx],(Nx,Ny))

    #    asfu[1,1] = asfu[1, end] = asfu[end, 1] = asfu[end, end] = 0.
    #    asfv[1,1] = asfv[1, end] = asfv[end, 1] = asfv[end, end] = 0.

    #    # anim = @animate for T in t
    #    #        asfu = reshape([u_exact(T,r_space_x[i],r_space_y[j]) for j in 1:Ny for i in 1:Nx],(Nx,Ny))
    #    #        asfv = reshape([v_exact(T,r_space_x[i],r_space_y[j]) for j in 1:Ny for i in 1:Nx],(Nx,Ny))
              
    #    #        heatmap(asfu)
    #    # end
    #    # gif(anim, "plots/Burgers2Dexact.gif", fps = 5)

   
    #    mu = max(asfu...)
    #    mv = max(asfv...)
    #    @test_broken asfu / mu ≈ solu′ / mu  atol=0.2 
    #    @test_broken asfv / mv ≈ solv′ / mv  atol=0.2 
end