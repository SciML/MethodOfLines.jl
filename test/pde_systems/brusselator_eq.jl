using ModelingToolkit, MethodOfLines, LinearAlgebra, OrdinaryDiffEq
using DomainSets

# using Plots

# local sol
@testset "Test 01: Brusselator equation 2D" begin
       @parameters x y t
       @variables u(..) v(..)
       Dt = Differential(t)
       Dx = Differential(x)
       Dy = Differential(y)
       Dxx = Differential(x)^2
       Dyy = Differential(y)^2

       ∇²(u) = Dxx(u) + Dyy(u)

       brusselator_f(x, y, t) = (((x-0.3)^2 + (y-0.6)^2) <= 0.1^2) * (t >= 1.1) * 5.

       x_min = y_min = t_min = 0.0
       x_max = y_max = 1.0
       t_max = 11.5

       α = 10.

       u0(x,y,t) = 22(y*(1-y))^(3/2)
       v0(x,y,t) = 27(x*(1-x))^(3/2)

       eq = [Dt(u(x,y,t)) ~ 1. + v(x,y,t)*u(x,y,t)^2 - 4.4*u(x,y,t) + α*∇²(u(x,y,t)) + brusselator_f(x, y, t),
             Dt(v(x,y,t)) ~ 3.4*u(x,y,t) - v(x,y,t)*u(x,y,t)^2 + α*∇²(v(x,y,t))]

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

       #MethodOfLines.generate_code(pdesys, discretization)
       # Convert the PDE problem into an ODE problem
       @time prob = discretize(pdesys,discretization)

       @time sol = solve(prob, TRBDF2(),saveat=0.01)

       Nx = floor(Int64, (x_max - x_min) / dx) + 1
       Ny = floor(Int64, (y_max - y_min) / dy) + 1

       @variables u[1:Nx,1:Ny](t)
       @variables v[1:Nx,1:Ny](t)
       
       # Solve reference problem
       
       const N = 31
       const xyd_brusselator = range(0,stop=1,length=N)
       brusselator_f(x, y, t) = (((x-0.3)^2 + (y-0.6)^2) <= 0.1^2) * (t >= 1.1) * 5.
       limit(a, N) = a == N+1 ? 1 : a == 0 ? N : a
       function brusselator_2d_loop(du, u, p, t)
              A, B, alpha, dx = p
              alpha = alpha/dx^2
              @inbounds for I in CartesianIndices((N, N))
                     i, j = Tuple(I)
           x, y = xyd_brusselator[I[1]], xyd_brusselator[I[2]]
           ip1, im1, jp1, jm1 = limit(i+1, N), limit(i-1, N), limit(j+1, N), limit(j-1, N)
           du[i,j,1] = alpha*(u[im1,j,1] + u[ip1,j,1] + u[i,jp1,1] + u[i,jm1,1] - 4u[i,j,1]) +
           B + u[i,j,1]^2*u[i,j,2] - (A + 1)*u[i,j,1] + brusselator_f(x, y, t)
           du[i,j,2] = alpha*(u[im1,j,2] + u[ip1,j,2] + u[i,jp1,2] + u[i,jm1,2] - 4u[i,j,2]) +
                       A*u[i,j,1] - u[i,j,1]^2*u[i,j,2]
           end
       end
       p = (3.4, 1., 10., step(xyd_brusselator))
       
       function init_brusselator_2d(xyd)
           N = length(xyd)
           u = zeros(N, N, 2)
           for I in CartesianIndices((N, N))
               x = xyd[I[1]]
               y = xyd[I[2]]
               u[I,1] = 22*(y*(1-y))^(3/2)
               u[I,2] = 27*(x*(1-x))^(3/2)
           end
           u
       end
       u0_manual = init_brusselator_2d(xyd_brusselator)
       prob = ODEProblem(brusselator_2d_loop,u0_manual,(0.,11.5),p)
       
       msol = solve(prob,TRBDF2(),saveat=0.01) # 2.771 s (5452 allocations: 65.73 MiB)
       
       t = sol[t]
       for k in 1:length(t)
              @test msol.u[k][:,:,1] ≈ reshape([sol[u[(i-1)*Ny+j]][k] for i in 1:Nx for j in 1:Ny],(Nx,Ny))[2:end,2:end] rtol = 0.1
              msolv = msol.u[k][:,:,2] ≈ reshape([sol[v[(i-1)*Ny+j]][k] for i in 1:Nx for j in 1:Ny],(Nx,Ny))[2:end,2:end] rtol = 0.1
       end
   
       
       # Nx = floor(Int64, (x_max - x_min) / dx) + 1
       # Ny = floor(Int64, (y_max - y_min) / dy) + 1

       #  @variables u[1:Nx,1:Ny](t)
       #  @variables v[1:Nx,1:Ny](t)
       #  t = sol[t]
       #   anim = @animate for k in 1:length(t)
       #          solu = reshape([sol[u[(i-1)*Ny+j]][k] for i in 1:Nx for j in 1:Ny],(Nx,Ny))
       #          #solv = reshape([sol[v[(i-1)*Ny+j]][k] for i in 1:Nx for j in 1:Ny],(Nx,Ny))
       #          heatmap(solu[2:end, 2:end], title="$(t[k])")
       #   end
       #   gif(anim, "plots/Brusselator2Dsol.gif", fps = 8)
end