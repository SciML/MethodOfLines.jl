@parameters t ξ η
@variables ρ(..) ϕ(..)
Dt = Differential(t);
Dξ = Differential(ξ);
Dη = Differential(η);

a = 0.1;
L = 3.0;
dx = 0.5;

eq = [Dt(ρ(t,ξ)) + Dx(ϕ(t,η)) ~ 0,
      Dt(ϕ(t,η)) + a^2 * Dx(ρ(t,ξ)) ~ 0]
bcs = [ρ(0.0,ξ) ~ exp(-(ξ-L/2)^2),
       ϕ(0.0,η) ~ 0.0,
       ρ(t,0) ~ ρ(t,L-dx),
       ϕ(t,dx) ~ ϕ(t,L)];


domains = [t in Interval(0.0, 10.0),
           ξ in Interval(0.0, L-dx),
           η in Interval(dx, L)];

@named pdesys = PDESystem(eq, bcs, domains, [t,ξ,η], [ρ(t,ξ), ϕ(t,η)]);
