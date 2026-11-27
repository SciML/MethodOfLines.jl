@setup_workload begin
    # Both entry points users hit: `discretize`, and the symbolic path through `mtkcompile`.
    function precompile_discretize(pdesys, disc)
        discretize(pdesys, disc)
        sys, tspan = symbolic_discretize(pdesys, disc)
        return ODEProblem(mtkcompile(sys), nothing, tspan)
    end
    @compile_workload begin
        # 1D diffusion, Dirichlet left and Neumann right, centered scheme.
        let
            @parameters t x
            @variables u(..)
            Dt = Differential(t)
            Dx = Differential(x)
            Dxx = Differential(x)^2
            eq = Dt(u(t, x)) ~ Dxx(u(t, x))
            bcs = [u(0, x) ~ sin(pi * x), u(t, 0.0) ~ 0.0, Dx(u(t, 1.0)) ~ 0.0]
            domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0)]
            @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
            precompile_discretize(pdesys, MOLFiniteDifference([x => 0.1], t))
        end
        # 1D advection-diffusion, periodic, upwind and WENO schemes. Periodic boundaries are
        # interfaces, which make `should_transform` warn, so it is switched off here.
        for scheme in (UpwindScheme(), WENOScheme())
            @parameters t x
            @variables u(..)
            Dt = Differential(t)
            Dx = Differential(x)
            Dxx = Differential(x)^2
            eq = Dt(u(t, x)) ~ -u(t, x) * Dx(u(t, x)) + 0.01 * Dxx(u(t, x))
            bcs = [u(0, x) ~ 1.0 + 0.25 * sin(pi * x), u(t, 0.0) ~ u(t, 2.0)]
            domains = [t ∈ Interval(0.0, 0.5), x ∈ Interval(0.0, 2.0)]
            @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
            precompile_discretize(pdesys, MOLFiniteDifference([x => 0.1], t; advection_scheme = scheme, should_transform = false))
        end
        # 2D reaction-diffusion with two variables, periodic.
        let
            @parameters x y t
            @variables u(..) v(..)
            Dt = Differential(t)
            Dxx = Differential(x)^2
            Dyy = Differential(y)^2
            eq = [
                Dt(u(x, y, t)) ~ 1.0 + v(x, y, t) * u(x, y, t)^2 - 4.4 * u(x, y, t) +
                    10.0 * (Dxx(u(x, y, t)) + Dyy(u(x, y, t))),
                Dt(v(x, y, t)) ~ 3.4 * u(x, y, t) - v(x, y, t) * u(x, y, t)^2 +
                    10.0 * (Dxx(v(x, y, t)) + Dyy(v(x, y, t))),
            ]
            domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0), t ∈ Interval(0.0, 1.0)]
            bcs = [
                u(x, y, 0) ~ 22 * (y * (1 - y))^(3 / 2), v(x, y, 0) ~ 27 * (x * (1 - x))^(3 / 2),
                u(0, y, t) ~ u(1, y, t), u(x, 0, t) ~ u(x, 1, t),
                v(0, y, t) ~ v(1, y, t), v(x, 0, t) ~ v(x, 1, t),
            ]
            @named pdesys = PDESystem(eq, bcs, domains, [x, y, t], [u(x, y, t), v(x, y, t)])
            precompile_discretize(pdesys, MOLFiniteDifference([x => 0.2, y => 0.2], t; approx_order = 2, should_transform = false))
        end
        # 3D diffusion, Dirichlet.
        let
            @parameters t x y z
            @variables u(..)
            Dt = Differential(t)
            Dxx = Differential(x)^2
            Dyy = Differential(y)^2
            Dzz = Differential(z)^2
            eq = Dt(u(t, x, y, z)) ~ Dxx(u(t, x, y, z)) + Dyy(u(t, x, y, z)) + Dzz(u(t, x, y, z))
            bcs = [
                u(0, x, y, z) ~ sin(pi * x) * sin(pi * y) * sin(pi * z),
                u(t, 0, y, z) ~ 0, u(t, 1, y, z) ~ 0, u(t, x, 0, z) ~ 0,
                u(t, x, 1, z) ~ 0, u(t, x, y, 0) ~ 0, u(t, x, y, 1) ~ 0,
            ]
            domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0), z ∈ Interval(0.0, 1.0)]
            @named pdesys = PDESystem(eq, bcs, domains, [t, x, y, z], [u(t, x, y, z)])
            precompile_discretize(pdesys, MOLFiniteDifference([x => 0.25, y => 0.25, z => 0.25], t))
        end
        # 2D third-order mixed derivative, Dirichlet.
        let
            @parameters t x y
            @variables u(..)
            Dt = Differential(t)
            eq = [Dt(u(t, x, y)) ~ (Differential(x)^2 * Differential(y))(u(t, x, y))]
            bcs = [
                u(0, x, y) ~ sin(pi * (x + y)),
                u(t, 0, y) ~ sin(pi * y), u(t, 1, y) ~ sin(pi * (1 + y)),
                u(t, x, 0) ~ sin(pi * x), u(t, x, 1) ~ sin(pi * (x + 1)),
            ]
            domains = [t ∈ Interval(0.0, 0.1), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]
            @named pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])
            precompile_discretize(pdesys, MOLFiniteDifference([x => 0.2, y => 0.2], t))
        end
        # 2D reaction-diffusion with two variables, Neumann.
        let
            @parameters t x y
            @variables u(..) v(..)
            Dt = Differential(t)
            Dx = Differential(x)
            Dy = Differential(y)
            Dxx = Differential(x)^2
            Dyy = Differential(y)^2
            eq = [
                Dt(u(t, x, y)) ~ 2.0e-5 * (Dxx(u(t, x, y)) + Dyy(u(t, x, y))) - u(t, x, y) * v(t, x, y)^2 + 0.04 * (1 - u(t, x, y)),
                Dt(v(t, x, y)) ~ 1.0e-5 * (Dxx(v(t, x, y)) + Dyy(v(t, x, y))) + u(t, x, y) * v(t, x, y)^2 - 0.1 * v(t, x, y),
            ]
            bcs = [
                u(0, x, y) ~ 1 - exp(-100 * ((x - 0.5)^2 + (y - 0.5)^2)), v(0, x, y) ~ 0.25 * exp(-100 * ((x - 0.5)^2 + (y - 0.5)^2)),
                Dx(u(t, 0, y)) ~ 0, Dx(u(t, 1, y)) ~ 0, Dy(u(t, x, 0)) ~ 0, Dy(u(t, x, 1)) ~ 0,
                Dx(v(t, 0, y)) ~ 0, Dx(v(t, 1, y)) ~ 0, Dy(v(t, x, 0)) ~ 0, Dy(v(t, x, 1)) ~ 0,
            ]
            domains = [t ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]
            @named pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y), v(t, x, y)])
            precompile_discretize(pdesys, MOLFiniteDifference([x => 0.2, y => 0.2], t))
        end
    end
end
