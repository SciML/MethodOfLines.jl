@setup_workload begin
    @compile_workload begin
        if false # Temporarily disabled during compat update
            begin
                @parameters x y t
                @variables u(..) v(..)
                Dt = Differential(t)
                Dx = Differential(x)
                Dy = Differential(y)
                Dxx = Differential(x)^2
                Dyy = Differential(y)^2

                ∇²(u) = Dxx(u) + Dyy(u)

                brusselator_f(x, y, t) = (((x - 0.3)^2 + (y - 0.6)^2) <= 0.1^2) * (t >= 1.1) *
                    5.0

                x_min = y_min = t_min = 0.0
                x_max = y_max = 1.0
                t_max = 5.0

                α = 10.0

                u0(x, y, t) = 22(y * (1 - y))^(3 / 2)
                v0(x, y, t) = 27(x * (1 - x))^(3 / 2)

                eq = [
                    Dt(u(x, y, t)) ~
                        1.0 + v(x, y, t) * u(x, y, t)^2 - 4.4 * u(x, y, t) +
                        α * ∇²(u(x, y, t)) + brusselator_f(x, y, t) +
                        Dx(u(x, y, t)) + Dy(v(x, y, t))
                    Dt(v(x, y, t)) ~
                        3.4 * u(x, y, t) - v(x, y, t) * u(x, y, t)^2 +
                        α * ∇²(v(x, y, t)) + Dx(v(x, y, t) * Dx(u(x, y, t))) +
                        Dy(v(x, y, t) * Dy(u(x, y, t)))
                ]

                domains = [
                    x ∈ Interval(x_min, x_max),
                    y ∈ Interval(y_min, y_max),
                    t ∈ Interval(t_min, t_max),
                ]

                # Periodic BCs
                bcs = [
                    u(x, y, 0) ~ u0(x, y, 0),
                    u(0, y, t) ~ u(1, y, t),
                    u(x, 0, t) ~ u(x, 1, t), v(x, y, 0) ~ v0(x, y, 0),
                    v(0, y, t) ~ v(1, y, t),
                    v(x, 0, t) ~ v(x, 1, t),
                ]

                @named pdesys = PDESystem(eq, bcs, domains, [x, y, t], [u(x, y, t), v(x, y, t)])

                N = 6

                order = 2

                discretization = MOLFiniteDifference(
                    [x => N, y => N], t, approx_order = order, grid_align = center_align
                )

                # Convert the PDE problem into an ODE problem
                prob = discretize(pdesys, discretization)
            end
            begin
                @parameters x t
                @variables u(..)
                Dx = Differential(x)
                Dt = Differential(t)
                x_min = 0.0
                x_max = 1.0
                t_min = 0.0
                t_max = 6.0

                analytic_u2(t, x) = x / (t + 1)

                eq = Dt(u(t, x)) ~ -u(t, x) * Dx(u(t, x))

                bcs = [
                    u(0, x) ~ x,
                    u(t, x_min) ~ analytic_u2(t, x_min),
                    u(t, x_max) ~ analytic_u2(t, x_max),
                ]

                domains = [
                    t ∈ Interval(t_min, t_max),
                    x ∈ Interval(x_min, x_max),
                ]

                dx = 6

                @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

                disc = MOLFiniteDifference(
                    [x => dx], t, advection_scheme = WENOScheme(), grid_align = edge_align
                )

                prob = discretize(pdesys, disc)
            end
        end
    end # if false
end
