using MethodOfLines, ModelingToolkit, DomainSets, Test, PDEBase
using ModelingToolkit: get_bcs, get_eqs
using PDEBase: VariableMap, cardinalize_eqs!, parse_bcs, construct_discrete_space,
    d_orders, all_ivs

@parameters t x y x1 x2 vel
@variables u(..) v(..)

function run_boundary_validation(pdesys, disc)
    vmap = VariableMap(pdesys, disc)
    cardinalize_eqs!(pdesys)
    bcorders = Dict(iv => d_orders(iv, get_bcs(pdesys)) for iv in all_ivs(vmap))
    boundarymap = parse_bcs(get_bcs(pdesys), vmap, bcorders)
    s = construct_discrete_space(vmap, disc)
    MethodOfLines.validate_system_wellposedness(get_eqs(pdesys), boundarymap, s, disc)
    return nothing
end

function standard_1d_disc(;
        xspan = (0.0, 1.0),
        tspan = (0.0, 1.0),
        dx = 0.1,
        advection = nothing,
    )
    t0, t1 = tspan
    x0, x1 = xspan
    disc_kwargs = advection === nothing ? () : (; advection_scheme = advection)
    domains = [t ∈ Interval(t0, t1), x ∈ Interval(x0, x1)]
    disc = MOLFiniteDifference([x => dx], t; disc_kwargs...)
    return domains, disc
end

function build_1d_pdesys(eq, bcs; xspan = (0.0, 1.0), tspan = (0.0, 1.0))
    domains, _ = standard_1d_disc(; xspan, tspan)
    @named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
    return pdesys
end

function build_1d_pdesys_multi(eqs, bcs, depvars; xspan = (0.0, 1.0), tspan = (0.0, 1.0))
    domains, _ = standard_1d_disc(; xspan, tspan)
    @named pdesys = PDESystem(eqs, bcs, domains, [t, x], depvars)
    return pdesys
end

function standard_2d_disc(;
        xspan = (0.0, 1.0),
        yspan = (0.0, 1.0),
        tspan = (0.0, 1.0),
        dx = 0.1,
        dy = 0.1,
    )
    t0, t1 = tspan
    x0, x1b = xspan
    y0, y1b = yspan
    domains = [
        t ∈ Interval(t0, t1),
        x ∈ Interval(x0, x1b),
        y ∈ Interval(y0, y1b),
    ]
    disc = MOLFiniteDifference([x => dx, y => dy], t)
    return domains, disc
end

function build_2d_pdesys(eq, bcs; xspan = (0.0, 1.0), yspan = (0.0, 1.0), tspan = (0.0, 1.0))
    domains, _ = standard_2d_disc(; xspan, yspan, tspan)
    @named pdesys = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])
    return pdesys
end

@testset "Boundary condition well-posedness" begin

    @testset "1st-order static advection" begin
        Dt, Dx = Differential(t), Differential(x)
        eq = Dt(u(t, x)) ~ -Dx(u(t, x))
        _, disc = standard_1d_disc(; advection = UpwindScheme())

        @test run_boundary_validation(
            build_1d_pdesys(eq, [u(0, x) ~ 0.0, u(t, 0) ~ 0.0]), disc
        ) === nothing

        @test_throws ArgumentError run_boundary_validation(
            build_1d_pdesys(eq, [u(0, x) ~ 0.0]), disc
        )
    end

    @testset "1st-order dynamic advection" begin
        Dt, Dx = Differential(t), Differential(x)
        _, disc = standard_1d_disc(; advection = UpwindScheme())

        eq_burgers = Dt(u(t, x)) ~ -u(t, x) * Dx(u(t, x))
        @test run_boundary_validation(
            build_1d_pdesys(eq_burgers, [u(0, x) ~ 1.0, u(t, 0) ~ 1.0, u(t, 1) ~ 1.0]),
            disc,
        ) === nothing
        @test_throws ArgumentError run_boundary_validation(
            build_1d_pdesys(eq_burgers, [u(0, x) ~ 1.0, u(t, 0) ~ 1.0]), disc
        )

        eq_param = Dt(u(t, x)) ~ -vel * Dx(u(t, x))
        @test run_boundary_validation(
            build_1d_pdesys(eq_param, [u(0, x) ~ 0.0, u(t, 0) ~ 0.0, u(t, 1) ~ 0.0]), disc
        ) === nothing
        @test_throws ArgumentError run_boundary_validation(
            build_1d_pdesys(eq_param, [u(0, x) ~ 0.0, u(t, 0) ~ 0.0]), disc
        )
    end

    @testset "2nd-order spatial derivatives" begin
        Dt, Dxx = Differential(t), Differential(x)^2
        eq = Dt(u(t, x)) ~ Dxx(u(t, x))
        _, disc = standard_1d_disc()

        @test_throws ArgumentError run_boundary_validation(
            build_1d_pdesys(eq, [u(0, x) ~ 0.0, u(t, 0) ~ 0.0]), disc
        )
        @test run_boundary_validation(
            build_1d_pdesys(eq, [u(0, x) ~ 0.0, u(t, 0) ~ 0.0, u(t, 1) ~ 0.0]), disc
        ) === nothing
    end

    @testset "3rd-order spatial derivatives" begin
        Dt, Dx, Dx3 = Differential(t), Differential(x), Differential(x)^3
        eq = Dt(u(t, x)) ~ -Dx3(u(t, x))
        _, disc = standard_1d_disc()

        @test_throws ArgumentError run_boundary_validation(
            build_1d_pdesys(eq, [u(0, x) ~ 0.0, u(t, 0) ~ 0.0, u(t, 1) ~ 0.0]), disc
        )
        @test run_boundary_validation(
            build_1d_pdesys(
                eq, [
                    u(0, x) ~ 0.0,
                    u(t, 0) ~ 0.0,
                    u(t, 1) ~ 0.0,
                    Dx(u(t, 0)) ~ 0.0,
                ]
            ), disc
        ) === nothing
    end

    @testset "4th-order spatial derivatives" begin
        Dt, Dx, Dxxxx = Differential(t), Differential(x), Differential(x)^4
        eq = Dt(u(t, x)) ~ -Dxxxx(u(t, x))
        _, disc = standard_1d_disc()

        @test_throws ArgumentError run_boundary_validation(
            build_1d_pdesys(
                eq, [
                    u(0, x) ~ 0.0,
                    u(t, 0) ~ 0.0,
                    u(t, 1) ~ 0.0,
                    Dx(u(t, 0)) ~ 0.0,
                ]
            ), disc
        )
        @test run_boundary_validation(
            build_1d_pdesys(
                eq, [
                    u(0, x) ~ 0.0,
                    u(t, 0) ~ 0.0,
                    Dx(u(t, 0)) ~ 0.0,
                    u(t, 1) ~ 0.0,
                    Dx(u(t, 1)) ~ 0.0,
                ]
            ), disc
        ) === nothing
    end

    @testset "Periodic boundaries" begin
        Dt, Dx, Dxx, Dxxxx = Differential(t), Differential(x), Differential(x)^2,
            Differential(x)^4
        _, disc = standard_1d_disc()

        periodic_bcs = [u(0, x) ~ 0.0, u(t, 0) ~ u(t, 1)]
        @test run_boundary_validation(
            build_1d_pdesys(Dt(u(t, x)) ~ Dxx(u(t, x)), periodic_bcs), disc
        ) === nothing

        @test_throws ArgumentError run_boundary_validation(
            build_1d_pdesys(Dt(u(t, x)) ~ -Dxxxx(u(t, x)), periodic_bcs), disc
        )

        @test run_boundary_validation(
            build_1d_pdesys(
                Dt(u(t, x)) ~ -Dxxxx(u(t, x)), [
                    u(0, x) ~ 0.0,
                    u(t, 0) ~ u(t, 1),
                    Dx(u(t, 0)) ~ Dx(u(t, 1)),
                ]
            ), disc
        ) === nothing
    end

    @testset "Coupled systems" begin
        _, disc = standard_1d_disc()

        Dt_u, Dxx = Differential(t), Differential(x)^2
        eqs = [Dt_u(u(t, x)) ~ Dxx(u(t, x)), Dt_u(v(t, x)) ~ Dxx(v(t, x))]
        @test_throws ArgumentError run_boundary_validation(
            build_1d_pdesys_multi(eqs, [u(0, x) ~ 0.0, v(0, x) ~ 0.0, u(t, 0) ~ v(t, 1)], [u(t, x), v(t, x)]),
            disc,
        )

        Dx1, Dxx1 = Differential(x1), Differential(x1)^2
        Dx2, Dxx2 = Differential(x2), Differential(x2)^2
        interface_eqs = [
            Dt_u(u(t, x1)) ~ Dxx1(u(t, x1)),
            Dt_u(v(t, x2)) ~ Dxx2(v(t, x2)),
        ]
        interface_bcs = [
            u(0, x1) ~ 1.0,
            v(0, x2) ~ 0.0,
            u(t, 0.0) ~ 1.0,
            v(t, 2.0) ~ 0.0,
            u(t, 1.0) ~ v(t, 1.0),
            Dx1(u(t, 1.0)) ~ Dx2(v(t, 1.0)),
        ]
        interface_domains = [
            t ∈ Interval(0.0, 1.0),
            x1 ∈ Interval(0.0, 1.0),
            x2 ∈ Interval(1.0, 2.0),
        ]
        @named interface_pdesys = PDESystem(
            interface_eqs, interface_bcs, interface_domains, [t, x1, x2],
            [u(t, x1), v(t, x2)],
        )
        interface_disc = MOLFiniteDifference([x1 => 0.1, x2 => 0.1], t)
        @test run_boundary_validation(interface_pdesys, interface_disc) === nothing
    end

    @testset "Multidimensional and mixed derivatives" begin
        Dt = Differential(t)
        Dxx, Dyy = Differential(x)^2, Differential(y)^2
        Dxxy = Differential(x)^2 * Differential(y)
        _, disc = standard_2d_disc()

        eq_heat = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y))
        ic = u(0, x, y) ~ 0.0
        bx0 = u(t, 0, y) ~ 0.0
        bx1 = u(t, 1, y) ~ 0.0
        by0 = u(t, x, 0) ~ 0.0
        by1 = u(t, x, 1) ~ 0.0

        @test_throws ArgumentError run_boundary_validation(
            build_2d_pdesys(eq_heat, [ic, bx0, by0, by1]), disc
        )
        @test_throws ArgumentError run_boundary_validation(
            build_2d_pdesys(eq_heat, [ic, bx0, bx1, by0]), disc
        )
        @test run_boundary_validation(
            build_2d_pdesys(eq_heat, [ic, bx0, bx1, by0, by1]), disc
        ) === nothing

        eq_mixed = Dt(u(t, x, y)) ~ Dxxy(u(t, x, y))
        @test_throws ArgumentError run_boundary_validation(
            build_2d_pdesys(eq_mixed, [ic, bx0, by0]), disc
        )
        @test run_boundary_validation(
            build_2d_pdesys(eq_mixed, [ic, bx0, bx1, by0]), disc
        ) === nothing
    end

    @testset "Time-dependent boundaries" begin
        Dt, Dxx = Differential(t), Differential(x)^2
        eq = Dt(u(t, x)) ~ Dxx(u(t, x))
        _, disc = standard_1d_disc()

        @test run_boundary_validation(
            build_1d_pdesys(
                eq, [
                    u(0, x) ~ 0.0,
                    u(t, 0) ~ sin(t) * exp(-t),
                    u(t, 1) ~ 0.0,
                ]
            ), disc
        ) === nothing
    end

    @testset "Nested nonlinear advection" begin
        Dt, Dx = Differential(t), Differential(x)
        eq = Dt(u(t, x)) ~ -sin(u(t, x)^2 + 1) * Dx(u(t, x))
        _, disc = standard_1d_disc(; advection = UpwindScheme())

        @test_throws ArgumentError run_boundary_validation(
            build_1d_pdesys(eq, [u(0, x) ~ 0.0, u(t, 0) ~ 0.0]), disc
        )
        @test run_boundary_validation(
            build_1d_pdesys(eq, [u(0, x) ~ 0.0, u(t, 0) ~ 0.0, u(t, 1) ~ 0.0]), disc
        ) === nothing
    end
end
