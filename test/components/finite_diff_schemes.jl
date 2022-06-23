using ModelingToolkit,
    MethodOfLines, DomainSets, Test, Symbolics, SymbolicUtils, LinearAlgebra

@parameters x, t
@variables u(..)

Dx(d) = Differential(x)^d

Dt = Differential(t)

t_min = 0.0
t_max = 2.0
x_min = 0.0
x_max = 20.0

dx = 1.0

domains = [t ∈ Interval(t_min, t_max), x ∈ Interval(x_min, x_max)]


@testset "Test 01: Cartesian derivative discretization" begin
    weights = []
    push!(weights, ([-0.5, 0, 0.5], [1.0, -2.0, 1.0], [-1 / 2, 1.0, 0.0, -1.0, 1 / 2]))
    push!(
        weights,
        (
            [1 / 12, -2 / 3, 0, 2 / 3, -1 / 12],
            [-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12],
            [1 / 8, -1.0, 13 / 8, 0.0, -13 / 8, 1.0, -1 / 8],
        ),
    )
    for d = 1:3
        for (i, a) in enumerate([2])
            pde = Dt(u(t, x)) ~ +Dx(d)(u(t, x))
            bcs = [u(0, x) ~ cos(x), u(t, 0) ~ exp(-t), u(t, Float64(π)) ~ -exp(-t)]

            @named pdesys = PDESystem(pde, bcs, domains, [t, x], [u(t, x)])

            # Test centered order 
            disc = MOLFiniteDifference([x => dx], t; approx_order = a)

            depvar_ops = map(x -> operation(x.val), pdesys.depvars)

            depvars_lhs = MethodOfLines.get_depvars(pde.lhs, depvar_ops)
            depvars_rhs = MethodOfLines.get_depvars(pde.rhs, depvar_ops)
            depvars = collect(depvars_lhs ∪ depvars_rhs)
            # Read the independent variables,
            # ignore if the only argument is [t]
            indvars = first(Set(filter(xs -> !isequal(xs, [t]), map(arguments, depvars))))
            x̄ = first(
                Set(
                    filter(
                        !isempty,
                        map(
                            u -> filter(
                                x -> t === nothing || !isequal(x, t.val),
                                arguments(u),
                            ),
                            depvars,
                        ),
                    ),
                ),
            )

            s = MethodOfLines.DiscreteSpace(domains, depvars, x̄, disc)

            derivweights = MethodOfLines.DifferentialDiscretizer(pdesys, s, disc)

            #@show pde.rhs, operation(pde.rhs), arguments(pde.rhs)
            for II in s.Igrid[s.ū[1]][2:end-1]
                #II = s.Igrid[end-1]
                I1 = MethodOfLines.unitindices(1)[1]

                rules = MethodOfLines.generate_finite_difference_rules(
                    II,
                    s,
                    depvars,
                    pde,
                    derivweights,
                )

                disc_pde = substitute(pde.lhs, rules) ~ substitute(pde.rhs, rules)
                #@show disc_pde
                #@test disc_pde == dot(weights[i][d], s.discvars[depvars[1]][[II + j*I1 for j in MethodOfLines.half_range(length(weights[i][d]))]])
                ufunc(u, Itap, x) = s.discvars[u][Itap]
                if isodd(d)
                    @test isequal(
                        substitute(pde.rhs, rules),
                        MethodOfLines.upwind_difference(
                            d,
                            II,
                            s,
                            derivweights,
                            (1, x),
                            depvars[1],
                            ufunc,
                            true,
                        ),
                    )
                else
                    @test isequal(
                        substitute(pde.rhs, rules),
                        MethodOfLines.central_difference(
                            derivweights.map[Differential(x)^d],
                            II,
                            s,
                            (1, x),
                            depvars[1],
                            ufunc,
                        ),
                    )
                end
            end
        end
    end
end

@testset "Test 01: Nonlinear Diffusion discretization" begin

    pde = Dt(u(t, x)) ~ Dx(1)(u(t, x))
    bcs = [u(0, x) ~ cos(x), u(t, 0) ~ exp(-t), u(t, Float64(π)) ~ -exp(-t)]

    @named pdesys = PDESystem(pde, bcs, domains, [t, x], [u(t, x)])

    # Test centered order 

    depvar_ops = map(x -> operation(x.val), pdesys.depvars)

    depvars_lhs = MethodOfLines.get_depvars(pde.lhs, depvar_ops)
    depvars_rhs = MethodOfLines.get_depvars(pde.rhs, depvar_ops)
    depvars = collect(depvars_lhs ∪ depvars_rhs)
    # Read the independent variables,
    # ignore if the only argument is [t]
    indvars = first(Set(filter(xs -> !isequal(xs, [t]), map(arguments, depvars))))
    x̄ = first(
        Set(
            filter(
                !isempty,
                map(
                    u -> filter(x -> t === nothing || !isequal(x, t.val), arguments(u)),
                    depvars,
                ),
            ),
        ),
    )

    for order in [2]
        disc = MOLFiniteDifference([x => dx], t; approx_order = order)
        s = MethodOfLines.DiscreteSpace(domains, depvars, x̄, disc)

        derivweights = MethodOfLines.DifferentialDiscretizer(pdesys, s, disc)

        ufunc(u, I, x) = s.discvars[u][I]
        #TODO Test Interpolation of params
        # Test simple case
        for II in s.Igrid[s.ū[1]][2:end-1]
            expr = MethodOfLines.cartesian_nonlinear_laplacian(
                (1 ~ 1).lhs,
                II,
                derivweights,
                s,
                depvars,
                x,
                u(t, x),
            )
            expr2 = MethodOfLines.central_difference(
                derivweights.map[Differential(x)^2],
                II,
                s,
                (1, x),
                u(t, x),
                ufunc,
            )
            @test isequal(expr, expr2)
        end
    end
end

@testset "Test 02: Spherical Diffusion discretization" begin

    pde = Dt(u(t, x)) ~ 1 / x^2 * Dx(1)(x^2 * Dx(1)(u(t, x)))

    bcs = [u(0, x) ~ cos(x), u(t, 0) ~ exp(-t), u(t, Float64(π)) ~ -exp(-t)]

    @named pdesys = PDESystem(pde, bcs, domains, [t, x], [u(t, x)])

    # Test centered order 
    disc = MOLFiniteDifference([x => dx], t; approx_order = 2)

    depvar_ops = map(x -> operation(x.val), pdesys.depvars)

    depvars_lhs = MethodOfLines.get_depvars(pde.lhs, depvar_ops)
    depvars_rhs = MethodOfLines.get_depvars(pde.rhs, depvar_ops)
    depvars = collect(depvars_lhs ∪ depvars_rhs)
    # Read the independent variables,
    # ignore if the only argument is [t]
    indvars = first(Set(filter(xs -> !isequal(xs, [t]), map(arguments, depvars))))
    x̄ = first(
        Set(
            filter(
                !isempty,
                map(
                    u -> filter(x -> t === nothing || !isequal(x, t.val), arguments(u)),
                    depvars,
                ),
            ),
        ),
    )

    s = MethodOfLines.DiscreteSpace(domains, depvars, x̄, disc)

    derivweights = MethodOfLines.DifferentialDiscretizer(pdesys, s, disc)

    for II in s.Igrid[s.ū[1]][2:end-1]
        #TODO Test Interpolation of params
        expr = MethodOfLines.spherical_diffusion(
            (1 ~ 1).lhs,
            II,
            derivweights,
            s,
            depvars,
            x,
            u(t, x),
        )
        #@show II, expr
    end

end
