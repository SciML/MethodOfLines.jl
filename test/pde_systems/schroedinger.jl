using MethodOfLines, OrdinaryDiffEq, DomainSets, ModelingToolkit, Test

@testset "Schroedinger" begin
    @parameters t, x
    @variables ψ(..)

    Dt = Differential(t)
    Dxx = Differential(x)^2

    xmin = 0
    xmax = 1

    V(x) = 0.0

    eq = [im * Dt(ψ(t, x)) ~ (Dxx(ψ(t, x)) + V(x) * ψ(t, x))] # You must enclose complex equations in a vector, even if there is only one equation

    ψ0 = x -> ((1 + im) / sqrt(2)) * sinpi(2 * x)

    bcs = [
        ψ(0, x) => ψ0(x), # Initial condition must be marked with a => operator
        ψ(t, xmin) ~ 0,
        ψ(t, xmax) ~ 0,
    ]

    domains = [t ∈ Interval(0, 1), x ∈ Interval(xmin, xmax)]

    @named sys = PDESystem(eq, bcs, domains, [t, x], [ψ(t, x)])

    disc = MOLFiniteDifference([x => 100], t)

    prob = discretize(sys, disc)

    # Use Rodas5P which supports mass matrices (required for complex PDEs in MTK v11)
    sol = solve(prob, Rodas5P(), saveat = 0.01)

    discx = sol[x]
    disct = sol[t]

    discψ = sol[ψ(t, x)]

    analytic(t, x) = sqrt(2) * sin(2 * pi * x) * exp(-im * 4 * pi^2 * t) *
        ((1 + im) / sqrt(2))

    analψ = [analytic(t, x) for t in disct, x in discx]

    for i in 1:length(disct)
        u = abs.(analψ[i, :]) .^ 2
        u2 = abs.(discψ[i, :]) .^ 2

        @test u ./ maximum(u) ≈ u2 ./ maximum(u2) atol = 1.0e-2
    end
    #using Plots

    # anim = @animate for i in 1:length(disct)
    #     u = analψ[i, :]
    #     u2 = discψ[i, :]
    #     plot(discx, [real.(u), imag.(u)], ylim = (-1.5, 1.5), title = "t = $(disct[i])", xlabel = "x", ylabel = "ψ(t,x)", label = ["re(ψ)" "im(ψ)"], legend = :topleft)
    # end
    # gif(anim, "schroedinger.gif", fps = 10)
end

@testset "Schroedinger with complex bcs" begin
    @parameters t, x
    @variables ψ(..)

    Dt = Differential(t)
    Dxx = Differential(x)^2

    xmin = 0
    xmax = 1
    ϵ = 1.0e-2

    V(x) = 0.0

    eq = [(im * ϵ) * Dt(ψ(t, x)) ~ (-0.5 * ϵ^2)Dxx(ψ(t, x)) + V(x) * ψ(t, x)] # You must enclose complex equations in a vector, even if there is only one equation

    ψ0 = x -> exp((im / ϵ) * 1.0e-1 * sum(x))

    bcs = [
        ψ(0, x) => ψ0(x),
        ψ(t, xmin) ~ exp((im / ϵ) * (1.0e-1 * sum(xmin) - 0.5e-2 * t)),
        ψ(t, xmax) ~ exp((im / ϵ) * (1.0e-1 * sum(xmax) - 0.5e-2 * t)),
    ]

    domains = [t ∈ Interval(0, 1), x ∈ Interval(xmin, xmax)]

    @named sys = PDESystem(eq, bcs, domains, [t, x], [ψ(t, x)])

    disc = MOLFiniteDifference([x => 300], t)

    prob = discretize(sys, disc)

    # Use Rodas5P which supports mass matrices (required for complex PDEs in MTK v11)
    sol = solve(prob, Rodas5P(), saveat = 0.01)

    @test SciMLBase.successful_retcode(sol)
end
