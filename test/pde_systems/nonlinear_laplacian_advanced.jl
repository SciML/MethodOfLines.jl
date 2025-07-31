using ModelingToolkit, MethodOfLines, LinearAlgebra, Test, OrdinaryDiffEq, DomainSets
using Symbolics
using Symbolics: wrap, unwrap
using SciMLBase

function halfar_dome(t, x, y, R0, H0, ρ, A = 1e-16)
    n = 3.0
    grav = 9.8101
    alpha = 1.0 / 9.0
    beta = 1.0 / 18.0

    Gamma = 2.0 / (n + 2.0) * A * (ρ * grav)^n

    xcenter = 0.0
    ycenter = 0.0

    t0 = (beta / Gamma) * (7.0 / 4.0)^3 * (R0^4 / H0^7)  # Note: this line assumes n=3!
    tr = (t + t0) / t0

    r = sqrt((x - xcenter)^2 + (y - ycenter)^2)
    r = r / R0
    inside = max(0.0, 1.0 - (r / tr^beta)^((n + 1.0) / n))
    out = H0 * inside^(n / (2.0 * n + 1.0)) / tr^alpha

    return out
end

H0 = 100
R0 = 1000

function asf(dt, dx, dy)
    halfar_dome(dt, dx, dy, R0, H0, 917)
end

@test_broken begin#@testset "Halfar ice dome glacier model." begin
    rmax = 2 * 1000
    rmin = -rmax

    xmin = ymin = rmin
    xmax = ymax = rmax
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

    Γ = 2.0 / (n + 2.0) * A * (ρ * grav)^n

    abs∇H = sqrt(Dx(H(t, x, y))^2 + Dy(H(t, x, y)^2))

    eqs = [Dt(H(t, x, y)) ~
           Dx(Γ * H(t, x, y)^(n + 2) * (abs∇H^(n - 1)) * Dx(H(t, x, y))) +
           Dy(Γ * H(t, x, y)^(n + 2) * (abs∇H^(n - 1)) * Dy(H(t, x, y)))]

    bcs = [H(0.0, x, y) ~ asf(0.0, x, y),
        H(t, xmin, y) ~ 0.0,
        H(t, xmax, y) ~ 0.0,
        H(t, x, ymin) ~ 0.0,
        H(t, x, ymax) ~ 0.0]

    domains = [x ∈ Interval(rmin, rmax),
        y ∈ Interval(rmin, rmax),
        t ∈ Interval(0.0, 1.0e6)]

    @named pdesys = PDESystem(eqs, bcs, domains, [x, y, t], [H(t, x, y)])

    disc = MOLFiniteDifference([x => 24, y => 24], t)

    prob = discretize(pdesys, disc)

    sol = solve(prob, FBDF())

    @test SciMLBase.successful_retcode(sol)

    solx = sol[x]
    soly = sol[y]
    solt = sol[t]

    solexact = [asf(unwrap(dt), unwrap(dx), unwrap(dy))
                for dt in solt, dx in solx, dy in soly]

    @test sum(abs2, sol[H(t, x, y)] .- solexact) < 1e-2
end
