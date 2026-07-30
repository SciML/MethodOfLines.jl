# PDE problems for the WENO benchmarks: periodic advection, inviscid Burgers, and a
# two-domain interface pulse.

using ModelingToolkit, DomainSets, MethodOfLines, SciMLBase

# Linear advection: u_t = -u_x, periodic on [0, 2].
function advection_system(; t_end = 0.5)
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    eq = Dt(u(t, x)) ~ -Dx(u(t, x))
    bcs = [
        u(0, x) ~ sinpi(x),
        u(t, 0.0) ~ u(t, 2.0),
    ]
    domains = [t ∈ Interval(0.0, t_end), x ∈ Interval(0.0, 2.0)]
    @named weno_advection = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    return (; pdesys = weno_advection, xvar = x, tvar = t, xspan = (0.0, 2.0))
end

function advection_discretization(sys, kind::Symbol, n::Int)
    spec = grid_spec(kind, sys.xspan..., n)
    return MOLFiniteDifference([sys.xvar => spec], sys.tvar; advection_scheme = WENOScheme())
end

# Inviscid Burgers: shock forms at t ≈ 4/π.
function burgers_system(; t_end = 1.5)
    @parameters t x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)

    eq = Dt(u(t, x)) ~ -u(t, x) * Dx(u(t, x))
    bcs = [
        u(0, x) ~ 1.0 + 0.25 * sinpi(x),
        u(t, 0.0) ~ u(t, 2.0),
    ]
    domains = [t ∈ Interval(0.0, t_end), x ∈ Interval(0.0, 2.0)]
    @named weno_burgers = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

    return (; pdesys = weno_burgers, xvar = x, tvar = t, xspan = (0.0, 2.0))
end

function burgers_discretization(sys, kind::Symbol, n::Int)
    spec = grid_spec(kind, sys.xspan..., n)
    return MOLFiniteDifference([sys.xvar => spec], sys.tvar; advection_scheme = WENOScheme())
end

# Two-domain interface pulse; mirrors test/Convection_WENO/MOL_1D_WENO_NU_Interface.jl.
function interface_system(; t_end = 0.5)
    @parameters t x1 x2
    @variables u1(..) u2(..)
    Dt = Differential(t)
    Dx1 = Differential(x1)
    Dx2 = Differential(x2)

    pulse(x_, t_) = exp(-((x_ - t_) - 0.7)^2 / (2 * 0.1^2))

    eqs = [
        Dt(u1(t, x1)) ~ -Dx1(u1(t, x1)),
        Dt(u2(t, x2)) ~ -Dx2(u2(t, x2)),
    ]
    bcs = [
        u1(0, x1) ~ pulse(x1, 0.0),
        u2(0, x2) ~ pulse(x2, 0.0),
        u1(t, 0.0) ~ pulse(0.0, t),
        u1(t, 1.0) ~ u2(t, 1.0),
        Dx2(u2(t, 2.0)) ~ 0.0,
    ]
    domains = [
        t ∈ Interval(0.0, t_end),
        x1 ∈ Interval(0.0, 1.0),
        x2 ∈ Interval(1.0, 2.0),
    ]
    @named weno_interface = PDESystem(
        eqs, bcs, domains, [t, x1, x2], [u1(t, x1), u2(t, x2)]
    )

    return (; pdesys = weno_interface, x1var = x1, x2var = x2, tvar = t)
end

# Seam requires vector grids; interval counts mismatched (n : 3n/2) to exercise the seam.
function interface_discretization(sys, kind::Symbol, n::Int)
    n2 = round(Int, 3 * n / 2)
    if kind === :uniform_vector
        g1 = uniform_grid(0.0, 1.0, n)
        g2 = uniform_grid(1.0, 2.0, n2)
    elseif kind === :stretched
        g1 = stretched_grid(0.0, 1.0, n; amp = 0.03)
        g2 = stretched_grid(1.0, 2.0, n2; amp = 0.04)
    else
        error("interface benchmark supports :uniform_vector and :stretched, got $kind")
    end
    @assert all(diff(g1) .> 0) && all(diff(g2) .> 0)
    return MOLFiniteDifference(
        [sys.x1var => g1, sys.x2var => g2], sys.tvar; advection_scheme = WENOScheme()
    )
end

interface_min_spacing(kind::Symbol, n::Int) = min(
    kind === :uniform_vector ? 1.0 / (n - 1) : minimum(diff(stretched_grid(0.0, 1.0, n; amp = 0.03))),
    kind === :uniform_vector ? 1.0 / (round(Int, 3 * n / 2) - 1) :
        minimum(diff(stretched_grid(1.0, 2.0, round(Int, 3 * n / 2); amp = 0.04))),
)
